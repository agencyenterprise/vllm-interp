# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterable
from itertools import islice
from typing import Optional, Union

import torch
from torch import nn
from transformers import Gemma2Config
# this is added for input signature
from vllm.inputs import InterventionInputs
from vllm.model_executor.interp import InterpCodec, apply_steering, init_codec_for_rank

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)


class Gemma2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if not (hidden_act == hidden_activation == "gelu_pytorch_tanh"):
            raise ValueError(
                "Gemma2 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma2Attention(nn.Module):

    def __init__(self,
                 config: Gemma2Config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 rope_theta: float,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 attn_logits_soft_cap: Optional[float] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
        )

        layer_idx = extract_layer_index(prefix)
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        sliding_window = config.sliding_window if is_sliding else None

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              logits_soft_cap=attn_logits_soft_cap,
                              per_layer_sliding_window=sliding_window,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        cached_codecs: Optional[dict[int, InterpCodec]] = None,
    ) -> None:
        super().__init__()
        self.cached_codecs = cached_codecs
        self.tp_rank = get_tp_group().local_rank

        self.hidden_size = config.hidden_size
        self.self_attn = Gemma2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=config.attn_logit_softcapping,
            prefix=f"{prefix}.self_attn",
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        intervention_enabled: bool,
        intervention_list: Optional[list[InterventionInputs]],
        steer_positions: Optional[list[int]],
        feature_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        # run intervention if enabled in the layer and if the codec is initialized
        if intervention_enabled and self.cached_codecs is not None:
            result = apply_steering(
                self.cached_codecs[self.tp_rank], hidden_states,
                intervention_list, steer_positions,
            )
            hidden_states = result.hidden_states
            add_tensor = result.add_tensor_projected
        else:
            add_tensor = None
        # get the feature readout if enabled in the layer and if the codec is initialized
        if feature_enabled and self.cached_codecs is not None:
            feature_tensor = self.cached_codecs[self.tp_rank].encode(hidden_states)
        else:
            feature_tensor = None
        return hidden_states, residual, add_tensor, feature_tensor


@support_torch_compile
class Gemma2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.steering_layer = vllm_config.model_config.steering_layer
        self.feature_layer = vllm_config.model_config.feature_layer
        codec_type = vllm_config.model_config.codec_type
        if codec_type is not None:
            self.cached_codecs = init_codec_for_rank(
                codec_type,
                sae_name=vllm_config.model_config.sae_name,
                sae_filepath=vllm_config.model_config.sae_filepath,
                hidden_size=vllm_config.model_config.hidden_size,
                sae_expansion_factor=vllm_config.model_config.sae_expansion_factor,
                sae_release=vllm_config.model_config.sae_release,
                sae_id=vllm_config.model_config.sae_id,
                directions_filepath=vllm_config.model_config.directions_filepath,
            )
        else:
            self.cached_codecs = None

        # this is the scale factor to subtract the steered tensor from the hidden states
        self.scale_factor_add_tensor = vllm_config.model_config.steering_scale_factor
        # this is the add tensor for the intervention
        self.add_tensor = None
        # this is the feature tensor for the feature readout
        self.feature_tensor = None
        # this is the activations tensor for pulling activations from the model
        self.activations_tensor = None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma2DecoderLayer(
                config, cache_config, quant_config, prefix=prefix, cached_codecs=self.cached_codecs),
            prefix=f"{prefix}.layers")
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer",
                             torch.tensor(normalizer),
                             persistent=False)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        interventions: Optional[list[InterventionInputs]] = None,
        steer_positions: Optional[list[int]] = None,
        get_activations_layer: Optional[set[int]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            hidden_states *= self.normalizer
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        # set activations tensor to empty dictionary
        self.activations_tensor = {}
        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            if idx == self.steering_layer:
                intervention_enabled = True
            else:
                intervention_enabled = False
            if idx == self.feature_layer:
                feature_enabled = True
            else:
                feature_enabled = False
            hidden_states, residual, add_tensor, feature_tensor = layer(
                positions,
                hidden_states,
                residual,
                intervention_enabled=intervention_enabled,
                intervention_list=interventions,
                steer_positions=steer_positions,
                feature_enabled=feature_enabled,
            )
            if add_tensor is not None:
                self.add_tensor = add_tensor
            if feature_tensor is not None:
                self.feature_tensor = feature_tensor
            if idx in get_activations_layer:
                self.activations_tensor[idx] = hidden_states.detach().cpu()
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        # subtract the add tensor from the hidden states before running final norm
        if self.add_tensor is not None:
            hidden_states -= self.add_tensor * self.scale_factor_add_tensor
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, self.feature_tensor, self.activations_tensor

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class Gemma2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        del lora_config  # Unused.
        super().__init__()
        self.config = config
        # currently all existing Gemma models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.quant_config = quant_config
        self.model = Gemma2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        # need to add the following for compatibility with regular vllm setup
        interventions: Optional[list[InterventionInputs]] = None,
        steer_positions: Optional[list[int]] = None,
        # get_activations_layer is extra input for pulling activations from the model
        get_activations_layer: Optional[set[int]] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states, feature_tensor, activations_tensor = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, interventions, steer_positions, get_activations_layer)
        return hidden_states, feature_tensor, activations_tensor

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
