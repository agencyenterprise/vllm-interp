import asyncio
import time
import uuid
from typing import List, Optional
import os

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokenInputs, InterventionInputs
from vllm.model_executor.models.llama_models_and_saes import (
    llama_models_and_saes,
)
from vllm.model_executor.models.goodfire_sae import SparseAutoEncoder, load_sae

load_dotenv()


class ChatMessage(BaseModel):
    """Chat message used for the chat template."""
    role: str
    content: str


class InterventionSpec(BaseModel):
    """One steering intervention: either an SAE feature_id or a pre-computed vector_id."""
    feature_id: Optional[int] = None
    vector_id: Optional[int] = None
    value: float
    mode: Optional[str] = "add"  # "add" or "clamp"


class GenerateRequest(BaseModel):
    """Request payload for generation, including streaming and SAE interventions."""
    prompt: List[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 1024
    repetition_penalty: float = 1.0
    seed: Optional[int] = 324
    intervention: Optional[List[InterventionSpec]] = Field(default=None, description="List of interventions to apply")
    stream: bool = False


class GenerateResponse(BaseModel):
    """generation response."""
    text: str
    num_tokens: int
    request_time_s: float


app = FastAPI(title="vllm-interp Server")


# Global engine/tokenizer initialized at startup
engine: Optional[AsyncLLMEngine] = None
tokenizer = None
# Optional in-memory SAE used for LLaMA models for feature readout routes
cached_sae: Optional[SparseAutoEncoder] = None

# Reuse the same model configuration as the async example
MODEL_STR = "meta-llama/Meta-Llama-3.3-70B-Instruct"
MODEL_CONFIG = llama_models_and_saes[MODEL_STR]


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the vLLM engine/tokenizer and optionally preload the SAE."""
    global engine, tokenizer, cached_sae
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL_CONFIG["model_id"],
            dtype=torch.bfloat16,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            # these are for goodfire SAE
            tensor_parallel_size=torch.cuda.device_count(),
            sae_name=MODEL_CONFIG["sae_id"],
            sae_filepath=MODEL_CONFIG["sae_filepath"],
            hidden_size=MODEL_CONFIG["d_model"],
            sae_expansion_factor=MODEL_CONFIG["expansion_factor"],
            steering_layer=MODEL_CONFIG["steering_layer"],
            feature_layer=MODEL_CONFIG["feature_layer"],
            # quantization config
            quantization=MODEL_CONFIG["quantization"],
            # disable prefix caching for consistent steering
            enable_prefix_caching=False,
        )
    )
    tokenizer = await engine.get_tokenizer()
    # if "llama" in MODEL_STR:
    #     cached_sae = load_sae(MODEL_CONFIG["sae_id"], MODEL_CONFIG["sae_filepath"], MODEL_CONFIG["d_model"], MODEL_CONFIG["expansion_factor"], torch.device("cuda:0"))

    #     print("cached_sae", cached_sae)


@app.get("/health")
async def health() -> dict:
    if engine is None:
        return {"status": "initializing"}
    return {"status": "ok", "model": MODEL_STR}

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

def _sse_format(event: str, data: str) -> str:
    """Format a Server-Sent Event (SSE) line."""
    return f"event: {event}\n" + f"data: {data}\n\n"


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate text from chat messages with optional SAE feature interventions.

    - If `stream=True`, returns an SSE stream of incremental tokens.
    - Otherwise, returns the final text and token count.
    """
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Engine not initialized yet")

    st = time.time()

    messages = [{"role": m.role, "content": m.content} for m in req.prompt]

    # Convert messages to prompt token IDs using the model's chat template
    prompt_token_ids: List[int] = tokenizer.apply_chat_template(messages)
    # Ensure the assistant header is present (Llama 3 style) so the model continues as the assistant
    prompt_token_ids.extend([128000, 128006, 78191, 128007])

    token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=messages)

    if req.intervention is not None:
        # Package interventions (SAE features and/or pre-computed vectors)
        intervention_list = []
        for iv in req.intervention:
            entry: dict = {"value": iv.value, "mode": iv.mode}
            if iv.vector_id is not None:
                entry["vector_id"] = iv.vector_id
            if iv.feature_id is not None:
                entry["feature_id"] = iv.feature_id
            intervention_list.append(entry)
        interventions = InterventionInputs(intervention=intervention_list)
    else:
        interventions = None

    # Launch asynchronous generation; `is_feature_decode=False` selects normal text decoding
    results_generator = engine.generate(
        prompt=token_inputs,
        sampling_params=SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            repetition_penalty=req.repetition_penalty,
            seed=req.seed,
        ),
        request_id=str(uuid.uuid4()),
        interventions=interventions,
        # this is false for text generation
        is_feature_decode=False,
    )

    if req.stream:
        # Stream token deltas as SSE until completion, then send the final text
        async def event_publisher():
            previous_text = ""
            async for request_output in results_generator:
                if not request_output.outputs:
                    continue
                current_text = request_output.outputs[0].text
                if len(current_text) > len(previous_text):
                    delta_text = current_text[len(previous_text):]
                    previous_text = current_text
                    yield _sse_format("token", delta_text)
            yield _sse_format("done", previous_text)

        return StreamingResponse(
            event_publisher(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Accumulate the final output from the async generator
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None or not final_output.outputs:
            raise HTTPException(status_code=500, detail="No output generated")

        text_output = final_output.outputs[0].text
        num_tokens = len(final_output.outputs[0].token_ids)

        return GenerateResponse(
            text=text_output,
            num_tokens=num_tokens,
            request_time_s=time.time() - st,
        )


if __name__ == "__main__":
    import uvicorn
    # Local development entrypoint
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


