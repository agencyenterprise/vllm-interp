
gemma_models_and_saes = {
    "google/gemma-2-2b-it-l16": {
        "model_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_16/width_16k/canonical",
        "labels_url": "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/gemma-2-2b/16-gemmascope-res-16k/explanations/",
        "steering_layer": 16,
        "feature_layer": 16,
        "threshold_prior_mean": 5,
        "threshold_prior_std": 5,
        "threshold_upper_bound": 100,
        "tensor_parallel_size": 1,
        "quantization" : None,
    },
}