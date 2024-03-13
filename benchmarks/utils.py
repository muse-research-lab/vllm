import os

OPT_MODELS = {
    "opt-125m",
    "opt-350m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
}

LLAMA_MODELS = {
    "Llama-2-7b",
    "Llama-2-13b",
}

MISTRAL_MODELS = {
    "Mistral-7B-v0.1"
}

FALCON_MODELS = {
    "falcon-7b"
}

def get_model_name(model: str) -> str:
    for model_name in OPT_MODELS.union(LLAMA_MODELS, MISTRAL_MODELS, FALCON_MODELS):
        if model_name in model:
            return model_name
    
    raise ValueError(f"Unknown model: {model}")


def get_dataset(dataset_name, model_name) -> str:
    dataset_path = os.path.abspath(os.path.join(os.getcwd(), "datasets"))
    
    if model_name in OPT_MODELS:
        if model_name == "opt-6.7b":
            dataset_path = os.path.join(dataset_path, "opt-6.7")
        else:
            dataset_path = os.path.join(dataset_path, "opt")
    elif model_name in LLAMA_MODELS:
        dataset_path = os.path.join(dataset_path, "llama")
    elif model_name in MISTRAL_MODELS:
        dataset_path = os.path.join(dataset_path, "mistral")
    elif model_name in FALCON_MODELS:
        dataset_path = os.path.join(dataset_path, "falcon")
    else:
        raise ValueError(f"Unknown dataset {dataset_name} for model {model_name}")
    
    dataset_path = os.path.join(dataset_path, f"{dataset_name}_data.pkl")
    if dataset_name not in ["alpaca", "sharegpt", "cnn_dailymail", "dolly"]:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_path
    
def get_dataset_name(dataset: str) -> str:
    if "sharegpt" in dataset.lower():
        return "sharegpt"
    elif "alpaca" in dataset.lower():
        return "alpaca"
    elif "cnn_dailymail" in dataset.lower():
        return "cnn_dailymail"
    elif "dolly" in dataset.lower():
        return "dolly"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
def get_sampling_dir_name(
    n1: float,
    n2: float,
    n3: float,
    n4: float,
    n6: float,
    n2_beam: float,
    n4_beam: float,
    n6_beam: float,
    n8_beam: float,
) -> str:
    method = ""
    if n1 > 0.0:
        method = "n1" if n1 == 1.0 else method + f"n1-{n1}-"
    if n2 > 0.0:
        method = "n2" if n2 == 1.0 else method + f"n2-{n2}-"
    if n3 > 0.0:
        method = "n3" if n3 == 1.0 else method + f"n3-{n3}-"
    if n4 > 0.0:
        method = "n4" if n4 == 1.0 else method + f"n4-{n4}-"
    if n6 > 0.0:
        method = "n6" if n6 == 1.0 else method + f"n6-{n6}-"
    if n2_beam > 0.0:
        method = "n2-beam" if n2_beam == 1.0 else method + f"n2-beam-{n2_beam}-"
    if n4_beam > 0.0:
        method = "n4-beam" if n4_beam == 1.0 else method + f"n4-beam-{n4_beam}-"
    if n6_beam > 0.0:
        method = "n6-beam" if n6_beam == 1.0 else method + f"n6-beam-{n6_beam}-"
    if n8_beam > 0.0:
        method = "n8-beam" if n8_beam == 1.0 else method + f"n8-beam-{n8_beam}-"
    return method[:-1] if method.endswith("-") else method