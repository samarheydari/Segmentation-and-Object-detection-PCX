import torch

from .pidnet import get_pidnet
# from .ssd import get_ssd
MODELS = {
    # object detectors
    
    "pidnet": get_pidnet, 
}

def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name in MODELS:
        model = MODELS[model_name](**kwargs)
        return model
    else:
        print(f"Model {model_name} not available")
        exit()