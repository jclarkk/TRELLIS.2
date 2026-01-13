import importlib
import torch

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Context manager to skip initialization
    class no_init:
        def __enter__(self):
            def skip_init(module, *args, **kwargs):
                pass
            self.original_inits = {}
            for name in ['normal_', 'uniform_', 'constant_', 'xavier_normal_', 'xavier_uniform_', 'kaiming_normal_', 'kaiming_uniform_', 'zeros_', 'ones_', 'eye_', 'dirac_', 'orthogonal_']:
                if hasattr(torch.nn.init, name):
                    self.original_inits[name] = getattr(torch.nn.init, name)
                    setattr(torch.nn.init, name, skip_init)
        
        def __exit__(self, exc_type, exc_value, traceback):
            for name, func in self.original_inits.items():
                setattr(torch.nn.init, name, func)

    with no_init():
        model = __getattr__(config['name'])(**config['args'], **kwargs)
    
    state_dict = load_file(model_file)
    try:
        model.load_state_dict(state_dict, strict=False, assign=True)
    except Exception as e:
        # Fallback for older PyTorch versions or if assign=True fails
        # If parameters are on meta device, we need to materialize them first
        is_meta = any(p.device.type == 'meta' for p in model.parameters())
        if is_meta:
            print(f"[WARNING] Model initialized on meta device. Materializing to CPU before loading. Error with assign=True: {e}")
            model.to_empty(device='cpu')
        
        model.load_state_dict(state_dict, strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
