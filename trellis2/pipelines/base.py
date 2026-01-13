from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, ignore_models: List[str] = None, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        import concurrent.futures

        _models = {}
        if ignore_models is None:
            ignore_models = []

        def load_model(k, v):
            if k in ignore_models:
                return None
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                return None
            try:
                return k, models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                return k, models.from_pretrained(v)

        # Limit workers to avoid I/O or CPU contention. 
        # 6 workers to cover the 5 main heavy models + potential overheads, assuming skip_init reduces memory pressure.
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(load_model, k, v) for k, v in args['models'].items()]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    _models[result[0]] = result[1]

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))