"""
Module handling Distributed Data Parallel issues.
"""

from typing import Mapping, Any

import torch


class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """
    Useful to get access to method of models wrapped by DataParallel or transfer learning from model that was not using DDP
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        """
        Overwriting of load_state_dict to load state dict for DDP even if it comes from a non-DDP training.
        """
        keys = list(state_dict.keys())
        for key in keys:
            if key.split(".")[0] != "module":
                new_key = f"module.{key}"
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        super().load_state_dict(state_dict, strict)



def convert_checkpoint_from_ddp(checkpoint: dict, state_dict_name: str) -> dict:
    """
    Convert a checkpoint to make it usable without DDP even if it comes from a DDP training.
    """
    keys = list(checkpoint[state_dict_name].keys())
    for key in keys:
        checkpoint[state_dict_name][key.split("module.")[1]] = checkpoint[state_dict_name][key]
        del checkpoint[state_dict_name][key]
    return checkpoint