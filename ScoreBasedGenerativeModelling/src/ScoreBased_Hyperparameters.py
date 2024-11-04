import argparse
import copy
import hashlib
import numbers
import math
import os
import sys
import time
from pathlib import Path
from omegaconf import OmegaConf

from pytorch_lightning.loggers import WandbLogger

import wandb


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif type(v) == str:
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
    elif isinstance(v, numbers.Number):
        assert v in [0, 1]
        if v == 1:
            return True
        if v == 0:
            return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid Value: {type(v)}")


def process_hparams(hparams, print_hparams=True):
    # hparams = hparams.parse_args()

    if hparams.logging == "online":
        hparams.show = False

    """Create HParam ID for saving and loading checkpoints"""
    hashable_config = copy.deepcopy(hparams)
    id = hashlib.sha1(
        repr(sorted(hashable_config.__dict__.items())).encode()
    ).hexdigest()  # int -> abs(int) -> str(abs(int)))
    hparams.hparams_id = id

    if print_hparams:
        print(OmegaConf.to_container(hparams, resolve=True))
        # [print(f"\t {key}: {value}") for key, value in sorted(hparams.__dict__.items())]

    return hparams
