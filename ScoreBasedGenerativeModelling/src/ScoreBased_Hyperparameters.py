import argparse
import copy
import hashlib
import numbers
import math
import os
import sys
import time
from pathlib import Path

from pytorch_lightning.loggers import WandbLogger

import wandb

filepath = Path(__file__).absolute()
phd_path = Path(str(filepath).split('PhD')[0] + 'PhD')
jax_path = Path(str(filepath).split('Jax/ScoreBasedGenerativeModelling')[0] + 'Jax/ScoreBasedGenerativeModelling')
sys.path.append(str(phd_path))
sys.path.append(str(jax_path))



# file_path = os.path.dirname(os.path.abspath(__file__)) + '/ContSchNet_HyperparameterParser.py'
# cwd = os.path.dirname(os.path.abspath(__file__)) # current directory PhD/MLMD/src

# sys.path.append("/".join(cwd.split("/")[:-2])) # @PhD: detect MLMD folder


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
	hparams = hparams.parse_args()
	
	if hparams.logging == 'online':
		hparams.show = False
	
	'''Create HParam ID for saving and loading checkpoints'''
	hashable_config = copy.deepcopy(hparams)
	id = hashlib.sha1(repr(sorted(hashable_config.__dict__.items())).encode()).hexdigest()  # int -> abs(int) -> str(abs(int)))
	hparams.hparams_id = id
	
	'''
	Logging
	'''
	if hparams.logging == 'disabled':
		os.environ['WANDB_MODE'] = 'disabled'
	while True:
		try:
			os.environ["WANDB_DISABLE_CODE"] = "false"
			logger = WandbLogger(entity="ludwigwinkler",
			                     project=hparams.project,
			                     name=hparams.experiment,
			                     mode=hparams.logging)
			break
		except:
			print(f"Waiting 2 seconds ...")
			time.sleep(2)
	
	hparams.__dict__.update({"logger": logger})
	wandb.run.log_code(str(phd_path / 'Jax/ScoreBasedGenerativeModelling'))
	
	if print_hparams:
		[print(f"\t {key}: {value}") for key, value in sorted(hparams.__dict__.items())]
	
	return hparams
