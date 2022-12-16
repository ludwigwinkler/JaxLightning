import sys, argparse
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
import jax, jax.numpy as jnp, equinox as eqx, optax, jax.random as jr
import diffrax as dfx
import functools as ft
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import torch, einops


filepath = Path().resolve()
phd_path = Path(str(filepath).split('PhD')[0] + 'PhD')
jax_path = Path(str(filepath).split('Jax')[0] + 'Jax')
score_path = Path(str(filepath).split('Jax')[0] + 'Jax/ScoreBasedGenerativeModelling')
sys.path.append(str(phd_path))
sys.path.append(str(jax_path))
from Jax.ScoreBasedGenerativeModelling.src.ScoreBased_Data import MNISTDataModule
from Jax.ScoreBasedGenerativeModelling.src.ScoreBased_Models import Mixer2d
from Jax.ScoreBasedGenerativeModelling.src.ScoreBased_Hyperparameters import str2bool, process_hparams


class JaxLightning(pl.LightningModule):
	def __init__(self, **wkargs):
		super().__init__()
		self.save_hyperparameters()
		self.automatic_optimization = False
		
		self.key = jax.random.PRNGKey(1)
		self.key, self.model_key, self.train_key, self.loader_key, self.sample_key = jax.random.split(self.key, 5)
		
		self.key, subkey = jax.random.split(self.key)
		self.model = Mixer2d((1, 28, 28),
		                     patch_size=4,
		                     hidden_size=64,
		                     mix_patch_size=512,
		                     mix_hidden_size=512,
		                     num_blocks=4,
		                     t1=10,
		                     key=subkey)
		
		self.t1 = 10.
		self.int_beta = lambda t: t
		self.weight = lambda t: 1 - jnp.exp(-self.int_beta(t))
		self.dt0 = 0.1
		self.samples = 10
		
		self.global_step_ = 0
		
		self.configure_optimizers()
	
	@staticmethod
	def args(parent_parser,
	         logging='online',
	         load_last_checkpoint=True,
	         fast_dev_run=str(0),
	         show=False,
	         save_figs=False,
	         watch_model_training=True,
	         project='testing1234',
	         experiment='nd',
	         seed=12345):
		parser = parent_parser.add_argument_group("Experiment")
		parser.add_argument("--logging", type=str, default=logging)
		parser.add_argument("-f", type=str, default='')
		parser.add_argument("--project", type=str, default=project)
		parser.add_argument("--experiment", type=str, default=experiment)
		parser.add_argument("--load_last_checkpoint", type=str2bool, default=load_last_checkpoint)
		parser.add_argument("--show", type=str, default=show)
		parser.add_argument("--save_figs", type=str2bool, default=save_figs)
		parser.add_argument("--fast_dev_run", type=str2bool, default=fast_dev_run)
		parser.add_argument("--seed", type=int, default=seed)
		return parent_parser
	
	def on_fit_start(self) -> None:
		pathlib.Path.mkdir(phd_path / f'Jax/checkpoints/ScoreBased', parents=True, exist_ok=True)
		if self.hparams.load_last_checkpoint:
			try:
				self.model = eqx.tree_deserialise_leaves(phd_path / f'Jax/checkpoints/ScoreBased/last.eqx', self.model)
				print('Loaded weights')
			except:
				print('Didnt load weights')
		
		self.logger.experiment.log({"Samples P1": [wandb.Image(self.sample(), caption="Samples P1")]},
		                           step=self.global_step_)
	
	def on_fit_end(self):
		pathlib.Path.mkdir(phd_path / f'Jax/checkpoints/ScoreBased', parents=True, exist_ok=True)
		eqx.tree_serialise_leaves(phd_path / f"Jax/checkpoints/ScoreBased/last.eqx", self.model)
	
	def training_step(self, batch):
		data = batch[0]
		value, self.model, self.train_key, self.opt_state = JaxLightning.make_step(self.model,
		                                                                           self.weight,
		                                                                           self.int_beta,
		                                                                           data,
		                                                                           self.t1,
		                                                                           self.train_key,
		                                                                           self.opt_state,
		                                                                           self.optim.update)
		dict = {'loss': torch.scalar_tensor(value.item())}
		self.log_dict(dict, prog_bar=True)
		self.global_step_ += 1
		return dict
	
	def sample(self):
		self.sample_key, *sample_key = jr.split(self.sample_key, self.samples ** 2 + 1)
		sample_key = jnp.stack(sample_key)
		sample_fn = ft.partial(JaxLightning.single_sample_fn, self.model, self.int_beta, (1, 28, 28), self.dt0, self.t1)
		sample = jax.vmap(sample_fn)(sample_key)
		# sample = data_mean + data_std * sample
		# sample = jnp.clip(sample, data_min, data_max)
		sample = einops.rearrange(sample, "(n1 n2) 1 h w -> (n1 h) (n2 w)", n1=self.samples, n2=self.samples)
		fig = plt.figure()
		plt.imshow(sample, cmap="Greys")
		plt.axis("off")
		plt.title(f'{self.global_step_}')
		plt.tight_layout()
		if self.hparams.show:
			plt.show()
		return fig
	
	def on_train_epoch_end(self) -> None:
		pathlib.Path.mkdir(phd_path / f'Jax/checkpoints/ScoreBased', parents=True, exist_ok=True)
		eqx.tree_serialise_leaves(phd_path / f"Jax/checkpoints/ScoreBased/last.eqx", self.model)
		self.logger.experiment.log({"Samples P1": [wandb.Image(self.sample(), caption="Samples P1")]},
		                           step=self.global_step_)
	
	
	def configure_optimizers(self):
		self.optim = optax.adam(3e-4)
		self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))
	
	@staticmethod
	@eqx.filter_jit
	def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
		'''
		Sampling a single trajectory starting from normal noise at t1 and recovering data distribution at t0
		:param model:
		:param int_beta:
		:param data_shape:
		:param dt0:
		:param t1:
		:param key:
		:return:
		'''
		
		def drift(t, y, args):
			'''
			compute time derivative of function dß(t)/dt
			Noising SDE: dx(t) = -1/2 ß(t) x(t) dt + ß(t)^1/2 dW_t -> μ(x(t)) = - 1/2 ß(t) x(t) dt and σ^2 = ß(t)
			Reverse SDE: μ(x(tau)) = 1/2 ß(t) x(t) + ß(t) ∇ log p
			:param t:
			:param y:
			:param args:
			:return:
			'''
			_, beta = jax.jvp(fun=int_beta, primals=(t,), tangents=(jnp.ones_like(t),))
			return -0.5 * beta * (y + model(t, y))  # negative because we use -dt0 when solving
		
		term = dfx.ODETerm(drift)
		solver = dfx.Tsit5()
		t0 = 0
		y1 = jr.normal(key, data_shape)  # noise at t1, from which integrate backwards to data distribution
		# reverse time, solve from t1 to t0
		sol = dfx.diffeqsolve(term=term, solver=solver, t0=t1, t1=t0, dt0=-dt0, y0=y1, adjoint=dfx.NoAdjoint())
		return sol.ys[0]
	
	@staticmethod
	def single_loss_fn(model, weight, int_beta, data, t, key):
		'''
		OU process provides analytical mean and variance
		int_beta(t) = ß = θ
		E[X_t] = μ + exp[-θ t] ( X_0 - μ) w/ μ=0 gives =X_0 * exp[ - θ t ]
		V[X_t] = σ^2/(2θ) ( 1 - exp(-2 θ t) ) w/ σ^2=ß=θ gives = 1 - exp(-2 ß t)
		:param model:
		:param weight:
		:param int_beta:
		:param data:
		:param t:
		:param key:
		:return:
		'''
		mean = data * jnp.exp(-0.5 * int_beta(t))  # analytical mean of OU process
		var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)  # analytical variance of OU process
		std = jnp.sqrt(var)
		noise = jr.normal(key, data.shape)
		y = mean + std * noise
		pred = model(t, y)
		return weight(t) * jnp.mean((pred + noise / std) ** 2)  # loss
	
	@staticmethod
	def batch_loss_fn(model, weight, int_beta, data, t1, key):
		batch_size = data.shape[0]
		tkey, losskey = jr.split(key)
		losskey = jr.split(losskey, batch_size)
		'''
		Low-discrepancy sampling over t to reduce variance
		by sampling very evenly by sampling uniformly and independently from (t1-t0)/batch_size bins
		t = [U(0,1), U(1,2), U(2,3), ...]
		'''
		t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
		t = t + (t1 / batch_size) * jnp.arange(batch_size)
		''' Fixing the first three arguments of single_loss_fn, leaving data, t and key as input '''
		loss_fn = ft.partial(JaxLightning.single_loss_fn, model, weight, int_beta)
		loss_fn = jax.vmap(loss_fn)
		return jnp.mean(loss_fn(data, t, losskey))
	
	@staticmethod
	@eqx.filter_jit
	def make_step(model, weight, int_beta, data, t1, key, opt_state, opt_update):
		loss_fn = eqx.filter_value_and_grad(JaxLightning.batch_loss_fn)
		loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
		updates, opt_state = opt_update(grads, opt_state)
		model = eqx.apply_updates(model, updates)
		key = jr.split(key, 1)[0]
		return loss, model, key, opt_state


hparams = argparse.ArgumentParser()
hparams = JaxLightning.args(hparams,
                            logging=['disabled', 'online'][0, 1, 1 if torch.cuda.is_available() else 0][-1],
                            project='jax_score_based',
                            load_last_checkpoint=True,
                            show=True,
                            seed=[12345, 2345, 98765][1],
                            fast_dev_run=0,
                            save_figs=[False, True][0],
                            watch_model_training=[False, True][0])

temp_args, _ = hparams.parse_known_args()

hparams = process_hparams(hparams, print_hparams=True)

dm = MNISTDataModule()
dm.prepare_data()
dm.setup()

scorebased = JaxLightning(**vars(hparams))

trainer = Trainer.from_argparse_args(hparams, max_steps=1_000_000, accelerator='cpu')
trainer.fit(scorebased, dm)
