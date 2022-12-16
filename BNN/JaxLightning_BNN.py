import torch, numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pl_bolts.datamodules import MNISTDataModule
import matplotlib.pyplot as plt
from typing import Tuple
import einops
import jax, jax.numpy as jnp
import optax, equinox as eqx


def numpy_collate(batch):
	if isinstance(batch[0], np.ndarray):
		return np.stack(batch)
	elif isinstance(batch[0], (tuple, list)):
		transposed = zip(*batch)
		return [numpy_collate(samples) for samples in transposed]
	else:
		return np.array(batch)


class NumpyTensorDataset(Dataset):
	r"""Dataset wrapping tensors.

	Each sample will be retrieved by indexing tensors along the first dimension.

	Args:
		*tensors (Tensor): tensors that have the same size of the first dimension.
	"""
	tensors: Tuple[Tensor, ...]
	
	def __init__(self, *tensors: Tensor) -> None:
		'''
		PyTorch: tensor.size() -> tuple -> tensor.size(0) gives batch dim
		Numpy: nparray.size() is not indexable, so we use shape
		'''
		assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
		self.tensors = tensors
	
	def __getitem__(self, index):
		return tuple(tensor[index] for tensor in self.tensors)
	
	def __len__(self):
		return self.tensors[0].shape[0]


class RegressionDataModule(pl.LightningDataModule):
	num_samples = 50
	x_noise_std = 0.01
	y_noise_std = 0.2
	
	def __init__(self):
		super().__init__()
	
	def setup(self, stage: str) -> None:
		x = jnp.linspace(-0.35, 0.55, self.num_samples)
		
		x_noise = np.random.normal(0., self.x_noise_std, size=x.shape)
		
		std = np.linspace(0, self.y_noise_std, self.num_samples)  # * y_noise_std
		# print(std.shape)
		non_stationary_noise = np.random.normal(loc=0, scale=std)
		y_noise = non_stationary_noise
		
		y = x + 0.3 * jnp.sin(2 * jnp.pi * (x + x_noise)) + 0.3 * jnp.sin(4 * jnp.pi * (x + x_noise)) + y_noise
		
		x, y = x.reshape(-1, 1), y.reshape(-1, 1)
		
		x = (x - x.mean(axis=0)) / x.std(axis=0)
		y = (y - x.mean(axis=0)) / y.std(axis=0)
		
		self.X = np.array(x)
		self.Y = np.array(y)
	
	def train_dataloader(self):
		return DataLoader(NumpyTensorDataset(self.X, self.Y),
		                  shuffle=True,
		                  batch_size=16,
		                  collate_fn=numpy_collate,
		                  drop_last=False)
	
	def val_dataloader(self):
		return DataLoader(TensorDataset(self.X[:100], self.Y[:100]),
		                  shuffle=True,
		                  batch_size=32,
		                  collate_fn=numpy_collate)


class BayesLinear(eqx.Module):
	weight_mu: jax.numpy.ndarray
	weight_rho: jax.numpy.ndarray
	bias_mu: jax.numpy.ndarray
	bias_rho: jax.numpy.ndarray
	
	def __init__(self, in_size, out_size, key):
		wkey, bkey = jax.random.split(key)
		self.weight_mu = jax.random.normal(wkey, (out_size, in_size)) / (out_size + in_size) ** 0.5
		self.weight_rho = jax.numpy.full(shape=self.weight_mu.shape, fill_value=-3)
		self.bias_mu = jax.random.normal(bkey, (out_size,)) * 0.1
		self.bias_rho = jax.numpy.full(shape=self.bias_mu.shape, fill_value=-3)
	
	def __call__(self, x, key):
		w_eps = jax.random.normal(key=key, shape=self.weight_mu.shape)
		b_eps = jax.random.normal(key=key, shape=self.bias_mu.shape)
		w = self.weight_mu + jax.nn.softplus(self.weight_rho) * w_eps
		b = self.bias_mu + jax.nn.softplus(self.bias_rho) * b_eps
		# print(x.shape, w.shape, b.shape)
		
		return x @ w.T + b
	
	def kl_div(self):
		weight_scale = jax.nn.softplus(self.weight_rho)
		kl_div = jnp.log(1.) - jnp.log(weight_scale)
		kl_div += (weight_scale ** 2 + (self.weight_mu - 0) ** 2) / (2) - 0.5
		return kl_div.sum()


class BNN(eqx.Module):
	bnn: eqx.nn.Sequential
	
	def __init__(self, key):
		hidden = 51
		self.bnn = eqx.nn.Sequential([BayesLinear(1, hidden, key), eqx.nn.Lambda(jax.nn.gelu),
		                              BayesLinear(hidden, hidden, key), eqx.nn.Lambda(jax.nn.gelu),
		                              BayesLinear(hidden, hidden, key), eqx.nn.Lambda(jax.nn.gelu),
		                              BayesLinear(hidden, hidden, key), eqx.nn.Lambda(jax.nn.gelu),
		                              BayesLinear(hidden, 1, key)])
	
	def __call__(self, x, key):
		return self.bnn(x, key=key)
	
	def kl_div(self):
		kl_div = 0
		for layer in self.bnn.layers:
			if type(layer) == BayesLinear:
				kl_div += layer.kl_div()
		return kl_div


class JaxLightning(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.automatic_optimization = False
		self.key = jax.random.PRNGKey(1)
		self.MC = 5
		
		self.key, subkey = jax.random.split(self.key)
		self.bnn = BNN(key=subkey)
		self.global_step_ = 0
	
	@property
	def global_step(self):
		'''
		self.global_step is an attribute without setter and is updated somewhere deep within Lightning
		Simply overwrite global_step as a property to access it normally in the LightningModule
		:return:
		'''
		return self.global_step_
	
	def on_fit_start(self) -> None:
		self.num_data_samples = self.trainer.datamodule.num_samples
		self.viz_network('On Train Start')
		
	
	def training_step(self, batch):
		self.global_step_ += 1
		data, target = jnp.array(batch[0].reshape(-1, *batch[0].shape[1:])), jnp.array(batch[1])
		data = einops.repeat(data, '... -> b ...', b=self.MC)
		target = einops.repeat(target, '... -> b ...', b=self.MC)
		
		self.key, *subkeys = jax.random.split(self.key, num=self.MC + 1)  # creating new keys
		subkeys = jnp.stack(subkeys)
		
		loss, metrics, self.bnn, self.optim, self.opt_state = JaxLightning.make_step(self.bnn,
		                                                                             data,
		                                                                             target,
		                                                                             self.num_data_samples,
		                                                                             subkeys,
		                                                                             self.optim,
		                                                                             self.opt_state)
		dict = {'loss': torch.from_numpy(np.array(loss)), 'global_step': self.global_step, 'current_epoch': self.current_epoch
		        # **{key: torch.from_numpy(np.array(value)).item() for key, value in metrics.items()}
		        }
		self.log_dict(dict, prog_bar=True)
		return dict
	
	def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
		'''
		In automatic_optimization=False mode, the global_step attribute is (for some reason)  not automatically incremented, so we have to use our own should_stop criterion
		'''
		if self.global_step >= self.trainer.max_steps and self.trainer.max_steps > 0:
			self.trainer.should_stop = True
		
	def on_train_end(self) -> None:
		self.viz_network(title='On Fit End')
	
	def viz_network(self, title=''):
		
		MC = 201
		
		self.key, *subkeys = jax.random.split(self.key, num=MC + 1)  # creating new keys
		subkeys = jnp.stack(subkeys)
		pred_model = jax.jit(jax.vmap(self.bnn, in_axes=(0, 0)))
		
		viz_input = jnp.linspace(-4, 4, 100).reshape(-1, 1)
		x_mc = einops.repeat(viz_input, '... -> b ...', b=MC)
		print(x_mc.shape, len(subkeys))
		mc_pred = pred_model(x_mc, subkeys)
		
		fig = plt.figure(figsize=(10, 10))
		for pred in mc_pred.squeeze():
			plt.plot(viz_input, pred, color='red', alpha=1 / min(mc_pred.shape[0], 50.))
		X, Y = self.trainer.datamodule.X, self.trainer.datamodule.Y
		plt.scatter(X, Y, alpha=1 / min(mc_pred.shape[0], 1.), s=5)
		plt.ylim(-3, 3)
		plt.title(title)
		plt.show()
	
	def configure_optimizers(self):
		self.optim = optax.adam(0.001)
		self.opt_state = self.optim.init(eqx.filter(self.bnn, eqx.is_array))
	
	@staticmethod
	@eqx.filter_value_and_grad(has_aux=True)
	def criterion(model, x, y, num_samples, keys):
		model_vmap = jax.vmap(model, in_axes=(0, 0))  # takes [MC, B, F] features and [MC] keys
		pred = model_vmap(x, keys)
		assert pred.ndim == 3
		std = jax.lax.stop_gradient(pred.std(axis=0))
		# std = 0.01
		mse = (y - pred.mean(axis=0)) ** 2
		nll = -(-0.5 * mse / std ** 2).sum() * num_samples
		kl = 1. * model.kl_div()
		return (nll + kl, {'nll': nll, 'kl': kl, 'std': std.mean(), 'mse': mse.mean()})
	
	@staticmethod
	@eqx.filter_jit
	def make_step(model, x, y, num_samples, keys, optim, opt_state):
		(loss, metrics), grads = JaxLightning.criterion(model, x, y, num_samples, keys)
		updates, opt_state = optim.update(grads, opt_state)
		model = eqx.apply_updates(model, updates)
		return loss, metrics, model, optim, opt_state


print(jax.devices())

bnn = JaxLightning()
dm = RegressionDataModule()
trainer = Trainer(max_steps=4000)
trainer.fit(bnn, dm)
