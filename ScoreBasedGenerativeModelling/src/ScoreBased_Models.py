import einops
import equinox as eqx
import jax
from flax import linen as fnn
from jax import numpy as jnp, random as jr


class Encoder(fnn.Module):
	features: int = 64
	training: bool = True
	
	@fnn.compact
	def __call__(self, x):
		z1 = fnn.Conv(self.features, kernel_size=(3, 3))(x)
		z1 = fnn.relu(z1)
		z1 = fnn.Conv(self.features, kernel_size=(3, 3))(z1)
		z1 = fnn.BatchNorm(use_running_average=not self.training)(z1)
		z1 = fnn.relu(z1)
		z1_pool = fnn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))
		
		z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z1_pool)
		z2 = fnn.relu(z2)
		z2 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z2)
		z2 = fnn.BatchNorm(use_running_average=not self.training)(z2)
		z2 = fnn.relu(z2)
		z2_pool = fnn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))
		
		z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z2_pool)
		z3 = fnn.relu(z3)
		z3 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z3)
		z3 = fnn.BatchNorm(use_running_average=not self.training)(z3)
		z3 = fnn.relu(z3)
		z3_pool = fnn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))
		
		z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z3_pool)
		z4 = fnn.relu(z4)
		z4 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z4)
		z4 = fnn.BatchNorm(use_running_average=not self.training)(z4)
		z4 = fnn.relu(z4)
		z4_dropout = fnn.Dropout(0.5, deterministic=False)(z4)
		z4_pool = fnn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))
		
		z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z4_pool)
		z5 = fnn.relu(z5)
		z5 = fnn.Conv(self.features * 16, kernel_size=(3, 3))(z5)
		z5 = fnn.BatchNorm(use_running_average=not self.training)(z5)
		z5 = fnn.relu(z5)
		z5_dropout = fnn.Dropout(0.5, deterministic=False)(z5)
		
		return z1, z2, z3, z4_dropout, z5_dropout


class Decoder(fnn.Module):
	features: int = 64
	training: bool = True
	
	@fnn.compact
	def __call__(self, z1, z2, z3, z4_dropout, z5_dropout):
		z6_up = jax.image.resize(z5_dropout,
		                         shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2,
		                                z5_dropout.shape[3]),
		                         method='nearest')
		z6 = fnn.Conv(self.features * 8, kernel_size=(2, 2))(z6_up)
		z6 = fnn.relu(z6)
		z6 = jnp.concatenate([z4_dropout, z6], axis=3)
		z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
		z6 = fnn.relu(z6)
		z6 = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z6)
		z6 = fnn.BatchNorm(use_running_average=not self.training)(z6)
		z6 = fnn.relu(z6)
		
		z7_up = jax.image.resize(z6,
		                         shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
		                         method='nearest')
		z7 = fnn.Conv(self.features * 4, kernel_size=(2, 2))(z7_up)
		z7 = fnn.relu(z7)
		z7 = jnp.concatenate([z3, z7], axis=3)
		z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
		z7 = fnn.relu(z7)
		z7 = fnn.Conv(self.features * 4, kernel_size=(3, 3))(z7)
		z7 = fnn.BatchNorm(use_running_average=not self.training)(z7)
		z7 = fnn.relu(z7)
		
		z8_up = jax.image.resize(z7,
		                         shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
		                         method='nearest')
		z8 = fnn.Conv(self.features * 2, kernel_size=(2, 2))(z8_up)
		z8 = fnn.relu(z8)
		z8 = jnp.concatenate([z2, z8], axis=3)
		z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
		z8 = fnn.relu(z8)
		z8 = fnn.Conv(self.features * 2, kernel_size=(3, 3))(z8)
		z8 = fnn.BatchNorm(use_running_average=not self.training)(z8)
		z8 = fnn.relu(z8)
		
		z9_up = jax.image.resize(z8,
		                         shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
		                         method='nearest')
		z9 = fnn.Conv(self.features, kernel_size=(2, 2))(z9_up)
		z9 = fnn.relu(z9)
		z9 = jnp.concatenate([z1, z9], axis=3)
		z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
		z9 = fnn.relu(z9)
		z9 = fnn.Conv(self.features, kernel_size=(3, 3))(z9)
		z9 = fnn.BatchNorm(use_running_average=not self.training)(z9)
		z9 = fnn.relu(z9)
		
		y = fnn.Conv(1, kernel_size=(1, 1))(z9)
		y = fnn.sigmoid(y)
		
		return y


class UNet(fnn.Module):
	features: int = 64
	training: bool = True
	
	@fnn.compact
	def __call__(self, x):
		z1, z2, z3, z4_dropout, z5_dropout = Encoder(self.training)(x)
		y = Decoder(self.training)(z1, z2, z3, z4_dropout, z5_dropout)
		
		return y


class MixerBlock(eqx.Module):
	patch_mixer: eqx.nn.MLP
	hidden_mixer: eqx.nn.MLP
	norm1: eqx.nn.LayerNorm
	norm2: eqx.nn.LayerNorm
	
	def __init__(self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key):
		tkey, ckey = jr.split(key, 2)
		self.patch_mixer = eqx.nn.MLP(num_patches, num_patches, mix_patch_size, depth=1, key=tkey)
		self.hidden_mixer = eqx.nn.MLP(hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey)
		self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
		self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))
	
	def __call__(self, y):
		y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
		y = einops.rearrange(y, "c p -> p c")
		y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
		y = einops.rearrange(y, "p c -> c p")
		return y


class Mixer2d(eqx.Module):
	conv_in: eqx.nn.Conv2d
	conv_out: eqx.nn.ConvTranspose2d
	blocks: list
	norm: eqx.nn.LayerNorm
	t1: float
	
	def __init__(self, img_size, patch_size, hidden_size, mix_patch_size, mix_hidden_size, num_blocks, t1, *, key, ):
		input_size, height, width = img_size
		assert (height % patch_size) == 0
		assert (width % patch_size) == 0
		num_patches = (height // patch_size) * (width // patch_size)
		inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)
		
		self.conv_in = eqx.nn.Conv2d(input_size + 1, hidden_size, patch_size, stride=patch_size, key=inkey)
		self.conv_out = eqx.nn.ConvTranspose2d(hidden_size, input_size, patch_size, stride=patch_size, key=outkey)
		self.blocks = [MixerBlock(num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=bkey) for bkey in
		               bkeys]
		self.norm = eqx.nn.LayerNorm((hidden_size, num_patches))
		self.t1 = t1
	
	def __call__(self, t, y):
		t = t / self.t1
		_, height, width = y.shape
		t = einops.repeat(t, "-> 1 h w", h=height, w=width)
		y = jnp.concatenate([y, t])
		y = self.conv_in(y)
		_, patch_height, patch_width = y.shape
		y = einops.rearrange(y, "c h w -> c (h w)")
		for block in self.blocks:
			y = block(y)
		y = self.norm(y)
		y = einops.rearrange(y, "c (h w) -> c h w", h=patch_height, w=patch_width)
		return self.conv_out(y)
