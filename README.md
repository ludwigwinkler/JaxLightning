# JaxLightning
PyTorch Lightning + Jax = nice

# PyTorch Lightning

The package has become the go-to standard of ML research for PyTorch.
Above all, it removes all the boiler plate code that you usually have to write for kicking off a simple experiment.
Additionally, you got amazing logging, general code structure, data management via LightningDataModules and other templates making quick iteration a breeze.

# Jax

Recent packages such as Equinox and Treex are on the top very similar in structure and handling like PyTorch.
This makes the code very readable and succinct.
The biggest advantage of Jax is probably its clean functional programming (I've come around to that) and its speed.
Vmap, derivatives in all directions and automatic accelerator management (no more tensor.to(deviceXYZ)) is also part of the gift box.

# Can we get the best of both worlds? Yes, we can[1].

[1] Obama

### Tensors vs Arrays

The main idea of combining the great and convenient code structure of PyTorch Lightning with the versatility of Jax is to restrict PyTorch LIghtning to pure Numpy/Jax.Numpy until the data 'reaches' the Jax model.
Therefore we can reuse almost all DataModules and DataSets and remove the single line, where data is cast to torch.Tensors.
Thus the dataloader/datamodules etc restricted to Numpy/Jax.Numpy operations.

### Optimization

Secondly, we can't use PyTorch Lightning automatic optimization which makes setting up experiments in PL so convenient.
But at the same time Jax does automatic device placement and moving tensors to the correct devices.
Thus by simply setting the class variable `automatic_optimization=False` we gain complete control and tell PL that we'll do our optimization on our own.

Since Jax requires pure functions, all we have to do is make the forward step a `@staticmethod` without the `self` argument.
Similarly, we can create a static gradient function in the same way.

Then, we can jit-compile the entire forward and backward pass with **JAX** with a simple decorator inside the training setup of **Pytorch Lightning**.
Thus PyTorch Lightning takes care of all the data set management, the logging, the tracking and the overall training loop structure with all the convenience PL is famous for, and Jax does the fast computing inside of PL.

Everybody wins ...

![](now_kiss.jpeg)