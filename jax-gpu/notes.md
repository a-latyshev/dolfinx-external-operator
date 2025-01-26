# Some notes

In a spack env:
```python
pip install -U "jax[cuda12]"
```

To allocate 4 GPUs
```python
 si-gpu -N 1 -G 4 -t 0-02:00:00
```

> We recommend using one process per GPU and not one per node. In some cases,
> this can speed up jitted computation. The jax.distributed.initialize() API
> will automatically understand that configuration when run under SLURM.
> However, this only a rule of thumb and it may be useful to test both one
> process per GPU and one process per node on your use case.

https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html#computation-follows-data-sharding-and-is-automatically-parallelized
> With sharded input data, the compiler can give us parallel computation. In
> particular, functions decorated with jax.jit can operate over sharded arrays
> without copying data onto a single device. Instead, computation follows
> sharding: based on the sharding of the input data, the compiler decides
> shardings for intermediates and output values, and parallelizes their
> evaluation, even inserting communication operations as necessary.

https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster
> On Cloud TPU, Slurm and Open MPI environments, you can simply call
> jax.distributed.initialize() with no arguments. Default values for the
> arguments will be chosen automatically. When running on GPUs with Slurm and
> Open MPI, it is assumed that one process is started per GPU, i.e. each process
> will be assigned only one visible local device. Otherwise it is assumed that
> one process is started per host, i.e. each process will be assigned all local
> devices. The Open MPI auto-initialization is only used when the JAX processes
> are launched via mpirun/mpiexec.

1 process may have $N$ GPU 
Local devices are about A process.
Global devises are about all devices across all the processes.


`pmap`:
* https://colab.research.google.com/github/google/jax/blob/master/cloud_tpu_colabs/Pmap_Cookbook.ipynb 
* `out = pmap(lambda x: x ** 2, devices=jax.devices('cpu'))(jnp.arange(8))`
* https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap 
> Multi-process platforms: On multi-process platforms such as TPU pods, pmap()
> is designed to be used in SPMD Python programs, where every process is running
> the same Python code such that all processes run the same pmapped function in
> the same order. Each process should still call the pmapped function with
> mapped axis size equal to the number of local devices (unless devices is
> specified, see below), and an array of the same leading axis size will be
> returned as usual. However, any collective operations in fun will be computed
> over all participating devices, including those on other processes, via
> device-to-device communication. Conceptually, this can be thought of as
> running a pmap over a single array sharded across processes, where each
> process “sees” only its local shard of the input and output. The SPMD model
> requires that the same multi-process pmaps must be run in the same order on
> all devices, but they can be interspersed with arbitrary operations running in
> a single process.
