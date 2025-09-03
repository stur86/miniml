# Setup

To install MiniML, simply use `pip` or your package manager of choice:

```bash
pip install miniml-jax
```

By default MiniML depends on a version of Jax that is not enabled for GPU or TPU. If you want to make use of hardware acceleration (see [Jax's installation guide](https://docs.jax.dev/en/latest/installation.html)), you should update it separately with `pip` after having installed `miniml-jax`.