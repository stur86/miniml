# Fitting a model

Once a model is created and bound to its parameters, it can be fitted. Fitting in MiniML uses Jax's automatic differentiation functionality
to find the gradient of the model's `.total_loss` (given a regularization strength parameter $\lambda$); then it uses a MiniML optimizer to find the minimum of the loss for the given batch of training data. By default, MiniML uses the `ScipyOptimizer` wrapper around SciPy's `scipy.optimize.minimize` method, which makes it very easy to leverage SciPy's various optimization methods
for maximum efficiency. These can be often more powerful than classic gradient descent for small models where it's feasible to use them. For example, the default optimization method is `L-BFGS-B`: this method computes an approximation to the Hessian and thus scales with the square of the number of parameters. That would be forbidding for very large models, but it also gives excellent convergence for smaller ones where it's more affordable. [You can check SciPy's documentation for the supported methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

!!! note
    Different SciPy methods require different inputs. Some require only the value of the objective function, others the jacobian, others the Hessian or the product of the Hessian with a vector $\mathbf{p}$. MiniML handles all of this and produces the necessary functions with Jax, then optimizes them with `jax.jit` before passing them to the optimizer.

## Fitting success

The `.fit` method returns a result value containing information such as whether the optimizer succeeded (achieved convergence within the maximum number of iterations).

```py
    res = lin_model.fit(X, y)
    
    print(f"Fit converged: {res.success}")
    print(f"Final loss: {res.loss}")
```

## Custom optimizers

You can provide a custom optimizer to the `.fit` method. For example, to use a different SciPy method:

```py
from miniml.optim.scipy import ScipyOptimizer

optimizer = ScipyOptimizer(method="BFGS", options={"maxiter": 100})
res = model.fit(X, y, optimizer=optimizer)
```

## Pre-fitting

In some cases, one might want to fix certain parameters based on $X$ and $y$ without leaving them to the optimization process. This can be done by overloading the method `_pre_fit` in your model. `_pre_fit` takes the same arguments as `fit` and is expected to return a set of parameter names. It can do anything inside; all parameters in the returned set will be *fixed* and ignored in the subsequent optimization process.

## Recipe: batch fitting

The fitting method doesn't take care of complex fitting processes such as splitting into batches, or changing the regularization strength from one batch to another. If you want to do something like that you can however use a structure like this:

```py
from miniml.optim.scipy import ScipyOptimizer

optimizer = ScipyOptimizer(method="L-BFGS-B", options={"maxiter": 5})

for i, (X, y) in enumerate(training_batches):
    reg_lambda = reg_strength(i) # This could be changing based on iteration
    res = model.fit(X, y, reg_lambda=reg_lambda, optimizer=optimizer)
    print(f"Batch {i+1} | Loss = {res.loss}")
```

This way, fitting will run only for 5 iterations; it's likely that it won't achieve convergence in that time, but it will still store
the latest value of the parameters obtained. The next iteration will then use the updated regularization strength, a new batch, and advance towards the minimum.

