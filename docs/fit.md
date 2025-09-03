# Fitting a model

Once a model is created and bound to its parameters, it can be fitted. Fitting in MiniML simply uses Jax's automatic differentiation functionality
to find the gradient of the model's `.total_loss` (given a regularization strength parameter $\lambda$); then it uses SciPy's `scipy.optimize.minimize` method to find the minimum of the loss for the given batch of training data. This makes it very easy to leverage SciPy's various optimization methods
for maximum efficiency, which can be often more powerful than classic gradient descent for small models where it's feasible to use them. For example, the default optimization method is `L-BFGS-B`: this method computes an approximation to the Hermitian and thus scales with the square of the number of parameter. That would be forbidding for very large models, but it also gives excellent convergence for smaller ones where it's more affordable. [You can check SciPy's documentation for the supported methods and other arguments](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

!!! note
    Different SciPy methods require different inputs. Some require only the value of the objective function, others the jacobian, others the Hessian
    or the product of the Hessian with a vector $\mathbf{p}$. MiniML handles all of this and produces the necessary functions with Jax, then optimizes them with `jax.jit` before passing them to `.minimize`.

