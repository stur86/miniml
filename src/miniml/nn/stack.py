from jax import Array
from miniml.model import MiniMLModel, MiniMLModelList
from miniml.loss import LossFunction, squared_error_loss

class Stack(MiniMLModel):
    """A MiniML model that stacks multiple MiniML models sequentially."""
    
    def __init__(self, models: list[MiniMLModel], loss: LossFunction = squared_error_loss) -> None:
        """Initialize the Stack model.

        Args:
            models (list[MiniMLModel]): The models to stack.
            loss (LossFunction, optional): The loss function to use. Defaults to squared_error_loss.

        Raises:
            ValueError: If the model list is empty.
        """        
        if len(models) == 0:
            raise ValueError("Stack must contain at least one model")
        self._model_list = MiniMLModelList(models)
        super().__init__(loss=loss)

    def _predict_kernel(self, X: Array, buffer: Array) -> Array:
        for model in self._model_list.contents:
            X = model._predict_kernel(X, buffer)
        return X