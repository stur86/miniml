from miniml.param import MiniMLError, MiniMLParam
from miniml.model import MiniMLModel


def test_model():
    
    class ConstantModel(MiniMLModel):
        def __init__(self):
            self._c = MiniMLParam((1,))
            super().__init__()

    class LinearModel(MiniMLModel):
        
        def __init__(self):
            self._M = MiniMLParam((5,5))
            self._b = MiniMLParam((5,))
            self._c = ConstantModel()
            
            super().__init__()
            
            print(self._params)
            
    m = LinearModel()
    m.randomize()
    m.save("test.npz")
    m.load("test.npz")