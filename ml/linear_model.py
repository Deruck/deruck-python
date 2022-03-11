from .base import Regressor
import numpy as np
import torch

class LinearRegressor(Regressor):
    """Linear Regressor.
    
    A linear regressor with two implement method, gradient descent and analytical.
    
    Attributes:
        _method: strL {"GD", "analytical"}
        
        _learning_rate: float: 
        
        _tol: float: 
        
        _max_iter: float: 
        
        _param: torch.Tensor(n+1 x 1): _param[0] represents b, _param[1:] represents w
        
    Methods:
        fit(X: np.ndarray, y: np.ndarray) -> None: 
        
        predict(self, X: np.ndarray) -> np.ndarray
    """
    def __init__(self, method: str = "GD", learning_rate=10**(-1), tol = 10**(-5), max_iter = 10**(5)):
        super().__init__()
        self._method: str = method
        self._learning_rate = learning_rate
        self._tol = tol
        self._max_iter = max_iter
        
        self._param: torch.Tensor = torch.tensor([[]], requires_grad=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """train linear regression model
        
        
        Args:
            X: np.ndarray:
        
            y: np.ndarray:
        """
        self._X_fit = X
        self._y_fit = y
        self._n = X.shape[0]

        if self._method == "GD":
            self._train_gradient_descent()
        elif self._method == "analytical":
            self._train_analytical()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict with new input
        
        
        Args:
            X: np.ndarray(n x m)
        
        Returns:
            y_pred: np.ndarray(1 x m)
        
        Raises:
            Exception: the number of features does not match.
        """
        if (X.shape[0] != self._X_fit.shape[0]):
            raise(Exception("The number of features does not match."))
        return np.array((self._param[1:,:].T @ X + self._param[0, 0]).flatten())
    
    def _train_analytical(self) -> None:
        X = np.r_[np.ones((1, self._X_fit.shape[1])), self._X_fit]
        y = self._y_fit
        self._param = (np.linalg.inv(X @ X.T) @ X @ y.T).reshape(-1, 1)
            
    def _train_gradient_descent(self) -> None:
        pass