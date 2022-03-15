from deruck_python.ml.base import Regressor
from deruck_python.ml.cost_functions import squared_loss
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
        
        self._w: torch.Tensor = torch.tensor([[]])
        self._b: torch.Tensor = torch.tensor([[]])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """train linear regression model
        
        
        Args:
            X: np.ndarray:
        
            y: np.ndarray:
        """
        self._X_fit = torch.tensor(X).double()
        self._y_fit = torch.tensor(y).double()
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
        with torch.no_grad():
            return (self._w.T @ X + self._b).flatten().numpy()
    
    def _train_analytical(self) -> None:
        X = torch.cat([torch.ones(1, self._X_fit.shape[1]), self._X_fit], dim=0)
        y = self._y_fit
        param = (torch.inverse(X @ X.T) @ X @ y.T).reshape(-1, 1)
        self._w = param[1:, :]
        self._b = param[0, :]
            
    def _train_gradient_descent(self) -> None:
        self._w = torch.zeros((self._n, 1), requires_grad=True, dtype=torch.double)
        self._b = torch.tensor(0.0, requires_grad=True, dtype=torch.double)
        
        self._cost_path = []
        while(len(self._cost_path) < self._max_iter):
            cost = squared_loss(self._w.T @ self._X_fit + self._b, self._y_fit)
            self._cost_path.append(cost)
            cost.backward()
            with torch.no_grad():
                self._w -= torch.tensor(self._learning_rate) * self._w.grad
                self._b -= torch.tensor(self._learning_rate) * self._b.grad
                self._w.grad.zero_()
                self._b.grad.zero_()
        