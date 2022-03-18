from .base import Regressor
from .base import Classifier
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

    def __init__(self, method: str = "GD", learning_rate=10**(-1), tol=10**(-5), max_iter=10**(5)):
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
        self._X_fit = torch.tensor(X.T).double()
        self._y_fit = torch.tensor(y).double()
        self._n = self._X_fit.shape[0]

        if self._method == "GD":
            self._train_gradient_descent()
        elif self._method == "analytical":
            self._train_analytical()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict with new input


        Args:
            X: np.ndarray(m x n)

        Returns:
            y_pred: np.ndarray(1 x m)

        Raises:
            Exception: the number of features does not match.
        """
        X = X.T
        if (X.shape[0] != self._X_fit.shape[0]):
            raise(Exception("The number of features does not match."))
        with torch.no_grad():
            return (self._w.T @ X + self._b).flatten().numpy()

    def _train_analytical(self) -> None:
        X = torch.cat(
            [torch.ones(1, self._X_fit.shape[1]), self._X_fit], dim=0)
        y = self._y_fit
        param = (torch.inverse(X @ X.T) @ X @ y.T).reshape(-1, 1)
        self._w = param[1:, :]
        self._b = param[0, :]

    def _train_gradient_descent(self) -> None:
        self._w = torch.zeros(
            (self._n, 1), requires_grad=True, dtype=torch.double)
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


class LogisticRegression(Classifier):
    """Logistic Regression Classifier with Gradient Descent


    Attributes:
        _X_fit: torch.Tensor: n x m
        _y_fit: torch.Tensor: 1 x m
        _learning_rate: double:
        _tol: double: 
        _max_iter: int: 
        _w: torch.Tensor: n x 1
        _b: torch.Tensor: 1 x 1
        _cost_path: list: list of cost for iterations

    Methods:
        fit(X: np.array, y: np.array) -> None:
        predict(X: np.array) -> np.array:
        predict_prob(X: np.array) -> np.array:
    """

    def __init__(self, learning_rate, tol, max_iter=10000):
        super().__init__()
        self._learning_rate = torch.tensor(learning_rate)
        self._tol = tol
        self._max_iter = max_iter
        self._cost_path = []

    def fit(self, X, y):
        self._X_fit = torch.tensor(X.T).double()
        self._y_fit = torch.tensor(y).double().reshape(1, -1)
        self._n = self._X_fit.shape[0]
        self._m_fit = self._X_fit.shape[1]
        self._w = torch.ones(
            (self._n, 1), requires_grad=True, dtype=torch.double)
        self._b = torch.tensor([0.0], requires_grad=True, dtype=torch.double)
        sigmoid = torch.nn.Sigmoid()
        cross_entropy = torch.nn.BCELoss()

        for i in range(self._max_iter):
            cost = cross_entropy(
                sigmoid(self._w.T @ self._X_fit + self._b), self._y_fit)
            if (i > 2 and self._cost_path[-1] - cost < self._tol):
                break
            self._cost_path.append(cost)
            cost.backward()
            with torch.no_grad():
                self._w -= self._learning_rate * self._w.grad
                self._b -= self._learning_rate * self._b.grad
                self._w.grad.zero_()
                self._b.grad.zero_()

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        sigmoid = torch.nn.Sigmoid()
        X = torch.tensor(X.T)
        return sigmoid(self._w.T @ X + self._b).detach().numpy().flatten()

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred_prob = self.predict_prob(X)
        return (pred_prob > 0.5).astype(np.int)
