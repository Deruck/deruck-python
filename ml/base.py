"""
Abstract Basic Classes

"""

from abc import ABCMeta, abstractclassmethod
import numpy as np


class DataLoader(object, metaclass=ABCMeta):
    """The abstract base class of data loader.

    It is the abstract base class of data loader. There are different types of 
    data sets, like data sets used for regression, classification, clustering,
    etc.

    Attributes:
        _data_dict: Dict[data_name: str, file_name: str]: a map from data_name to file_name

    Methods:
        @abstractclassmethod
        load (data_name: str) -> ?: the method to load specific data set
    """

    def __init__(self) -> None:
        self._data_dict = {}
        pass

    @abstractclassmethod
    def load(self, data_name: str):
        pass


class Regressor(object, metaclass=ABCMeta):
    """An abstract class for regressor.


    Attributes:
        _X_fit: np.ndarray(n x m): 
        _y_fit: np.ndarray(1 x m): 

    Methods:
        fit (X: np.ndarray, y: np.ndarray) -> None: 
        predict(X: np.ndarray) -> np.ndarray: 
    """

    def __init__(self):
        self._X_fit = None
        self._y_fit = None

    @abstractclassmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractclassmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((1, X.shape[1]))
