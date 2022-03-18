from .base import DataLoader
import numpy as np
import pandas as pd
from typing import Tuple
from os import path
from pkgutil import get_data
from io import BytesIO
from sklearn import datasets


class RegDataLoader(DataLoader):
    """A data loader for regression data sets.


        Methods:
            load (data_name: str: {"wine", "airfoil"}) 
                -> (X: np.ndarray(m x n), y: np.ndarray(1 x m)): load specific data for regression
    """

    def __init__(self) -> None:
        super().__init__()
        self._data_dict = {
            "wine": "winequality-white.csv",
            "airfoil": "airfoil_self_noise.dat",
        }

    def load(self, data_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data


        Args:
            data_name: str: {"wine", "airfoil"}: the name of data


        Returns:
            X: np.ndarray(m x n): input data, n is the number of features, m is the number of inputs

            y: np.ndarray(1 x m): output data


        Raises:
            ValueError: the data set does not exist
        """
        if data_name not in self._data_dict.keys():
            raise ValueError("The data set does not exist.")

        data_dir = path.join("..", "data")
        file_path = path.join(data_dir, self._data_dict[data_name])
        data_bytes = BytesIO(get_data(__package__, file_path))

        if data_name == "wine":
            data: pd.DataFrame = pd.read_csv(data_bytes, sep=';')
        elif data_name == "airfoil":
            data: pd.DataFrame = pd.read_csv(data_bytes, sep='\t', header=None)

        X: np.ndarray = np.array(data.iloc[:, :-1])
        y: np.ndarray = np.array(data.iloc[:, -1])

        return X, y


class ClfDataLoader(DataLoader):
    """A data loader for classification data sets.


        Methods:
            load (data_name: str: {"iris_2", "iris_multi"}) 
                -> (X: np.ndarray(m x n), y: np.ndarray(1 x m)): 
    """
    def __init__(self) -> None:
        super().__init__()
        self._data_dict = {
            "iris_2": 0,
            "iris_multi": 0,
        }
    
    
    def load(self, data_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data


        Args:
            data_name: str: {"wine", "airfoil"}: the name of data


        Returns:
            X: np.ndarray(m x n): input data, n is the number of features, m is the number of inputs

            y: np.ndarray(1 x m): output data


        Raises:
            ValueError: the data set does not exist
        """
        
        if data_name not in self._data_dict.keys():
            raise ValueError("The data set does not exist.")
        elif self._data_dict[data_name] == 0:
            if data_name == "iris_multi":
                iris: datasets.base.Bunch = datasets.load_iris()
                X: np.ndarray = iris.data
                y: np.ndarray = iris.target
                return X, y
            elif data_name == "iris_2":
                iris: datasets.Bunch = datasets.load_iris()
                X: np.ndarray = iris.data
                y: np.ndarray = iris.target
                idx = np.argwhere((y == 1) | (y == 0)).flatten()
                X: np.ndarray = X[idx, :]
                y: np.ndarray = y[idx]
                return X, y
          
          
            
        return np.array(0), np.array(0)
    