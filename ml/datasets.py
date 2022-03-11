from abc import ABCMeta, abstractclassmethod
import numpy as np
import pandas as pd
from typing import Tuple
from os import path
from pkgutil import get_data
from io import BytesIO


class dataLoader(object, metaclass=ABCMeta):
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


class regDataLoader(dataLoader):
    """A data loader for regression data sets.


        Methods:
            load (data_name: str: {"wine", "airfoil"}) 
                -> (X: np.ndarray(n x m), y: np.ndarray(1 x m)): load specific data for regression
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
            X: np.ndarray(n x m): input data, n is the number of features, m is the number of inputs

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

        X: np.ndarray = np.array(data.iloc[:, :-1]).T
        y: np.ndarray = np.array(data.iloc[:, -1])

        return X, y
