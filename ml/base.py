from abc import ABCMeta, abstractclassmethod


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
