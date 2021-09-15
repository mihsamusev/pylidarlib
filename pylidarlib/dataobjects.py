import numpy as np


def is_valid_numpy(arr: np.ndarray):
    """
    test if input has same numpy type and right size
    """
    out = False
    if isinstance(arr, np.ndarray):
        out = arr.shape[1] == 4
    return out

class Container3D:
    pass

class PointCloud:
    """
    Cartesian coordinate representation of point could data 
    containing X Y Z and Intensity
    """
    def __init__(self, capacity: np.uint=32768):
        self.capacity = capacity
        self.size = 0
        self._data = np.zeros((self.capacity, 4))

    @property
    def data(self) -> np.ndarray:
        return self._data[:self.size]

    @property
    def xyz(self) -> np.ndarray:
        return self._data[:self.size, :3]

    @property
    def intensity(self) -> np.ndarray:
        return self._data[:self.size, 3:4]

    @staticmethod
    def from_numpy(arr: np.ndarray, **kwargs):
        """
        Constructs a PointCloud using [N x 4] numpy array
        """
        if is_valid_numpy(arr):
            pc = PointCloud(**kwargs)
            pc.extend(arr)
            return pc
        else:
            raise ValueError(
                "Input array should be 'numpy.ndarray' of size [N x 4]")

    def extend(self, arr: np.ndarray):
        """
        Extends the _data array with a [N x 4] numpy array
        """
        next_size = self.size + arr.shape[0]
        if next_size > self.capacity:
            next_pow2 = np.ceil(np.log2(next_size))
            self.capacity = int(np.power(2, next_pow2))

            extended_data = np.zeros((self.capacity, 4))
            extended_data[:self.size, :] = self.data
            self._data = extended_data

        self._data[self.size:next_size, :] = arr
        self.size = next_size

    def shrink(self):
        self.capacity = self.size
        self._data = self._data[:self.size, :]


class RangeImage:
    """
    Cyllindrical coordinate representation of point could data 
    containing Azimuth Elevation Radius and Intensity
    """
    def __init__(self, capacity: np.uint=32768):
        self.capacity = capacity
        self.size = 0
        self._data = np.zeros((self.capacity, 4))

    @property
    def data(self) -> np.ndarray:
        return self._data[:self.size]

    @property
    def azimuth(self) -> np.ndarray:
        return self._data[:self.size, 0:1]
    
    @property
    def elevation(self) -> np.ndarray:
        return self._data[:self.size, 1:2]
    
    @property
    def radius(self) -> np.ndarray:
        return self._data[:self.size, 2:3]

    @property
    def intensity(self) -> np.ndarray:
        return self._data[:self.size, 3:4]