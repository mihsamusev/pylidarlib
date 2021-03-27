import numpy as np


class PointCloud:
    """
    Point could containing XYZI data
    """
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self._data = np.zeros((self.capacity, 4))

    @property
    def data(self):
        return self._data[:self.size]

    @data.setter
    def data(self, value):
        self._data = value
        self.size = self._data.shape[0]

    @property
    def xyz(self):
        return self._data[:self.size, :3]

    @property
    def intensity(self):
        return self._data[:self.size, 3:4]

    @staticmethod
    def is_valid_numpy(arr):
        """
        test if input has same numpy type and right size
        """
        out = False
        if isinstance(arr, np.ndarray):
            out = arr.shape[1] == 4
        return out

    @staticmethod
    def from_numpy(arr):
        """
        Constructs a PointCloud using [N x 4] numpy array
        """
        if PointCloud.is_valid_numpy(arr):
            pc = PointCloud()
            pc.extend(arr)
            return pc
        else:
            raise ValueError(
                "Input array should be 'numpy.ndarray' of size [N x 4]")

    def extend(self, arr):
        """
        Extends the _data array with a [N x 4] numpy array
        """
        next_size = self.size + arr.shape[0]
        if next_size > self.capacity:
            self._add_capacity()
        self._data[self.size:next_size, :] = arr
        self.size = next_size

    def _add_capacity(self):
        """
        Increase the capacity by doubling the size of the _data array
        """
        self.capacity *= 2
        self._data = np.vstack([self._data, np.zeros_like(self._data)])

    def shrink(self):
        self.capacity = self.size
        self._data = self._data[:self.size,:]

