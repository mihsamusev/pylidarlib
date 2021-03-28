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
    def from_numpy(arr, **kwargs):
        """
        Constructs a PointCloud using [N x 4] numpy array
        """
        if PointCloud.is_valid_numpy(arr):
            pc = PointCloud(**kwargs)
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
            next_pow2 = np.ceil(np.log2(next_size))
            self.capacity = int(np.power(2, next_pow2))
            added_rows = self.capacity - self.size
            self._data = np.vstack([
                self._data[:self.size, :],
                np.zeros((added_rows, 4))
            ])

        self._data[self.size:next_size, :] = arr
        self.size = next_size

    def shrink(self):
        self.capacity = self.size
        self._data = self._data[:self.size, :]
