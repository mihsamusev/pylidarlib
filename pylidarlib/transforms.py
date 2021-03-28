import numpy as np
from pylidarlib import PointCloud


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, pc):
        for t in self.transforms:
            pc = t.apply(pc)
        return pc


class Translate:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def apply(self, pc):
        """translate .xyz of the point cloud, this
        implementation creates outputs
        """
        new_data = pc.data + np.array([self.x, self.y, self.z, 0])
        return PointCloud.from_numpy(
            new_data,
            capacity=new_data.shape[0])


class CartesianClipper:
    def __init__(
            self,
            x_range=[-np.inf, np.inf],
            y_range=[-np.inf, np.inf],
            z_range=[-np.inf, np.inf],
            inverse=False):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.inverse = inverse

    def apply(self, pc):
        """
        Points that lay right at the boundary are not included
        """
        x, y, z = pc.xyz.T
        x_mask = (x > self.x_range[0]) & (x < self.x_range[1])
        y_mask = (y > self.y_range[0]) & (y < self.y_range[1])
        z_mask = (z > self.z_range[0]) & (z < self.z_range[1])
        mask = x_mask & y_mask & z_mask

        if self.inverse:
            mask = np.invert(mask)

        new_data = pc.data[mask, :]
        return PointCloud.from_numpy(
            new_data,
            capacity=new_data.shape[0])
