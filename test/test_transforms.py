# -*- coding: utf-8 -*-
import unittest
from pylidarlib import PointCloud
import pylidarlib.transforms as T
import numpy as np


class TestTransforms(unittest.TestCase):
    """Test cases for transforms
    """
    def test_translation(self):
        data = np.random.random((6, 4))
        pc = PointCloud.from_numpy(data)

        t = T.Translate(x=1, y=-2.5, z=3)
        pc = t.apply(pc)
        expected = np.add(data, np.array([1.0, -2.5, 3.0, 0]))
        np.testing.assert_array_equal(pc.data, expected)

    def test_pointcloud_is_copied_during_transform(self):
        data = np.random.random((6, 4))
        pc1 = PointCloud.from_numpy(data)

        t = T.Translate(x=1)
        pc2 = t.apply(pc1)
        self.assertNotEqual(pc1, pc2)

    def test_compose(self):
        data = np.random.random((6, 4))
        pc = PointCloud.from_numpy(data)

        pipe = T.Compose([
            T.Translate(1.1, 2.2, 3.3),
            T.Translate(-1.1, -2.2, -3.3)
        ])
        pc = pipe.apply(pc)
        np.testing.assert_array_almost_equal(pc.data, data)

    def test_cartesian_clip_non_empty_output(self):
        data = np.asarray([
            [1, 0, 0, 1],
            [1, 2, 0, 1],
            [0, 0, 3, 1],
        ])

        # only 1 dim
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClipper(
            x_range=[-1.0, 1.0]
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 1)
        np.testing.assert_array_equal(pc_test.data, data[2:3, :])

        # 2 dim
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClipper(
            x_range=[-1.0, 1.5],
            z_range=[-1.0, 1.0]
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 2)
        np.testing.assert_array_equal(pc_test.data, data[:2, :])

        # 2dim inverse
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClipper(
            x_range=[-1.0, 1.5],
            z_range=[-1.0, 1.0],
            inverse=True
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 1)
        np.testing.assert_array_equal(pc_test.data, data[2:3, :])

    def test_cartesian_clip_empty_output(self):
        data = np.asarray([
            [1, 0, 0, 1],
            [1, 2, 0, 1],
            [0, 0, 3, 1],
        ])

        # only 1 dim
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClipper(
            inverse=True
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 0)
        np.testing.assert_array_equal(pc_test.data, data[:0, :])

    def test_quaternion_rotate(self):
        data = np.asarray([
            [-1, -1, 0, 1],
            [1, -1, 0, 1],
            [1, 1, 0, 1],
            [-1, 1, 0, 1],
        ])
        pc = PointCloud.from_numpy(data)
        # rotate 30 degrees around Z
        rotator = T.AxisRotate(
            axis=[0, 0, 1],
            angle=np.pi / 6)
        pc = rotator.apply(pc)
        expected = np.asarray([
            [-1.3660254, -0.3660254, 0.0, 1.0],
            [0.3660254, -1.3660254, 0.0, 1.0],
            [1.3660254,  0.3660254, 0.0, 1.0],
            [-0.3660254,  1.3660254, 0.0, 1.0]
        ])
        np.testing.assert_array_equal(pc.data, expected)
