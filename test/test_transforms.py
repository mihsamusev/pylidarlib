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
        clipper = T.CartesianClip(
            x_range=[-1.0, 1.0]
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 1)
        np.testing.assert_array_equal(pc_test.data, data[2:3, :])

        # 2 dim
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClip(
            x_range=[-1.0, 1.5],
            z_range=[-1.0, 1.0]
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 2)
        np.testing.assert_array_equal(pc_test.data, data[:2, :])

        # 2dim inverse
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClip(
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
        clipper = T.CartesianClip(
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
            axis=[0.0, 0.0, 1.0],
            angle=np.pi / 6
        )
        pc = rotator.apply(pc)
        expected = np.asarray([
            [-0.3660254, -1.3660254, 0.0, 1.0],
            [1.3660254, -0.3660254, 0.0, 1.0],
            [0.3660254, 1.3660254, 0.0, 1.0],
            [-1.3660254, 0.3660254, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(pc.data, expected)

        # rotate 30 degrees, axis not normalized
        pc = PointCloud.from_numpy(data)
        rotator = T.AxisRotate(
            axis=[0.0, 0.0, 1.1],
            angle=np.pi / 6
        )
        pc = rotator.apply(pc)
        np.testing.assert_array_almost_equal(pc.data, expected)

    def test_rotate_translate_compose(self):
        data = np.asarray([
            [0, 0, 2, 1],
            [2, 0, 2, 1],
            [2, 2, 2, 1],
            [0, 2, 2, 1],
        ])
        pc = PointCloud.from_numpy(data)
        pipe = T.Compose([
            T.Translate(0, 0, -2),
            T.AxisRotate([1, 0, 0], np.pi / 2)
        ])
        pc = pipe.apply(pc)
        expected = np.asarray([
            [0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 2.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(pc.data, expected)

    def test_polygon_clipper(self):
        data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 2, 1],
            [1, 1, 3, 1],
            [-1, 1, 4, 1],
        ])
        # positive clipper given 2d list
        pc = PointCloud.from_numpy(data)
        poly = [
            [-2, -2],
            [0, -2],
            [0, 0],
            [2, 0],
            [2, 2],
            [-2, 2],
        ]
        clipper = T.PolygonClip(
            polygon=poly
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 3)
        np.testing.assert_array_equal(pc.data, data[[0,2,3], :])

        # inverse clipper given 2d list
        pc = PointCloud.from_numpy(data)
        clipper = T.PolygonClip(
            polygon=poly,
            inverse=True
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 1)
        np.testing.assert_array_equal(pc.data, data[[1], :])

        # positive clipper given z_range
        pc = PointCloud.from_numpy(data)
        clipper = T.PolygonClip(
            polygon=poly,
            z_range=[2, 5],
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 2)
        np.testing.assert_array_equal(pc.data, data[[2,3], :])

        # poly as numpy array
        poly = np.asarray(poly)
        pc = PointCloud.from_numpy(data)
        clipper = T.PolygonClip(
            polygon=poly,
            z_range=[2, 5],
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 2)
        np.testing.assert_array_equal(pc.data, data[[2,3], :])
