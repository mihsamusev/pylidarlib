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

    def test_cartesian_clip_non_empty_output_1d(self):
        data = np.asarray([
            [1, 0, 0, 1],
            [1, 2, 0, 1],
            [0, 0, 3, 1],
        ])
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClip(
            x_range=[-1.0, 1.0]
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 1)
        np.testing.assert_array_equal(pc_test.data, data[2:3, :])

    def test_cartesian_clip_non_empty_output_2d(self):
        data = np.asarray([
            [1, 0, 0, 1],
            [1, 2, 0, 1],
            [0, 0, 3, 1],
        ])
        pc = PointCloud.from_numpy(data)
        clipper = T.CartesianClip(
            x_range=[-1.0, 1.5],
            z_range=[-1.0, 1.0]
        )
        pc_test = clipper.apply(pc)
        self.assertEqual(pc_test.size, 2)
        np.testing.assert_array_equal(pc_test.data, data[:2, :])

    def test_cartesian_clip_non_empty_output_2d_inverse(self):
        data = np.asarray([
            [1, 0, 0, 1],
            [1, 2, 0, 1],
            [0, 0, 3, 1],
        ])
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

    def test_quaternion_rotate_30_around_z(self):
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

    def test_compose_2_translates(self):
        data = np.random.random((6, 4))
        pc = PointCloud.from_numpy(data)

        pipe = T.Compose([
            T.Translate(1.1, 2.2, 3.3),
            T.Translate(-1.1, -2.2, -3.3)
        ])
        pc = pipe.apply(pc)
        np.testing.assert_array_almost_equal(pc.data, data)

    def test_compose_rotate_translate(self):
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

    def test_polygon_clipper_clips_entire_pc(self):
        data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 2, 1],
            [1, 1, 3, 1],
            [-1, 1, 4, 1],
        ])
        pc = PointCloud.from_numpy(data)
        poly = [
            [-2, -2],
            [2, -2],
            [2, 2],
            [-2, 2],
        ]
        clipper = T.PolygonCrop(
            polygon=poly,
            inverse=True
        )
        new_pc = clipper.apply(pc)
        self.assertEqual(new_pc.size, 0)
        np.testing.assert_array_equal(new_pc.data.shape, (0, 4))

    def test_polygon_clipper_positive(self):
        data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 2, 1],
            [1, 1, 3, 1],
            [-1, 1, 4, 1],
        ])
        pc = PointCloud.from_numpy(data)
        poly = [
            [-2, -2],
            [0, -2],
            [0, 0],
            [2, 0],
            [2, 2],
            [-2, 2],
        ]
        clipper = T.PolygonCrop(
            polygon=poly
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 3)
        np.testing.assert_array_equal(pc.data, data[[0, 2, 3], :])

    def test_polygon_clipper_inverse(self):
        data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 2, 1],
            [1, 1, 3, 1],
            [-1, 1, 4, 1],
        ])
        pc = PointCloud.from_numpy(data)
        poly = [
            [-2, -2],
            [0, -2],
            [0, 0],
            [2, 0],
            [2, 2],
            [-2, 2],
        ]
        clipper = T.PolygonCrop(
            polygon=poly,
            inverse=True
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 1)
        np.testing.assert_array_equal(pc.data, data[[1], :])

    def test_polygon_clipper_z_range(self):
        data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 2, 1],
            [1, 1, 3, 1],
            [-1, 1, 4, 1],
        ])
        pc = PointCloud.from_numpy(data)
        poly = [
            [-2, -2],
            [0, -2],
            [0, 0],
            [2, 0],
            [2, 2],
            [-2, 2],
        ]
        clipper = T.PolygonCrop(
            polygon=poly,
            z_range=[2, 5],
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 2)
        np.testing.assert_array_equal(pc.data, data[[2, 3], :])

    def test_polygon_clipper_poly_nparray(self):
        data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 2, 1],
            [1, 1, 3, 1],
            [-1, 1, 4, 1],
        ])
        pc = PointCloud.from_numpy(data)
        poly = np.asarray([
            [-2, -2],
            [0, -2],
            [0, 0],
            [2, 0],
            [2, 2],
            [-2, 2],
        ])
        clipper = T.PolygonCrop(
            polygon=poly,
            z_range=[2, 5],
        )
        pc = clipper.apply(pc)
        self.assertEqual(pc.size, 2)
        np.testing.assert_array_equal(pc.data, data[[2, 3], :])

    def test_cloud_subtractor_empty_subtracted_cloud_error(self):
        subtracted_data = np.empty((0, 4))
        subtracted_pc = PointCloud.from_numpy(subtracted_data)
        self.assertRaises(ValueError, T.CloudSubtractor, subtracted_pc)

    def test_cloud_subtractor_nonpositive_radius_error(self):
        subtracted_data = np.zeros((1, 4))
        subtracted_pc = PointCloud.from_numpy(subtracted_data)
        self.assertRaises(ValueError, T.CloudSubtractor, subtracted_pc, radius=0)
        self.assertRaises(ValueError, T.CloudSubtractor, subtracted_pc, radius=-0.5)

    def test_cloud_subtractor_partial_subtraction(self):
        source_data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, 1, 1],
            [-1, 1, 1, 1],
        ])
        subtracted_data = np.asarray([
            [0.9, -0.9, 1.1, 1],
            [1.9, -0.9, 1.1, 1],
            [1.9, 1.1, 1.1, 1],
            [0.9, 1.1, 1.1, 1],
        ])
        expected = np.asarray([
            [-1, -1, 1, 1],
            [-1, 1, 1, 1],
        ])
        source_pc = PointCloud.from_numpy(source_data)
        subtracted_pc = PointCloud.from_numpy(subtracted_data)
        subtractor = T.CloudSubtractor(
            subtracted=subtracted_pc,
            radius=0.2
        )
        new_pc = subtractor.apply(source_pc)

        self.assertEqual(new_pc.size, 2)
        np.testing.assert_array_equal(new_pc.data, expected)

    def test_cloud_subtractor_complete_subtraction(self):
        source_data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, 1, 1],
            [-1, 1, 1, 1],
        ])
        source_pc = PointCloud.from_numpy(source_data)
        subtracted_pc = PointCloud.from_numpy(source_data)
        subtractor = T.CloudSubtractor(
            subtracted=subtracted_pc,
            radius=0.2
        )
        new_pc = subtractor.apply(source_pc)

        self.assertEqual(new_pc.size, 0)
        np.testing.assert_array_equal(new_pc.data.shape, (0, 4))

    def test_cloud_subtractor_no_subtraction(self):
        source_data = np.asarray([
            [-1, -1, 1, 1],
            [1, -1, 1, 1],
            [1, 1, 1, 1],
            [-1, 1, 1, 1],
        ])
        subtracted_data = np.asarray([
            [0.9, -0.9, 1.1, 1],
            [1.9, -0.9, 1.1, 1],
            [1.9, 1.1, 1.1, 1],
            [0.9, 1.1, 1.1, 1],
        ])
        expected = np.asarray([
            [-1, -1, 1, 1],
            [-1, 1, 1, 1],
        ])
        source_pc = PointCloud.from_numpy(source_data)
        subtracted_pc = PointCloud.from_numpy(subtracted_data)
        subtractor = T.CloudSubtractor(
            subtracted=subtracted_pc,
            radius=0.05
        )
        new_pc = subtractor.apply(source_pc)

        self.assertEqual(new_pc.size, 4)
        np.testing.assert_array_equal(new_pc.data, source_pc.data)

