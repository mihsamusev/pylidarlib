# -*- coding: utf-8 -*-
import unittest
from pylidarlib import PointCloud
import numpy as np


class TestDataObjects(unittest.TestCase):
    """Test cases for dataobjects.py
    """

    def test_cloud_construction(self):
        pc = PointCloud()
        np.testing.assert_array_equal(pc.data, np.zeros((pc.size, 4)))
        np.testing.assert_array_equal(pc.xyz, np.zeros((pc.size, 3)))
        np.testing.assert_array_equal(pc.intensity, np.zeros((pc.size, 1)))

        pc = PointCloud(capacity=55)
        self.assertEqual(pc.capacity, 55)

    def test_cloud_construction_from_numpy(self):
        # wrong dimensions
        data = np.random.rand(4, 3)
        self.assertRaises(ValueError, PointCloud.from_numpy, data)

        # wrong type
        data = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertRaises(ValueError, PointCloud.from_numpy, data)

        # all correct
        data = np.random.rand(2, 4)
        pc = PointCloud.from_numpy(data)
        np.testing.assert_array_equal(pc.xyz, data[:, :3])
        np.testing.assert_array_equal(pc.intensity, data[:, 3:4])
        self.assertEqual(pc.size, 2)

    def test_cloud_extend_within_capacity(self):
        pc = PointCloud(capacity=10)

        data_chunk1 = np.random.rand(2, 4)
        pc.extend(data_chunk1)
        self.assertEqual(pc.size, 2)

        data_chunk2 = np.random.rand(3, 4)
        pc.extend(data_chunk2)
        self.assertEqual(pc.size, 5)

        expected = np.vstack([data_chunk1, data_chunk2])
        np.testing.assert_array_equal(pc.data, expected)

    def test_cloud_extend_past_capacity(self):
        # past 1x capacity
        pc = PointCloud(capacity=2)
        data_chunk = np.random.rand(3, 4)
        pc.extend(data_chunk)
        self.assertEqual(pc.capacity, 4)
        self.assertEqual(pc.size, 3)

        # past 2x capacity
        pc = PointCloud(capacity=2)
        data_chunk = np.random.rand(5, 4)
        pc.extend(data_chunk)
        self.assertEqual(pc.capacity, 8)
        self.assertEqual(pc.size, 5)

        expected_data = np.zeros((8, 4))
        expected_data[:5, :] = data_chunk
        np.testing.assert_array_equal(pc._data, expected_data)

    def test_cloud_shrink(self):
        # normal
        data_chunk = np.random.rand(3, 4)
        pc = PointCloud.from_numpy(data_chunk)
        pc.shrink()
        self.assertEqual(pc.size, pc.capacity)
        np.testing.assert_array_equal(pc._data, data_chunk)

        # after extend
        data_chunk = np.random.rand(3, 4)
        pc = PointCloud(capacity=2)
        pc.extend(data_chunk)
        pc.shrink()
        self.assertEqual(pc.size, pc.capacity)
        np.testing.assert_array_equal(pc._data, data_chunk)

    def test_cloud_construct_from_numpy_past_capacity(self):
        # past 1x capacity
        data_chunk = np.random.rand(15, 4)
        pc = PointCloud.from_numpy(data_chunk, capacity=10)
        self.assertEqual(pc.capacity, 16)
        self.assertEqual(pc.size, 15)

        # past 2x capacity
        data_chunk = np.random.rand(30, 4)
        pc = PointCloud.from_numpy(data_chunk, capacity=10)
        self.assertEqual(pc.capacity, 32)
        self.assertEqual(pc.size, 30)

        expected_data = np.zeros((32, 4))
        expected_data[:30, :] = data_chunk
        np.testing.assert_array_equal(pc._data, expected_data)


if __name__ == '__main__':
    unittest.main()
