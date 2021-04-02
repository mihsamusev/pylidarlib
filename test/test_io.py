# -*- coding: utf-8 -*-
import unittest
import itertools
import numpy as np

from pylidarlib import PointCloud
from pylidarlib.io import HDL32e
from test.testdoubles import HDL32ePcapDouble

np.random.seed(seed=42)


class TestHDL32PcapIO(unittest.TestCase):
    """Test cases for readers / writters
    """
    def setUp(self):
        """
        Generate lidar data at different levels
        """
        self.test_azi_1rot = self._get_azimuth_block_single_rotation()
        self.test_azi_2rot = self._get_azimuth_block_double_rotation()
        self.test_elev = self._get_elevations_block()
        self.test_dist, self.test_intens = self._get_random_dist_and_intens()
        self.test_packet1 = HDL32ePcapDouble.build_single_return_packet(
            self.test_azi_1rot, self.test_dist, self.test_intens
        )
        self.test_packet2 = HDL32ePcapDouble.build_single_return_packet(
            self.test_azi_2rot, self.test_dist, self.test_intens
        )
        self.test_numpy_block_1rot = np.dstack([
            self.test_azi_1rot,
            self.test_elev,
            self.test_dist,
            self.test_intens
        ])
        self.test_numpy_block_2rot = np.dstack([
            self.test_azi_2rot,
            self.test_elev,
            self.test_dist,
            self.test_intens
        ])
        self.test_xyzi_block_1rot = self._get_xyzi_datablock(
            self.test_azi_1rot,
            self.test_elev,
            self.test_dist,
            self.test_intens
        )
        self.test_xyzi_block_2rot = self._get_xyzi_datablock(
            self.test_azi_2rot,
            self.test_elev,
            self.test_dist,
            self.test_intens
        )

    def _get_elevations_block(self):
        elevations = np.asarray([
            -30.67, -9.33, -29.33, -8.00, -28.00, -6.66,
            -26.66, -5.33, -25.33, -4.00, -24.00, -2.67,
            -22.67, -1.33, -21.33, 0.00, -20.00, 1.33,
            -18.67, 2.67, -17.33, 4.00, -16.00, 5.33,
            -14.67, 6.67, -13.33, 8.00, -12.00, 9.33,
            -10.67, 10.67
        ])
        return np.repeat(elevations, 12).reshape(32, 12).T

    def _get_azimuth_block_single_rotation(self):
        azimuths = np.array([
            345.5, 346.5, 347.5, 348.5,
            349.5, 350.5, 351.5, 352.5,
            353.5, 354.5, 355.5, 356.5
        ])
        return np.repeat(azimuths, 32).reshape(12, 32)

    def _get_azimuth_block_double_rotation(self):
        azimuths = np.array([
            356.5, 357.5, 358.5, 359.5,
            1.5, 2.5, 3.5, 4.5,
            5.5, 6.5, 7.5, 8.5
        ])
        return np.repeat(azimuths, 32).reshape(12, 32)

    def _get_random_dist_and_intens(self):
        """
        generates (12, 32) arrays for each
        azimuth, elevations, distances, intesities
        only nonzero distances unlike real-lidar
        """
        distances = np.random.randint(1, 70, (12, 32))
        intensities = np.random.randint(0, 100, (12, 32))
        return distances, intensities

    def _get_xyzi_datablock(self, azi, elev, dist, intens):
        """
        Returns (12, 32, 4) array where for each 12 firngs
        there is a (32, 4) array with cartiesian coordinates
        and intensity
        """
        azi_rad = np.deg2rad(azi)
        ele_rad = np.deg2rad(elev)
        rcos_ele = dist * np.cos(ele_rad)
        xyzi = np.dstack([
            rcos_ele * np.sin(azi_rad),
            rcos_ele * np.cos(azi_rad),
            dist * np.sin(ele_rad),
            intens
        ])
        return xyzi

    def test_HDL32e_pcap_packet_parser(self):
        payload = self.test_packet1[42:]  # remove header
        p_azi, p_dist, p_intens = HDL32e.parse_data_packet(payload)

        np.testing.assert_array_equal(
            p_azi, self.test_azi_1rot)
        np.testing.assert_array_almost_equal(
            p_dist, self.test_dist, decimal=2)
        np.testing.assert_array_equal(
            p_intens, self.test_intens)

    def test_HDL32e_pcap_firing_generator(self):
        payload = self.test_packet1[42:]  # remove header
        firings = HDL32e.yield_firings(payload)
        index = 8  # firing at this index is tested
        f = next(itertools.islice(firings, index, None))

        np.testing.assert_array_equal(
            f.azimuth, self.test_azi_1rot[index, :])
        np.testing.assert_array_almost_equal(
            f.distance, self.test_dist[index, :], decimal=2)
        np.testing.assert_array_equal(
            f.intensity, self.test_intens[index, :])

    def test_HDL32e_pcap_count_rotations(self):
        # acts as a mock to dpkt.pcap.Reader generator
        packets = [
            ("timestamp1", self.test_packet1),
            ("timestamp2", self.test_packet2)
        ]
        packet_stream = (p for p in packets)

        r = HDL32e.count_rotations(packet_stream)
        self.assertEqual(r, 2)

    def test_HDL32e_firings_to_xyzi(self):
        payload = self.test_packet1[42:]
        firings = HDL32e.yield_firings(payload)
        firings_list = [f for f in firings]
        xyzi = HDL32e.firings_to_xyzi(firings_list)

        self.assertEqual(xyzi.shape[0], 12 * 32)
        expected = self.test_xyzi_block_1rot.reshape((12 * 32, 4))
        np.testing.assert_array_almost_equal(
            xyzi, expected, decimal=2
        )

    def test_HDL32e_pcap_yield_clouds(self):
        # acts as a mock to dpkt.pcap.Reader generator
        packets = [
            ("timestamp1", self.test_packet1),
            ("timestamp2", self.test_packet2)
        ]
        
        # standart start_angle
        packet_stream = (p for p in packets)

        clouds = []
        cloud_gen = HDL32e.yield_clouds(packet_stream)
        for c in cloud_gen:
            clouds.append(c)

        self.assertEqual(len(clouds), 2)
        self.assertEqual(clouds[0].size, 16 * 32)
        self.assertEqual(clouds[1].size, 8 * 32)

        expected1 = self.test_xyzi_block_1rot.reshape(12 * 32, 4)
        expected2 = self.test_xyzi_block_2rot.reshape(12 * 32, 4)
        expected = np.vstack([expected1, expected2])
        split_idx = (12 + 4) * 32
        expected_pc1 = expected[:split_idx, :]
        np.testing.assert_array_almost_equal(
            clouds[0].data, expected_pc1, decimal=2)
        expected_pc2 = expected[split_idx:,:]
        np.testing.assert_array_almost_equal(
            clouds[1].data, expected_pc2, decimal=2)

        self.assertTrue(
            np.all(np.count_nonzero(clouds[0].xyz, axis=1)))
        self.assertTrue(
            np.all(np.count_nonzero(clouds[1].xyz, axis=1)))

    def test_HDL32e_pcap_yield_clouds_without_zeros(self):
        test_dist_w_zeros = self.test_dist
        rand_column = np.random.randint(0, 32, 3)
        
        test_dist_w_zeros[0, rand_column] = np.zeros(3)
        test_dist_w_zeros[3, rand_column] = np.zeros(3)
        test_dist_w_zeros[4, rand_column] = np.zeros(3)
        test_dist_w_zeros[11,rand_column] = np.zeros(3)

        test_packet1 = HDL32ePcapDouble.build_single_return_packet(
            self.test_azi_1rot, test_dist_w_zeros, self.test_intens
        )
        test_packet2 = HDL32ePcapDouble.build_single_return_packet(
            self.test_azi_2rot, test_dist_w_zeros, self.test_intens
        )
        
        # acts as a mock to dpkt.pcap.Reader generator
        packets = [
            ("timestamp1", test_packet1),
            ("timestamp2", test_packet2)
        ]
        
        # standart start_angle
        packet_stream = (p for p in packets)
        clouds = []
        cloud_gen = HDL32e.yield_clouds(packet_stream)
        for c in cloud_gen:
            clouds.append(c)

        self.assertEqual(len(clouds), 2)
        self.assertEqual(clouds[0].size, 10 * 32 + 6 * 29)
        self.assertEqual(clouds[1].size, 6 * 32 + 2 * 29)