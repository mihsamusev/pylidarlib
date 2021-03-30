# -*- coding: utf-8 -*-
import unittest
import os
import itertools
import numpy as np

from pylidarlib.io import HDL32e
from test.testdoubles import HDL32ePcapDouble


class TestHDL32PcapIO(unittest.TestCase):
    """Test cases for readers / writters
    """
    def setUp(self):
        np.random.seed(seed=42)
        azimuths = np.array([
            356.5, 357.5, 358.5, 359.5,
            1.5, 2.5, 3.5, 4.5,
            5.5, 6.5, 7.5, 8.5
        ])
        self.azimuths = np.repeat(azimuths, 32).reshape(12, 32)
        max_dist = 70
        self.distances = 70 * np.random.random((12, 32))
        self.intensities = np.random.randint(0, 100, (12, 32))
        
        self.elevations = np.asarray([
            -30.67, -9.33, -29.33, -8.00, -28.00, -6.66,
            -26.66, -5.33, -25.33, -4.00, -24.00, -2.67,
            -22.67, -1.33, -21.33, 0.00, -20.00, 1.33,
            -18.67, 2.67, -17.33, 4.00, -16.00, 5.33,
            -14.67, 6.67, -13.33, 8.00, -12.00, 9.33,
            -10.67, 10.67
        ])
        reader = HDL32ePcapDouble()
        self.test_packet = reader.get_single_return_packet(
            azi=self.azimuths,
            dist=self.distances,
            intens=self.intensities
        )

    def test_HDL32e_pcap_packet_parser(self):      
        payload = self.test_packet[42:] # remove header
        p_azi, p_dist, p_intens = HDL32e.parse_data_packet(payload)

        np.testing.assert_array_equal(
            p_azi, self.azimuths)
        np.testing.assert_array_almost_equal(
            p_dist, self.distances, decimal=2) # due to conversion
        np.testing.assert_array_equal(
            p_intens, self.intensities)

    def test_HDL32e_pcap_firing_generator(self):
        payload = self.test_packet[42:] # remove header
        
        firings = HDL32e.yield_firings(payload)
        index = 8 # firing at this index is tested
        f = next(itertools.islice(firings, index, None))

        np.testing.assert_array_equal(
            f.azimuth, self.azimuths[index, :])
        np.testing.assert_array_almost_equal(
            f.distance, self.distances[index, :], decimal=2) # due to conversion
        np.testing.assert_array_equal(
            f.intensity, self.intensities[index, :])
        np.testing.assert_array_equal(
            f.elevation, self.elevations)

    def test_HDL32e_LaserFiring_to_numpy(self):
        payload = self.test_packet[42:]
        firings = HDL32e.yield_firings(payload)
        index = 8 # firing at this index is tested
        f = next(itertools.islice(firings, index, None))
        farray = f.to_numpy()

        expected = np.vstack([
            self.azimuths[index, :],
            self.elevations,
            self.distances[index, :],
            self.intensities[index, :]
        ]).T
        np.testing.assert_array_almost_equal(
            farray, expected, decimal=2)

    def test_HDL32e_LaserFiring_xyzi_property(self):
        payload = self.test_packet[42:]
        firings = HDL32e.yield_firings(payload)
        index = 8 # firing at this index is tested
        f = next(itertools.islice(firings, index, None))
 
        azi_rad = np.deg2rad(self.azimuths[index, :])
        ele_rad = np.deg2rad(self.elevations)
        rcos_ele = self.distances[index, :] * np.cos(ele_rad)
        
        expected_xyzi = np.vstack([
            rcos_ele * np.sin(azi_rad),
            rcos_ele * np.cos(azi_rad),
            self.distances[index, :] * np.sin(ele_rad),
            self.intensities[index, :]
        ]).T
        np.testing.assert_array_almost_equal(
            f.xyzi, expected_xyzi, decimal=2)

    def test_PointCloud_creation_by_accum_of_HDL32e_firings(self):
        pass