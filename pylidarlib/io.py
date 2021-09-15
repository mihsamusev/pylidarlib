from dataclasses import dataclass
from typing import List, Tuple, Iterator
import numpy as np
from pylidarlib import PointCloud


t_packet_stream = List[Tuple[float, bytes]] # (timestam, databuffer)


@dataclass
class HDL32eLaserFiring:
        """
        inputs are 1x32 arrays for data from a single HDL32e firing
        contents are (32,) numpy arrays
        """
        azimuth: np.ndarray
        distance: np.ndarray
        intensity: np.ndarray


class HDL32e:
    """
    Each HDL32e UDP packet is 1248 bytes consists of:
    42 byte packet header
    12x100 byte blocks of data, each with [
        timestamp(uint),
        azimuth(uint) * 100,
        32x[distance(uint) * 500, intensity(byte)]
    ]
    4 byte timestamp, 2 byte factory
    """
    LASERS = 32
    BLOCKS = 12
    ELE_LUT = [
        -30.67, -9.33, -29.33,
        -8.00, -28.00, -6.66,
        -26.66, -5.33, -25.33,
        -4.00, -24.00, -2.67,
        -22.67, -1.33, -21.33,
        0.00, -20.00, 1.33,
        -18.67, 2.67, -17.33,
        4.00, -16.00, 5.33,
        -14.67, 6.67, -13.33,
        8.00, -12.00, 9.33,
        -10.67, 10.67
    ]
    ELE_ARRAY = np.asarray(ELE_LUT)
    ELE_ARRAY_RAD = np.deg2rad(ELE_ARRAY)
    ELE_ARRAY_COS = np.cos(ELE_ARRAY_RAD)
    ELE_ARRAY_SIN = np.sin(ELE_ARRAY_RAD)
    AZIMUTH_RES = 0.01
    DISTANCE_RES = 0.002
    HEADER_SIZE = 42
    PACKET_SIZE = 1248

    @staticmethod
    def parse_data_packet(buffer: bytes):
        """
        Parses HDL32e strongest return mode data packet
        Fast parsing from
        # https://stackoverflow.com/questions/36797088/speed-up-pythons-struct-unpakc
        using details of np.ndarray(
            shape - output array shape
            dtype - <H little-endian unsigned short aka np.uint16,
                    <B little-endian unsigned byte aka np.uint8
            buffer - the placeholder of the data itself
            offset - from header
            strides - that type re-appears every th)
        """
        azimuth = np.ndarray(
            (HDL32e.BLOCKS,), np.uint16, buffer, 2, (100, )
        ) * HDL32e.AZIMUTH_RES
        azimuth = np.repeat(azimuth, HDL32e.LASERS).reshape(
            HDL32e.BLOCKS, HDL32e.LASERS)

        distance = np.ndarray(
            (HDL32e.BLOCKS, HDL32e.LASERS), np.uint16, buffer, 4, (100, 3)
        ) * HDL32e.DISTANCE_RES

        intensity = np.ndarray(
            (HDL32e.BLOCKS, HDL32e.LASERS), np.uint8, buffer, 6, (100, 3))
        return azimuth, distance, intensity

    @staticmethod
    def yield_firings(buffer: bytes) -> Iterator[HDL32eLaserFiring]:
        """
        Generator for HDL32e lidar firings from data packets
        Only supports strongest return mode
        """
        azimuth, distance, intensity = HDL32e.parse_data_packet(buffer)
        for i in range(HDL32e.BLOCKS):
            firing = HDL32eLaserFiring(
                    azimuth[i],
                    distance[i],
                    intensity[i]
                )
            yield firing

    @staticmethod
    def yield_clouds(packet_stream: t_packet_stream) -> Iterator[PointCloud]:
        """
        Generator for point clouds from HDL32e pcap data
        packet stream
        """
        prev_azi = 0
        firings_buffer: List[HDL32eLaserFiring] = []
        for timestamp, packet in packet_stream:
            if len(packet) != HDL32e.PACKET_SIZE:
                continue

            for firing in HDL32e.yield_firings(packet[42:]):
                if prev_azi > firing.azimuth[0]:
                    xyzi = HDL32e.firings_to_xyzi(firings_buffer)
                    xyzi = xyzi[np.where(np.count_nonzero(xyzi[:, :3], axis=1))]
                    pc = PointCloud.from_numpy(xyzi)
                    firings_buffer = []
                    yield pc

                firings_buffer.append(firing)
                prev_azi = firing.azimuth[0]
        
        xyzi = HDL32e.firings_to_xyzi(firings_buffer)
        xyzi = xyzi[np.where(np.count_nonzero(xyzi[:, :3], axis=1))]
        pc = PointCloud.from_numpy(xyzi)
        yield pc

    @staticmethod
    def firings_to_xyzi(firings: List[HDL32eLaserFiring]) -> np.ndarray:
        """
        converts list of HDL32eLaserFiring to xyzi nuy
        """
        n = len(firings)
        xyzi = np.zeros((n * HDL32e.LASERS, 4))
        for i, firing in enumerate(firings): 
            azi_rad = np.deg2rad(firing.azimuth)
            rcos_ele = firing.distance * HDL32e.ELE_ARRAY_COS

            i_start = HDL32e.LASERS * i
            i_end = i_start + HDL32e.LASERS
            xyzi[i_start:i_end, 0] = rcos_ele * np.sin(azi_rad)
            xyzi[i_start:i_end, 1] = rcos_ele * np.cos(azi_rad)
            xyzi[i_start:i_end, 2] = firing.distance * HDL32e.ELE_ARRAY_SIN
            xyzi[i_start:i_end, 3] = firing.intensity
        return xyzi

    @staticmethod
    def count_rotations(packet_stream: t_packet_stream) -> int:
        """
        Counts full rotations (number of point clouds)
        of a HDL32e rounded up, new rotation is registered
        after crossing 0 .Accounts for new rotation starting
        both within packet but also from one packet to another
        """
        count = 1
        prev_max_azi = 0

        for timestamp, packet in packet_stream:
            if len(packet) != HDL32e.PACKET_SIZE:
                continue
            # use same binary parsing as in parse_data_packet()
            min_azi, max_azi = np.ndarray(
                (2,), np.uint16, packet, HDL32e.HEADER_SIZE + 2, (1100,))
            if (max_azi < min_azi or prev_max_azi > min_azi):
                count += 1
            prev_max_azi = max_azi

        return count


