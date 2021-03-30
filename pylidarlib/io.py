import numpy as np


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
    AZIMUTH_RES = 0.01
    DISTANCE_RES = 0.002
    HEADER_SIZE = 42


    @staticmethod
    def parse_data_packet(buffer):
        """
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
        azimuth = np.repeat(azimuth, 32).reshape(12, 32)
        
        distance = np.ndarray(
            (HDL32e.BLOCKS, HDL32e.LASERS), np.uint16, buffer, 4, (100, 3)
        ) * HDL32e.DISTANCE_RES
        
        intensity = np.ndarray(
            (HDL32e.BLOCKS, HDL32e.LASERS), np.uint8, buffer, 6, (100, 3))
        return azimuth, distance, intensity

    @staticmethod
    def yield_firings(buffer):
        azimuth, distance, intensity = HDL32e.parse_data_packet(buffer)
        for i in range(HDL32e.BLOCKS):
            firing = HDL32eLaserFiring(
                    azimuth[i, :],
                    distance[i, :],
                    intensity[i, :]
                )
            yield firing


class HDL32eLaserFiring:
    def __init__(self, azimuth, distance, intensity):
        """
        inputs are 1x32 arrays for data from a single HDL32e firing
        contents are (32,) numpy arrays
        """
        self.azimuth = azimuth
        self.elevation = np.asarray(HDL32e.ELE_LUT)
        self.distance = distance
        self.intensity = intensity

    @property
    def xyzi(self):
        """
        returns firing data in cartesian coordinates
        """
        azi_rad = np.deg2rad(self.azimuth)
        ele_rad = np.deg2rad(self.elevation)
        rcos_ele = self.distance * np.cos(ele_rad)
        return np.vstack([
            rcos_ele * np.sin(azi_rad),
            rcos_ele * np.cos(azi_rad),
            self.distance * np.sin(ele_rad),
            self.intensity
        ]).T

    def to_numpy(self):
        """
        returns (32,4) numpy array with columns
        [azimuth, elevation, distance, intensity]
        """
        return np.vstack([
            self.azimuth,
            self.elevation,
            self.distance,
            self.intensity
        ]).T