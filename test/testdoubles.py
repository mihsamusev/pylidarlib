from pylidarlib.io import HDL32e
import numpy as np


class HDL32ePcapDouble:
    """
    generates packets and pcap files with given data
    """
    @classmethod
    def get_pcap_header(cls):
        """
        Copied from a valid PCAP file
        """
        pcap_header = bytes.fromhex((
            "d4c3b2a1020004000000000000000000"
            "ffff00000100000085adc750ac970500"
            "e0040000e0040000"
        ))
        return pcap_header

    @classmethod
    def get_packet_header(cls, port=2368):
        """
        Copied from a valid PCAP file
        bytes 36-38
        port_hex = port.to_bytes(2, 'big').hex()
        """
        header = bytes.fromhex((
            "ffffffffffff6076"
            "8820126e08004500"
            "04d200004000ff11"
            "b4a9c0a801c9ffff"
            "ffff0940"
            "0940"
            "04be0000"
        ))
        return header

    @classmethod
    def build_firing_from_numpy(cls, azi, dist_row, int_row):
        """
        builds 2 + 2 + 32 * (2 + 1) = 100 bytes datablock
        corresponding to azimuth data + dist, intensity from firing
        of 32 laser channels
        """
        dist_bytes = [dist_row.tobytes()[i:i+2] for i in range(0, 2*32, 2)]
        int_bytes = [int_row.tobytes()[i:i+1] for i in range(0, 32, 1)]
        firing = np.random.bytes(2) + azi.tobytes()  # 2 byte flag first
        for d, i in zip(dist_bytes, int_bytes):
            firing += d + i
        return firing

    @classmethod
    def build_single_return_packet(
            cls, azi, dist, intens, port=2368, timestamp=0):
        header = cls.get_packet_header(port)
        payload = cls.numpy_data_to_packet_payload(
            azi, dist, intens)
        ts = timestamp.to_bytes(4, 'little')
        factory = np.random.bytes(2)
        return header + payload + ts + factory

    @classmethod
    def numpy_data_to_packet_payload(cls, azi, dist, intens):
        azi_pcap = (azi / HDL32e.AZIMUTH_RES).astype(np.uint16)
        dist_pcap = (dist / HDL32e.DISTANCE_RES).astype(np.uint16)
        intens_pcap = intens.astype(np.uint8)

        payload = b''
        for i in range(0, 12):
            payload += cls.build_firing_from_numpy(
                azi_pcap[i, 0],
                dist_pcap[i, :],
                intens_pcap[i, :]
            )
        return payload
