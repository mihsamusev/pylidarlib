# pylidarlib
Backend for `pylidartracker` project

## Repo status
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/mihsamusev/pylidarlib)
[![badge](https://github.com/mihsamusev/pylidarlib/workflows/build/badge.svg)](https://github.com/mihsamusev/pylidarlib/actions)

See which OS and Python versions combinations are supported [here](https://github.com/mihsamusev/pylidarlib/actions).

## Getting started

### Quick demo - PCAP to KITTI format
Convert your `.pcap` file collected with Velodyne HDL32e to KITTI compatiable format. Useful for using your date with one of the KITTI benchmark algorithms for 3D object detection /segmentation.

```python
import dpkt
from pylidarlib.io import HDL32e
import pylidarlib.transforms as PT

# compose a transformation pipeline PyTorch style
pipeline = PT.Compose([
    PT.AxisRotate(),
    PT.Clip()
])

with open("file.pcap", "r") as fin:
    packet_stream = dpkt.pcap.Reader(fin)
    pc_generator = HDL32e.yield_clouds(packet_stream)
    for pc in pc_generator:
        pc = pipeline.apply(pc)
        pc.serialize(f"00001.bin")
```

## Installation
0) Optionally create an environnment
```sh
# conda

```


1) Clone and install `pylidarlib` to your environment

```sh
git clone https://github.com/mihsamusev/pylidarlib.git
cd strategoutil
```

Create an environment
```sh
conda env create -f environment.yml
```
 or 
```sh
pip install -r requirements.txt
```
 and install package itself using `setup.py`
```
pip install -e .
```

2) Run tests.
```sh
python -m pytest
```


