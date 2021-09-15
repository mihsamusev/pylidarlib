# pylidarlib
Utilities for LIDAR data reading and transformation

## Repo status
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/mihsamusev/pylidarlib)
[![build-and-test-crossplatform](https://github.com/mihsamusev/pylidarlib/actions/workflows/build.yml/badge.svg)](https://github.com/mihsamusev/pylidarlib/actions/workflows/build.yml)

See which OS and Python versions combinations are supported [here](https://github.com/mihsamusev/pylidarlib/actions).

## Getting started

### Quick demo - PCAP to KITTI format
Convert your `.pcap` file collected with Velodyne HDL32e to KITTI compatiable format. Useful for using your own data with one of the KITTI benchmark algorithms for 3D object detection /segmentation.

```python
import dpkt
from pylidarlib.io import HDL32e
import pylidarlib.transforms as PT

# compose a transformation pipeline PyTorch style
pipeline = PT.Compose([
    PT.AxisRotate(
        axis=[0, 0, 1],
        angle=0.7070
        ),
    PT.PolygonCrop(polygon=[
            [0.0, -10.0],
            [-4.0, 0.0],
            [-4.0, 5.0],
            [-20.0, 10.0],
            [-20.0, -12.0],
            [0.0, -32.0]
        ]),
    PT.Translate(x=5, y=-10)
])

# read UDP stream using any package you like, here dpkg is shown
with open("file.pcap", "rb") as fin:
    packet_stream = dpkt.pcap.Reader(fin)

    # feed the stream into cloud generator
    pc_generator = HDL32e.yield_clouds(packet_stream)

    # do something with the clouds
    for i, pc in enumerate(pc_generator):
        pc = pipeline.apply(pc)
        pc.data.astype(np.float32).tofile(f"data/cloud_{i}.bin")
```

## Installation

Clone and install `pylidarlib` to your environment

```sh
git clone https://github.com/mihsamusev/pylidarlib.git
cd pylidarlib
```


Optionally create a conda environment
```sh
conda env create -f environment.yml
```

Or install requirements to an existing environment
```sh
pip install -r requirements.txt

```
Install the module itself using `setup.py`
```
pip install -e .
```

Run tests.
```sh
pytest
```


