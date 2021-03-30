# pylidarlib
Backend for `pylidartracker` project

## Repo status
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/mihsamusev/pylidarlib)
[![badge](https://github.com/mihsamusev/pylidarlib/workflows/build/badge.svg)](https://github.com/mihsamusev/pylidarlib/actions)

See which OS and Python versions combinations are supported [here](https://github.com/mihsamusev/pylidarlib/actions).

## Getting started
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

## First goal
Reproduce the data processing convert our data from `.pcap` to a list 
of `.bin` with the coordinate system compatiable with KITTI datset

- [] Extract clouds from `.pcap`
- [] Apply filters to the entire datastructure of clouds
- [x] Basic cloud transformations to prepare data for kitty
- [] Save new clouds to `.bin`

