from distutils.core import setup
from setuptools import find_packages

setup(
    name="pylidarlib",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="Python library for simple lidar data operations",
    author="Mihhail Samusev",
    url="https://github.com/mihsamusev",
    keywords=["pointcloud", "lidar"]
)