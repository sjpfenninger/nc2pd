#!/usr/bin/env python

from setuptools import setup, find_packages

# Sets the __version__ variable
exec(open('nc2pd/_version.py').read())

setup(
    name='nc2pd',
    version=__version__,
    author='Stefan Pfenninger',
    author_email='stefan@pfenninger.org',
    description='A python-netCDF4 wrapper to turn netCDF into pandas data structures',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "pandas >= 0.15.0",
        "netCDF4 >= 1.1.1"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4'
    ],
)
