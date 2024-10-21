from setuptools import setup

setup(
    name='RadarError',
    url='https://github.com/amyycb/raderr',
    author='Dr. Amy C. Green',
    author_email='amy.green3@newcastle.ac.uk',
    packages=['raderr', 'test'],
    install_requires=['numpy', 'wradlib', 'h5py', ],
    version='0.1.0',
    description='A Python package for simulating radar errors for a rainfall event',
)