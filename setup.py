import libpyhat
from setuptools import setup, find_packages

from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='libpyhat',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=libpyhat.__version__,

    description='The Python Hyperspectral Analysis Toolkit (PyHAT) for planetary spectral data.',
    long_description=long_description,
    url='https://github.com/USGS-Astrogeology/PyHAT',
    author="J. Laura, R.B. Anderson",
    author_email="jlaura@usgs.gov, rbanderson@usgs.gov",
    license='Unlicense',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        #'License :: OSI Approved :: Unlicense',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scipy', 'gdal', 'plio'],
    extras_require={
        'dev': [],
        'test': ['coverage', 'pytest-cov', 'coveralls'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        #'sample': [],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[]#,('my_data', ['data/data_file'])],
)
