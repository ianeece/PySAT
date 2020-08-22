import numpy as np
from pandas import Index, MultiIndex, DataFrame, concat
import pytest

import libpyhat
from libpyhat import Spectra, Spectrum

# Fixtures were fixed according to: https://docs.pytest.org/en/latest/deprecations.html#calling-fixtures-directly

def spectra():
    return Spectra(np.arange(1,17).reshape(4,4),
                   index=[2.22221, 3.33331, 4.400001, 5.500001],
                   wavelengths=[2.22221, 3.33331, 4.400001, 5.500001],
                   columns=['a', 'b', 'c', 'd'])

def spectra_multiindex():
    multi = [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b')]
    cols = MultiIndex.from_tuples(multi, names=['observationid', 'wv'])
    return Spectra(np.arange(1,21).reshape(5,4),
                   index=[2.22221, 3.33331, 4.400001, 5.500001, 6.6],
                   wavelengths=[2.22221, 3.33331, 4.400001, 5.500001, 6.6],
                   columns=cols)

def spectra_metadata():
        return Spectra([[1,2,3,4],
                        [1,2,3,4],
                        [1,2,3,4],
                        [1,2,3,4],
                        ['a', 'a', 'a', 'a'],
                        ['b', 'b', 'b', 'b'],
                        ['c', 'c', 'c', 'c']],
                   index=[2.22221, 3.33331, 4.400001, 5.500001, 'foo', 'bar', 'bat'],
                   wavelengths=[2.22221, 3.33331, 4.400001, 5.500001],
                   metadata=['foo', 'bar', 'bat'] ,
                   columns=['a', 'b', 'c', 'd'])


@pytest.fixture(name="spectra")
def spectra_fixture():
    return spectra()

@pytest.fixture(name="spectra_multiindex")
def spectra_multiindex_fixture():
    return spectra_multiindex()

@pytest.fixture(name="spectra_metadata")
def spectra_metadata_fixture():
    return spectra_metadata()


@pytest.mark.parametrize("spectra, cols, cls",
                         [(spectra(), 'a', Spectrum),
                          (spectra(), ['a', 'c'], Spectra),
                          (spectra_metadata(), 'a', Spectrum),
                          (spectra_metadata(), ['a', 'c'], Spectra)
                         ])
def test_slicing_types_cols(spectra, cols, cls):
    # As the spectra is sliced, are the proper subclasses returned?
    assert isinstance(spectra[cols], cls)

@pytest.mark.parametrize("spectra, cols, cls",
                         [(spectra(), 4.4, Spectrum),
                          (spectra(), [2.22, 3.33, 5.5], Spectra),
                          (spectra_metadata(), 4.4, Spectrum),
                          (spectra_metadata(), 'foo', Spectrum),
                          (spectra_metadata(), ['foo', 'bat'], Spectra)
                         ])
def test_slicing_types_loc_indices(spectra, cols, cls):
    assert isinstance(spectra.loc[cols], cls)

def test_false(spectra_multiindex):
    assert isinstance(spectra_multiindex, Spectra)
    assert isinstance(spectra_multiindex[0], Spectra)
    assert isinstance(spectra_multiindex[(0,'a')], Spectrum)

    s = spectra_multiindex.transpose()
    assert isinstance(s, Spectra)
    assert isinstance(s.loc[0], Spectra)
    assert isinstance(s.loc[(0, 'a')], Spectrum)

@pytest.mark.parametrize("spectra",
                         [(spectra_metadata()),
                           (spectra_multiindex())])
def test_get_data_type(spectra):
    assert isinstance(spectra.data, Spectra)

@pytest.mark.parametrize("spectra, col",
                         [(spectra_metadata(), 'a'),
                          (spectra_multiindex(), (0,'a'))])
def test_get_data(spectra, col):
    assert 'foo' not in spectra.data.columns
    assert col in spectra.data.columns


@pytest.mark.parametrize("spectra, clstype",
                         [(spectra_metadata(), Spectra),
                          (spectra_metadata().metadata.iloc[1:], Spectra),
                          (spectra_metadata().metadata.iloc[0], Spectrum),
                          (spectra_metadata().metadata.loc['foo'], Spectrum)])
def test_get_metadata_type(spectra, clstype):
    assert isinstance(spectra, clstype)

def test_get_metadata(spectra_metadata):
    assert spectra_metadata.metadata.equals(Spectra([['a', 'a', 'a', 'a'],
                                                     ['b', 'b', 'b', 'b'],
                                                     ['c', 'c', 'c', 'c']],
                                                     index=['foo', 'bar', 'bat'],
                                                     columns=['a', 'b', 'c', 'd']))

def test_get_metadata_slice(spectra_metadata):
    assert spectra_metadata.iloc[1:].metadata.equals(Spectra([['a', 'a', 'a', 'a'],
                                                     ['b', 'b', 'b', 'b'],
                                                     ['c', 'c', 'c', 'c']],
                                                     index=['foo', 'bar', 'bat'],
                                                     columns=['a', 'b', 'c', 'd']))

    assert spectra_metadata.metadata.iloc[1:].equals(Spectra([['b', 'b', 'b', 'b'],
                                                     ['c', 'c', 'c', 'c']],
                                                     index=['bar', 'bat'],
                                                     columns=['a', 'b', 'c', 'd']))


@pytest.mark.parametrize("spectra, tolerance, expected",
                        [(spectra(), 5, Index([2.22221, 3.33331, 4.4, 5.5])),
                         (spectra(), 1, Index([2.2, 3.3, 4.4, 5.5])),
                         (spectra_multiindex(), 5, Index([2.22221, 3.33331, 4.4, 5.5, 6.6])),
                         (spectra_multiindex(), 1, Index([2.2,  3.3,  4.4,  5.5,  6.6])),
                         (spectra_metadata(), 5, Index([2.22221, 3.33331, 4.4, 5.5, 'foo', 'bar', 'bat'])),
                         (spectra_metadata(), 1, Index([2.2, 3.3, 4.4, 5.5, 'foo', 'bar', 'bat']))])
def test_set_tolerance(spectra, tolerance, expected):
    spectra.tolerance = tolerance
    assert isinstance(spectra, Spectra)
    np.testing.assert_array_equal(spectra.index, expected)

def test_merge(spectra):
    s = spectra.merge(spectra)
    assert isinstance(s, Spectra)

def test_concat(spectra):
    s = concat([spectra, spectra])
    assert isinstance(s, Spectra)

    s = concat([spectra, DataFrame([['a', 'a', 'a', 'a'],
                                    ['b', 'b', 'b', 'b']],
                                    index=['foo', 'bar'],
                                    columns=['a', 'b', 'c', 'd'])])
    assert isinstance(s, Spectra)

    # Test that the merge is a left merge - the df first should result in a df
    s = concat([DataFrame([['a', 'a', 'a', 'a'],
                          ['b', 'b', 'b', 'b']],
                          index=['foo', 'bar'],
                          columns=['a', 'b', 'c', 'd']), spectra])

    assert isinstance(s, DataFrame)

@pytest.mark.parametrize("spectrum, func, cls",
                         [(spectra(), libpyhat.transform.smooth.boxcar, Spectra),
                          (spectra(), libpyhat.transform.smooth.gaussian, Spectra),
                          (spectra_metadata()[['a', 'b']], libpyhat.transform.smooth.boxcar, Spectra),
                          (spectra_metadata()[['a', 'b']], libpyhat.transform.smooth.gaussian, Spectra),
                          (spectra()['a'], libpyhat.transform.smooth.boxcar, Spectrum),
                          (spectra_metadata()['a'], libpyhat.transform.smooth.gaussian, Spectrum),
                          (spectra_metadata()[['a', 'c']], libpyhat.transform.smooth.gaussian, Spectra),
                          (spectra_multiindex(), libpyhat.transform.smooth.boxcar, Spectra),
                          (spectra_multiindex()[(0,'a')], libpyhat.transform.smooth.boxcar, Spectrum),
                          (spectra_multiindex()[0], libpyhat.transform.smooth.gaussian, Spectra)
                         ])
def test_func_return_type(spectrum, func, cls):
    # Using smoothing as a proxy, test the return type for a spectral func
    ss = spectrum.smooth(func=func, preserve_metadata=False)
    ss = spectrum.smooth(func=func, preserve_metadata=True)
    assert isinstance(ss, cls)


@pytest.mark.parametrize("spectrum, func, cls",
                         [(spectra(), libpyhat.transform.continuum.linear, Spectra),
                          (spectra(), libpyhat.transform.continuum.regression, Spectra),
                          (spectra_metadata()['a'], libpyhat.transform.continuum.regression, Spectrum),
                          (spectra_metadata(), libpyhat.transform.continuum.linear, Spectra),
                          (spectra_metadata()[['a', 'b']], libpyhat.transform.continuum.linear, Spectra),
                          (spectra_metadata()[['a', 'b']], libpyhat.transform.continuum.linear, Spectra),
                          (spectra()['a'],  libpyhat.transform.continuum.linear, Spectrum),
                          (spectra_multiindex(), libpyhat.transform.continuum.regression, Spectra),
                          (spectra_multiindex()[(0,'a')], libpyhat.transform.continuum.regression, Spectrum),
                          (spectra_multiindex()[0],  libpyhat.transform.continuum.linear, Spectra),
                          (spectra_metadata()['a'], libpyhat.transform.continuum.regression, Spectrum),
                         ])
def test_continuum_correction_metadata(spectrum, func, cls):
    cc, denom = spectrum.continuum_correct(func=func)
    print(cc)
    print(denom)
    assert isinstance(cc, cls)
