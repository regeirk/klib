# -*- coding: iso-8859-1 -*-
"""Interpolation module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to provide useful and fast interpolation
routines.

AUTHOR
    Sebastian Krieger
    email: naitsabes@regeirk.com

REVISION
    1 (2011-11-11 11:11)

"""

__version__ = '$Revision: 3 $'
# $Source$

__all__ = ['nearest', 'objective_analysis']

import numpy
from scipy.spatial import cKDTree
from common import lon180


def nearest(x, data, xi):
    """Nearest neighbour interpolation.
    
    """
    # Starts kd-tree instance
    if type(x).__name__ == 'list':
        xx, yy = numpy.meshgrid(x[0], x[1])
        X = numpy.array([lon180(xx.flatten()), yy.flatten()])
        zz = data.flatten()

    a, b = data.shape
    v, u = numpy.mgrid[0:a, 0:b]
    V, U = v.flatten(), u.flatten()

    k = cKDTree(X.transpose())

    if type(xi).__name__ == 'list':
        xi, yi = numpy.meshgrid(xi[0], xi[1])
        Xi = numpy.array([lon180(xi.flatten()), yi.flatten()]).transpose()
    else:
        Xi = xi

    D, I, UV = [], [], []
    for item in Xi:
        d, i = k.query(item)
        D.append(d)
        I.append(i)
        UV.append((U[i], V[i]))

    zi = zz[I]
    try:
        zi = zi.reshape(xi.shape)
    except:
        pass

    return zi, I, UV


def objective_analysis(x, y, t, xp, yp, tp, fp, L=None, Lx=None, Ly=None,
    T=None, err=0.1, result='full'):
    """Interpolates scalar variable using objective analysis.

    Parameters
    ----------
    x, y : array like
        Zonal and meridional coordinates for gridded values.
    t : float
        Time in matplotlib's number format.
    xp, yp : array like
        Zonal and meridional coordinates of discrete point measurements.
    fp : array like
        Data values for discrete point measurements.
    L, Lx, Ly : float, optinal
        Spatial correlation lengths. If not given, assumes maximum
        standard deviation of spatial coordinates. Note that either `L`
        or `Lx` and `Ly` should be given.
    T : float, optional
        Temporal correlation length. If not given, assumes standard
        deviation of sampling times.
    err : float, optional
        Normalized RMS error.
    result : string, optional.
        Determines which results are returned. Either `full` for
        interpolated values and errors, `value` for interpolated values,
        or `error` for errors.

    Returns
    -------
    [f, e] : array like
        Depending on parameter `result`, returns either arrays of
        interpolated values and/or relative error.


    References
    ----------
    Emery, W. J. & Thomson, R. E.; Data analysis methods in physical
        oceanography; Elsevier, 2001, 638.

    Bretherton, F. P.; Davis, R. E. & Fandry, C. B.; A technique for
        objective analysis and design of oceanographic experiments
        applied to MODE-73; Deep Sea Research and Oceanographic
        Abstracts, 1976, 23, 559-582.
    
    Barnes interpolation; available at https://en.wikipedia.org/wiki/
        Barnes_interpolation
    
    """
    # Makes sure that input arrays are line vectors and have same size
    _xp, _yp, _tp, _fp = xp.flatten(), yp.flatten(), tp.flatten(), fp.flatten()
    _x, _y = x.flatten(), y.flatten()
    if (_xp.shape != _yp.shape) | (_xp.shape != _fp.shape) | (_xp.shape != _tp.shape):
        raise ValueError('Shape of input data arrays does not match.')
    if (_x.shape != _y.shape):
        raise ValueError('Shape of output grid components do not match.')
    
    # Checks spatial and temporal correlation length parameters.
    if (L == None) & (Lx == None) & (Ly == None):
        Lx = Ly = max(_xp.std(), _yp.std())
    if (L != None):
        if (Lx == None) & (Ly == None):
            Lx = Ly = L
        else:
            raise ValueError('Ambiguous spatial correlation. Give either `L` '\
                             'or `Lx` and `Ly`.')
    if T == None:
        T = _tp.std()
    
    # Array of squared distances between observation points.
    np = _fp.size
    dpx2 = (_xp[:, None]*numpy.ones(np) - _xp[None, :]*numpy.ones(np)) ** 2.
    dpy2 = (_yp[:, None]*numpy.ones(np) - _yp[None, :]*numpy.ones(np)) ** 2.
    dtp2 = (_tp[:, None]*numpy.ones(np) - _tp[None, :]*numpy.ones(np)) ** 2.

    # Array of squared distances between grid points and observation points.
    n = _x.size
    dx2 = (_x[:, None] - _xp[None, :]*numpy.ones(np)) ** 2.
    dy2 = (_y[:, None] - _yp[None, :]*numpy.ones(np)) ** 2.
    dt2 = (t - _tp[None, :]*numpy.ones(np)) ** 2.

    # Calculates weights.
    C = (1 - err) * numpy.exp(-(dx2/Lx**2. + dy2/Ly**2 + dt2/T**2.))
    A = (1 - err) * numpy.exp(-(dpx2/Lx**2. + dpy2/Ly**2. + dtp2/T**2.))
    A = A + err * numpy.eye(np)

    # Calculates interpolated values and error array.
    f, _, _, _ = numpy.linalg.lstsq(A, fp)
    f = C.dot(f)
    e, _, _, _ = numpy.linalg.lstsq(A, C.T)
    e = 1 - (C.T * e).sum(axis=0) / (1 - err)
    #
    if result == 'full':
        return f, e
    elif result == 'value':
        return f
    elif result == 'error':
        return e


def data_in_template(xo, yo, xi, yi, zi, use_sigma=True, masked=True):
    """
    Rearrages data in vector array `zi` to MxN array according to
    coordinates `xo` and `yo`.

    Parameters
    ----------
    xo, yo : array like
        Vector arrays with output coordinates.
    xi, yi : array like
        Vector arrays with input coordinates.
    zi : array like
        Vector array wit input data.
    use_sigma : boolean, optional
    masked : boolean, optional

    Returns
    -------
    zo : array
        Two-dimensional array with rearranged data.
    io : arrays
        Two-dimensional array with indices of input data.
    
    """
    N, M = xo.size, yo.size
    if masked:
        zo = numpy.ma.empty((M, N)) * numpy.nan
    else:
        zo = numpy.empty((M, N)) * numpy.nan
    io = numpy.empty((M, N)) * numpy.nan
    for i, (x, y, z) in enumerate(zip(xi, yi, zi)):
        sel_x = (xo == x)
        sel_y = (yo == y)
        zo[sel_y, sel_x] = z
        io[sel_y, sel_x] = i
    # Convert *y* coordinate to local sigma coordinate, stretches the data to 
    # output range.
    if use_sigma:
        y0, y1 = yo.min(), yo.max()
        for n in range(N):
            sel_y = ~numpy.isnan(zo[:, n])
            if sel_y.sum() == 0:
                continue
            # Calculates sigma coordinates for valid data.
            sigma = ((yo[sel_y] - yo[sel_y].min()) /
                (yo[sel_y].max() - yo[sel_y].min()))
            # Inverts sigma coordinates considering output data range.
            amgis = sigma * y1 + y0
            # Interpolates values to stretched coordinates
            zo[:, n] = numpy.interp(yo, amgis, zo[sel_y, n])
    #
    if masked:
        zo.mask = numpy.isnan(zo.data)
    #
    return zo, io
