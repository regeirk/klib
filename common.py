# -*- coding: iso-8859-1 -*-
"""Useful and common functions used in the modules."""

__version__ = '$Revision: 2 $'
# $Source$

__all__ = ['num2latlon', 'lon180', 'lon360', 'profiler', 's2hms', 'distance',
           'reglist', 'step', 'meshgrid2']

import re
from time import time
from numpy import (angle, array, asarray, concatenate, cos, diff, floor, 
    iscomplex, log10, pi, sign, sqrt, ceil, floor, arange, loadtxt, zeros, 
    cumsum)
from pylab import find

omega = 7292115e-11 # Earth's rotation rate, according to Moritz (2000)
daysinyear = 365.2421896698 # Wikipedia (?)
hoursinday = 2 * pi / omega / 3600

def num2latlon(x, y, mode='full', padding=True, hemispherefirst=False,
               x180=True, dtype='float'):
    """Converts numerical longitude and latitude to text.

    PARAMETERS
        x, y (float) :
            Longitude and latitude in numerical form.
        mode (string, optional) :
            If set to 'full' (default), returns one string only, if set
            to 'each', returns a tuple of two strings containing the
            latitude and longitude formated text.
        padding (boolean, optional) :
            Pads the string with leading zeros to retain length.
            Default is 'True'.
        hemispherefirst (boolean, optional) :
            If set to 'True', puts the hemisphere in from of the
            coordinate value. This is usefull for making sure that
            strings can be ordered by increasing coordinates like in
            directory listings for example. This parameters applies
            only to 'float' or 'int' data type (see bellow).
        x180 (boolean, optional) :
            If set to 'True', forces longitude to be between -180 and 
            +180 degrees. Otherwise, returns longitude ranging from 0
            to 360 degrees.
        dtype (string, optional) :
            Sets the output format of the string. Valid options are
            'float' for 3 decimal points precision, 'int' for integer
            precision, 'label' for integer precision Latex formated
            text and 'label float' for 3 decimal points precision Latex
            formated text. For 'float' and 'int' type, the strings are
            padded with leading zeros. The default type is 'float'.

    RETURNS
        Depending on the selected mode, the funcion returns a single
        string (default) or a tuple of two strings containing the
        formated latitude and longitude coordinates.

    """
    NS, EW = 'N', 'E'
    x = lon180(x)
    if (not x180) and (x < 0):
        x += 360
    if x < 0:
        EW = 'W'
    if y < 0:
        NS = 'S'

    if (x in [0, 180]) or (not x180): EW = ''
    if y == 0: NS = ''

    if dtype in ['float', 'int']:
        if hemispherefirst:
            if dtype == 'float':
                if padding:
                    fmt1 = '%s%07.3f'
                    fmt2 = '%s%06.3f'
                else:
                    fmt1 = '%s%.3f'
                    fmt2 = '%s%.3f'
            elif dtype == 'int':
                if padding:
                    fmt1 = '%s%03d'
                    fmt2 = '%s%02d'
                else:
                    fmt1 = '%s%d'
                    fmt2 = '%s%d'
            lon = fmt1 % (EW, abs(x))
            lat = fmt2 % (NS, abs(y))
        else:
            if dtype == 'float':
                if padding:
                    fmt1 = '%07.3f%s'
                    fmt2 = '%06.3f%s'
                else:
                    fmt1 = '%.3f%s'
                    fmt2 = '%.3f%s'
            elif dtype == 'int':
                if padding:
                    fmt1 = '%s%03d'
                    fmt2 = '%s%02d'
                else:
                    fmt1 = '%s%d'
                    fmt2 = '%s%d'
            lon = fmt1 % (abs(x), EW)
            lat = fmt2 % (abs(y), NS)
    elif dtype == 'label':
        lon = '%d%s%s' % (abs(x), r'$^{\circ}$', EW)
        lat = '%d%s%s' % (abs(y), r'$^{\circ}$', NS)
    elif dtype == 'label float':
        lon = '%.3f%s%s' % (abs(x), r'$^{\circ}$', EW)
        lat = '%.3f%s%s' % (abs(y), r'$^{\circ}$', NS)
    else:
        raise Warning, 'Type \'%s\' not supported.' % (dtype)

    if mode == 'full':
        return lat + lon
    elif mode == 'each':
        return (lat, lon)
    else:
        raise Warning, 'Mode \'%s\' not supported.' % (mode)


lon180 = lambda x: x + (x <= -180) * 360 - (x > 180) * 360
lon360 = lambda x: x + (x <= 0) * 360 - (x >= 360) * 360


def profiler(N, n, t0, t1, t2):
    """Profiles the module usage.

    PARAMETERS
        N, n (int) :
            Number of total elements (N) and number of overall elements
            completed (n).
        t0, t1, t2 (float) :
            Time since the Epoch in seconds for the current module
            (t0), subroutine (t1) and step (t2).
    RETURNS
        s (string) :
            String containing the analysis result.

    EXAMPLE

    """
    n, N = float(n), float(N)
    perc = n / N * 100.
    elap0 = s2hms(time() - t0)[3]
    elap1 = s2hms(time() - t1)[3]
    elap2 = s2hms(time() - t2)[3]
    try:
        togo = s2hms(-(N - n) / n * (time()-t1))[3]
    except:
        togo = '?h??m??s'

    if t0 == 0:
        s = '%.1f%%, %s (%s, %s)' % (perc, elap1, togo, elap2)
    elif (t1 == 0) and (t2 == 0):
        s = '%.1f%%, %s' % (perc, elap0)
    else:
        s = '%.1f%%, %s (%s, %s, %s)' % (perc, elap1, togo, elap0, elap2)
    return s


def s2hms(t) :
    """Converts seconds to hour, minutes and seconds.

    PARAMETERS
        t (float) :
            Seconds value to convert

    RETURNS
        hh, mm, ss (float) :
            Calculated hour, minute and seconds
        s (string) :
            Formated output string.

    EXAMPLE
        hh, mm, ss, s = s2hms(123.45)

    """
    if t < 0:
        sign = -1
        t = -t
    else:
        sign = 1
    hh = int(t / 3600.)
    t -= hh * 3600.
    mm = int(t / 60)
    ss = t - (mm * 60.)
    dd = int(hh / 24.)
    HH = hh - dd * 24.

    if (hh > 0) | (mm > 0):
        s = '%04.1fs' % (ss)
        if hh > 0:
            s = '%dh%02dm%s' % (HH, mm, s)
            if dd > 0:
                s = '%dd%s' % (dd, s)
        else:
            s = '%dm%s' % (mm, s)
    else:
        s = '%.1fs' % (ss)
    if sign == -1:
        s = '-%s' % (s)
    #
    return (hh, mm, ss, s)


def distance(lon, lat, units='nm', origin=False):
    """Calculates the distance between two locations on the globe.

    It uses the 'Plane Sailing' method applying simple geometry to
    calculate the bearing of the path between position pairs.

    Based upon CSIRO, Phil Morgan & Steve Rintoul sw_dist function
    from the Matlab Seawater toolbox.

    PARAMETERS
        lat (array like) :
            Latitude in decimal degrees (+ve N, -ve S)  [ -90: +90].
        lon (array like) :
            Longitude in decimal degrees (+ve E, -ve W) [-180:+180].
        units (string, optional)
            Units of distance either 'nm' (default) for nautical miles
            or 'km' for kilometres.
        origin (boolean, optional)
            If set to true, includes the origin at zero (0) in the
            results.

    RETURNS
        dist (array like) :
            The distance between consecutive locations.
        phase (array like) :
            The angle between consecutive locations.

    EXAMPLE
        dist, phase = distance(lat, lon, units)

    """
    lon = asarray(lon).flatten()
    lat = asarray(lat).flatten()

    if lon.shape != lat.shape:
        raise Exception ('lon and lat must have the same number of vectors.')

    # Constants
    _deg2rad_ = (2. * pi / 360.)
    _rad2deg_ = 1./ _deg2rad_
    _deg2min_ = 60.
    _deg2nm_  = 60.
    _nm2km_   = 1.8520  # As of Pond & Pickard, p. 303

    # And now...
    dlon = diff(lon)
    dlon = (dlon * (abs(dlon) <= 180) + -sign(dlon) * (360 - abs(dlon)) *
           (abs(dlon) > 180))
    latrad = abs(lat * _deg2rad_)
    dep = cos((latrad[1:] + latrad[:-1]) / 2.) * dlon
    dlat = diff(lat)
    dist = _deg2nm_ * sqrt(dlat ** 2 + dep ** 2)  # distance in nautical miles
    if units == 'km':
        dist *= _nm2km_
    phase = angle(dep + 1j * dlat) * _rad2deg_

    if origin:
        dist = concatenate([asarray([0]), dist])
        phase = concatenate([asarray([0]), phase])

    return dist, phase


def reglist(full, pattern, sort=True):
    """ Compares each element in full array to regular expression pattern.

    Returns only the mathed strings and the mathing results.

    PARAMETERS
        full (string, array_like) :
            Array containing the list of elements to be matched.
        patterm (string) :
            Regular expression pattern string.
        sort (boolean) :
            Sets wether list should be sorted (default) or not.

    OUTPUT
        s (string, array_like):
            All the mathing results.
        m (string, array_like):
            The matched strings.

    EXAMPLE
        flist, match = reglist(fulllist, 'xt%s(.*)([NS]).gz')

    """
    N = len(full)
    p = re.compile(pattern)

    if sort:
        full.sort()

    s, m = [], []
    for n, item in enumerate(full):
        match = p.findall(item)
        if match:
            s.append(item)
            m.append(match[0])

    return (s, m)


def step(x, n=None, kind='linear', s0=2., returnrange=False):
    """Calculates ideal intervals.

    Usefull for plotting routines in which major and minor tick steps
    have to be determined.

    PARAMETERS
        x (array like) :
            Input data to determine the step.
        n (float, optional) :
            Number of steps. If ommited then the default step will be
            assumed to be one half of the standard deviation of x.
        kind (string, optional) :
            Determines the kind of application used. Allowed values
            are:
                'linear' -- for regular linear plots
                'polar'  -- for polar geometry (e.g. maps)
        s0 (float, optional) :
            Sets the the minimum and maximum range as 2 * s0 times the
            input data standard deviation.
        returnrange (boolean, optional) :
            If set to true, returns also the range of and the extend.
            This is usefull for setting the color range for contour
            plots.

    RETURNS
        major, minor (float) :
            Major and minor steps calculated from the input parameters.
        range (array like, optional) :
            Value range according to minor scale.
        ticks (array like, optional) :
            Ticks in range according to major scale.
        extend (string, optional) :
            Returns either 'neither', 'both', 'min', 'max' according to
            the data value extension.
    """
    if kind == 'linear':
        major = [1., 1.5, 2., 2.5, 5., 7.5]
        minor = [0.1, 0.5, 1., 0.5, 1., 2.5]
    elif kind == 'polar':
        major = [1, 2, 3, 6, 12, 18]
        minor = [0.5, 1, 0.5, 2, 3, 6]
    else:
        raise Warning, 'Unknown kind \'%s\'' % (kind)

    if type(x).__name__ in ['list']:
        x = asarray(x)
    if iscomplex(x).any():
        x = 0.5* (x.real + x.imag)
    xmin, xmax, xmean, xstd = x.min(), x.max(), x.mean(), x.std()
    if n:
        xstep = (xmax - xmin) / n
    else:
        xstep = xstd / 2
    base = floor(log10(xstep))
    order = 10 ** base
    i = abs(major - xstep / order)
    try:
        i = find(i == i.min())[0]
    except:
        print x
        i = 0
    xmajor = major[i] * order
    xminor = minor[i] * order

    if returnrange == False:
        return (xmajor, xminor)
    elif returnrange == True:
        if not n:
            rmin = xmean - s0 * xstd
            rmax = xmean + s0 * xstd
            if rmin < xmin:
                rmin = xmin
            if rmax > xmax:
                rmax = xmax
        else:
            rmin = xmin
            rmax = xmax

        rmin = floor(rmin / xminor) * xminor
        rmax = ceil(rmax / xminor) * xminor
        xrange_ = arange(rmin, rmax + xminor, xminor)
        xticks = arange(rmin, rmax + xmajor, xmajor)

        if (xmin < rmin) & (xmax > rmax):
            extend = 'both'
        elif (xmin < rmin) & (xmax <= rmax):
            extend = 'min'
        elif (xmin >= rmin) & (xmax > rmax):
            extend = 'max'
        elif (xmin >= rmin) & (xmax <= rmax):
            extend = 'neither'
        else:
            raise Warning, 'Unable to determine extend'

        return (xmajor, xminor, xrange_, xticks, extend)
    else:
        raise Warning, 'Wrong return parameter.'


def meshgrid2(*arrs):
    """
    Return coordinate matrices from N coordinate vectors.
    
    REFERENCES
        http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d    
    """
    
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans[::-1])


def simpson(y):
    """Simpson-rule column-wise cumulative summation.

    Numerical approximation of a function F(x) such that 
    Y(X) = dF/dX.  Each column of the input matrix Y represents
    the value of the integrand  Y(X)  at equally spaced points
    X = 0,1,...size(Y,1).

    The output is a matrix  F of the same size as Y.
    The first row of F is equal to zero and each following row
    is the approximation of the integral of each column of matrix
    Y up to the givem row.

    simpson assumes continuity of each column of the function Y(X)
    and uses Simpson rule summation.

    Similar to the command F = cumsum(Y), exept for zero first
    row and more accurate summation (under the assumption of
    continuous integrand Y(X)).

    See also numpy.cumsum, numpy.sum, numpy.trapz

    REFERENCES
    
    Based upon http://www-pord.ucsd.edu/~matlab/stream.htm
    """
    # 3-points interpolation coefficients to midpoints.
    # Second-order polynomial (parabolic) interpolation coefficients
    # from  Xbasis = [0 1 2]  to  Xint = [.5 1.5]
    c1, c2, c3 = 3./8., 6./8., -1./8.;

    # Checks input arguments
    y = asarray(y)
    # Determine the size of the input and make column if vector
    ist = False                   # if to be transposed
    a = y.shape[0]
    if a == 1:
        ist = True
        y = y.transpose();
        a = y.shape[0]
    f = zeros(y.shape);
    
    # If only 2 elements in columns - simple sum divided by 2
    if a == 2:
        f[1, :] = (y[0, :]+y[1]) / 2;
    # If more than two elements in columns - Simpson summation
    else:
        # Interpolate values of Y to all midpoints
        n = arange(0, a-2)
        f[n+1, :] = c1 * y[n, :] + c2 * y[n+1, :] + c3 * y[n+2, :]
        f[n+2, :] = f[n+2, :] + c3 * y[n, :] + c2 * y[n+1, :] + c1 * y[n+2, :]
        f[[1], :] = f[[1], :] * 2
        f[[a-1], :] = f[[a-1], :] * 2
        # Now Simpson (1,4,1) rule
        n = arange(1, a)
        f[n, :] = 2 * f[n, :] + y[n-1, :] + y[n, :]
        # Cumulative sum, 6 - denom. from the Simpson rule
        f = cumsum(f, axis=0) / 6;

    # Transpose output if necessary
    if ist:
        f = f.transpose()
    return f


class etopo:
    p = __file__[:__file__.rfind('/')]
    dat = loadtxt('%s/etopo20.xy.gz' % (p))
    x = dat[0, 1:]
    y = dat[1:, 0]
    z = dat[1:, 1:]
    del p


basins = dict(
    atl = dict(
        id = 1,
        longname = 'Atlantic Ocean',
        shortname = 'Atlantic'
    ),
    pac = dict(
        id = 2,
        longname = 'Pacific Ocean',
        shortname = 'Pacific',
    ),
    ind = dict(
        id = 3,
        longname = 'Indian Ocean',
        shortname = 'Indian',
    ),
    car = dict(
        id = 4,
        longname = 'Caribbean Sea',
        shortname = 'Caribbean',
    ),
    mex = dict(
        id = 5,
        longname = 'Gulf of Mexico',
        shortname = 'Gulf of Mexico',
    ),
    tas = dict(
        id = 6,
        longname = 'Tasman Sea',
        shortname = 'Tasman Sea',
    ),
    ben = dict(
        id = 7,
        longname = 'Bay of Bengal',
        shortname = 'Bay of Bengal',
    )
)
