# -*- coding: iso-8859-1 -*-
"""Useful and common functions used in the modules."""

__version__ = '$Revision: 2 $'
# $Source$

__all__ = ['distance', 'intersect', 'lon180', 'lon360', 'lon_n', 'meshgrid2',
    'num2latlon', 'num2ymd', 'profiler', 'reglist', 's2hms', 'simpson', 'step',
    'omega', 'daysinyear', 'hoursinday', 'latex_scientific', 'natural_keys']

import re
from dateutil.parser import parse
from matplotlib import dates
from numpy import (angle, arange, array, asarray, ceil, concatenate, cos, 
    cumsum, diff, empty, flatnonzero, floor, intersect1d, iscomplex, loadtxt,
    log10, pi, round, sign, sqrt, zeros, ma, int_)
from pylab import find
from time import time


omega = 7292115e-11 # Earth's rotation rate, according to Moritz (2000)
daysinyear = 365.2421896698 # Wikipedia (?)
hoursinday = 2 * pi / omega / 3600
secondsinday = 86400.

daysinyear_ = lambda y : 365 + int_(((y % 4) == 0) & 
    (((y % 100) != 0) | ((y % 400) == 0)))


def period2dhms(T, result='string'):
    """Converts periods in days to day, hour, minute, second."""
    if type(T) in [float, int]:
        T = [T]
    
    dhms = []
    for t in T:
        days = int(t)
        hours = (t - days) * 24
        minutes = (hours - int(hours)) * 60
        seconds = (minutes - int(minutes)) * 60
        #
        a = array([days, int(hours), int(minutes), seconds])
        #
        if result == 'string':
            # Finds first and last nonzero values:
            sel = flatnonzero(a)
            s = ''
            for i in range(sel[0], sel[-1]+1):
                if i == 0:
                    s += '{:.0f}d'
                elif i == 1:
                    s += '{:.0f}h'
                elif i == 2:
                    s += '{:.0f}m'
                elif i == 3:
                    s += '{:.0f}s'
            s = s.format(*list(a[sel[0]:sel[-1]+1]))
            dhms.append(s)
        else:
            dhms.append(a)
        #
            
    if result == 'string':
        return dhms
    else:
        return asarray(dhms)


def year2num(y):
    """
    Converts decimal year representation of date/time to a floating point
    value representing the number of days since 0001-01-01 00:00:00 UTC.
    Fraction part represents hours, minutes and seconds. For details, 
    please refer to matplotlib.dates module documentation.
    
    """
    if isinstance(y, int) | isinstance(y, float):
        y = asarray([y])
        return_array = False
    else:
        return_array = True
    #
    year = int_(y)
    residual = y - year
    julian_day = residual * daysinyear_(year)
    #
    num = [dates.date2num(dates.datetime.date(year=_y, month=1, day=1)) + _d 
        for _y, _d in zip(year, julian_day)]
    #
    if return_array:
        return asarray(num)
    else:
        return num[0]


def datestr2num(s):
    """
    Converts a string representation of date/time to a floating point
    value representing the number of days since 0001-01-01 00:00:00 UTC.
    Fraction part represents hours, minutes and seconds. For details,
    please refer to matplotlib.dates module documentation.

    """
    # Parse from string to date/time object
    datetime = parse(s)
    # Convert date/time object to matplotlib date/time number
    num = dates.date2num(datetime)
    return num


def _atoi(text):
    return float(text) if text.isdigit() else text


def natural_keys(text):
    """Natural (or human) sorting of string list.

    The solution is based on code provided by 'unutbu' available at
    
    http://stackoverflow.com/questions/5967500/
        how-to-correctly-sort-a-string-with-a-number-inside

    and

    http://www.regular-expressions.info/floatingpoint.html
    
    EXAMPLE
        >>> alist = ['rbio2.4', 'sym6', 'haar', 'bior3.1', 'sym3',
            'rbio1.1', 'db2', 'rbio1.3', 'sym7', 'rbio1.5', 'sym5',
            'sym4', 'sym9', 'sym8', 'rbio3.3', 'rbio3.1', 'rbio3.7',
            'sym10', 'sym11']
        >>> alist.sort(key=klib.common.natural_keys)

    """
    return [_atoi(c) for c in re.split('([-+]?[0-9]*\.?[0-9]+)', text)]


def latex_scientific(f):
    """Converts a floating point number to a LaTeX formatted string in
    scientific notation.

    """
    a = log10(f)
    if (a > 3) | (a <=-1):
        a = round(a)
        scale = 10 ** a
        label = r'%.1f \times 10^{%d}' % (f / scale, a)
    else:
        label = '%.2f' % (f)
    return label


def season(t, hemisphere='S', result='number'):
    """Determines meteorological season for matplotlib time.

    Parameters
    ----------
    t : array like, float, int
        Time array or number in matplotlib format.
    hemisphere : string, optional
        Indicates whether the northern or southern hemisphere is
        considered. Valid values are `N` or `S`.
    result : string, optional
        If set to `number` (default) returns either 1, 2, 3 or 4 for
        winter, spring, summer, fall, respectively. If set to `string`
        returns season string.

    Returns
    -------
    s : array like, string, int
        Meteorological season.

    """
    if isinstance(t, float) | isinstance(t, int):
        t = [t]
    # Converts time to year-month-day format.
    months = asarray([dates.num2date(_t).month for _t in t])
    s = empty(months.shape[0], dtype=int)
    #
    if hemisphere == 'N':
        # Winter: December, January, February
        s[(months==12) | (months==01) | (months==02)] = 1
        # Spring: March, April, May
        s[(months==3) | (months==4) | (months==5)] = 2
        # Summer: June, July, August
        s[(months==6) | (months==7) | (months==8)] = 3
        # Fall: September, October, November
        s[(months==9) | (months==10) | (months==11)] = 4
    elif hemisphere == 'S':
        # Winter: June, July, August
        s[(months==6) | (months==7) | (months==8)] = 1
        # Spring: September, October, November
        s[(months==9) | (months==10) | (months==11)] = 2
        # Summer: December, January, February
        s[(months==12) | (months==01) | (months==02)] = 3
        # Fall: March, April, May
        s[(months==3) | (months==4) | (months==5)] = 4

    if result == 'string':
        'winter', 'spring', 'summer', 'fall'
        S = empty(s.shape, dtype='S6')
        S[s==1] = 'winter'
        S[s==2] = 'spring'
        S[s==3] = 'summer'
        S[s==4] = 'fall'
        if len(s) == 1:
            return S[0]
        else:
            return S
    else:
        if len(s) == 1:
            return s[0]
        else:
            return s


def num2ymd(T, t0=None, **kwargs):
    """
    Converts matplotlib time to a year-month-day array format.

    Parameters
    ----------
    T : array_like
        Array of matplotlib time.
    t0 : float, datetime.date, datetime.datetime, optional
        Reference date to calculate Julian day. If not set, calculates
        Julian day using the first of January for each year.

    Returns
    -------
    YMD : array
        Two-dimensional array with columns indicating respectively
        0--year, 1--month, 2--day, 3--hour, 4--minute, 5--second,
        6--Julian day, 7--ISO week number, and 8--season. Season is
        given as a number from 1 to 4 indicating respectively winter,
        spring, summer and fall.
    
    See also
    --------
        season
    """
    #
    if t0 == None:
        _T0 = dates.datetime.date(year=1, month=1, day=1)
    elif isinstance(t0, float) | isinstance(t0, int):
        _T0 = dates.num2date(t0)
        _t0 = t0 - 1 # Makes sure Julian day starts at 1.
    elif (isinstance(t0, dates.datetime.date) |
         isinstance(t0, dates.datetime.datetime)):
        _T0 = t0
        _t0 = dates.date2num(_T0) - 1 # Makes sure Julian day starts at 1.
    # If checks whether `t0` is an integer. This will be used later to decide 
    # if Julian day will be returned as an integer.
    is_int = isinstance(t0, int)
    #
    Time = []
    for t in T:
        # Converts matplotlib number to datetime object.
        day = dates.num2date(t)
        if is_int:
            t = int(t)
        # Checks if _T0.year is the same as current year for Julian day
        # calculation.
        if (t0 == None) & (_T0.year != day.year):
            _T0 = dates.datetime.date(year=day.year, month=1, day=1)
            _t0 = dates.date2num(_T0) - 1 # Makes sure Julian day starts at 1.
        # Appends current date and time values to output array.
        Time.append([day.year, day.month, day.day, day.hour, day.minute,
            day.second, t-_t0, day.isocalendar()[1], season(t, **kwargs)])
    #
    return asarray(Time)


def num2latlon(x, y, mode='full', padding=True, hemispherefirst=False,
               x180=True, separator='.', precision=.2, dtype='float'):
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
        separator (string, optional) :
            Decimal separator, '.' is the default value.
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
    elif dtype == 'label dms':
        x, y = abs(x), abs(y)
        minute = (x - int(x)) * 60
        second = (minute - int(minute)) * 60
        lon = '%d$^{\circ}$%d\'%.3f\"%s' % (int(x), int(minute), second, EW)
        minute = (y - int(y)) * 60
        second = (minute - int(minute)) * 60
        lat = '%d$^{\circ}$%d\'%.3f\"%s' % (int(y), int(minute), second, NS)
    elif dtype == 'label dm':
        x, y = abs(x), abs(y)
        minute = (x - int(x)) * 60
        lon = '{deg:d}$^{{\circ}}${min:{precision}f}\'{hemis}'.format(
            deg=int(x), min=minute, hemis=EW, precision=precision)
        minute = (y - int(y)) * 60
        lat = '{deg:d}$^{{\circ}}${min:{precision}f}\'{hemis}'.format(
            deg=int(y), min=minute, hemis=NS, precision=precision)
    else:
        raise Warning, 'Type \'%s\' not supported.' % (dtype)
    
    if separator != '.':
        lat = lat.replace('.', separator)
        lon = lon.replace('.', separator)
    
    if mode == 'full':
        if dtype in ['label dms', 'label dm']:
            return '%s; %s' % (lat, lon)
        else:
            return lat + lon
    elif mode == 'each':
        return (lat, lon)
    else:
        raise Warning, 'Mode \'%s\' not supported.' % (mode)


lon_n = lambda x, n: x + (x <= (n-360)) * 360 - (x >= n) * 360
lon180 = lambda x: lon_n(x, 180)
lon360 = lambda x: lon_n(x, 360)



def profiler(N, n, t0, t1, t2):
    """Profiles the module usage.

    Parameters
    ----------
    N, n (int) :
        Number of total elements (N) and number of overall elements
        completed (n).
    t0, t1, t2 (float) :
        Time since the Epoch in seconds for the current module
        (t0), subroutine (t1) and step (t2).
    
    Returns
    -------
    s (string) :
        String containing the analysis result.

    Example
    -------

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

    #if type(x).__name__ in ['list']:
    x = ma.masked_invalid(x).flatten()
    if iscomplex(x).any():
        x = abs(x)
    xmin, xmax, xmean, xstd = x.min(), x.max(), x.mean(), x.std()
    if n:
        xstep = min([(xmax - xmin), (4 * xstd)]) / n
    else:
        xstep = xstd
    base = floor(log10(xstep))
    order = 10 ** base
    i = abs(major - xstep / order)
    try:
        i = find(i == i.min())[0]
    except:
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


def intersect(*args) :
    """Intersects every two arrays and returns the intersected values 
    and data indices.
    
    PARAMETERS
        A, B, ... (array like) :
            Sequence of arrays to be intersected. Note that A is 
            intersected with B, C is intersected with D, etc.
    
    RETURNS
        intersect, idx1, idx1 (array like) :
            The intersection and the indices in each array.
    
    """
    
    n = len(args)
    result = []

    # Walks through every pair of input arrays.
    for i in range(0, n, 2):
        a = args[i]
        b = args[i+1]
        ab = intersect1d(a, b)
        #
        idx = dict((k, i) for i, k in enumerate(a))
        sel1 = [idx[i] for i in ab]
        idx = dict((k, i) for i, k in enumerate(b))
        sel2 = [idx[i] for i in ab]
        #
        result.append(ab)
        result.append(sel1)
        result.append(sel2)
    #
    return result


class etopo:
    p = __file__[:__file__.rfind('/')]
    dat = loadtxt('%s/aux/etopo20.xy.gz' % (p))
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
