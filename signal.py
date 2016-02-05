# -*- coding: utf-8 -*-
"""Signal analysis.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to give a set of functions relevant to the
oceanographic community.

AUTHOR
    Sebastian Krieger
    email: solutions@nublia.com

REVISION
    1 (2013-12-02 15:37 DST)

"""
from __future__ import division

__version__ = '$Revision: 1 $'
# $Source$

__all__ = ['fftnegativeshift', 'next_power_of_two', 'power_spectral_density',
    'errors', 'climatology']

from matplotlib.dates import drange, num2date
from numpy import (arange, arctan2, array, asarray, ceil, concatenate, cos, 
    deg2rad, digitize, empty, exp, flatnonzero, fliplr, flipud, floor, 
    iscomplex, isnan, linalg, log2, ma, median, nan, ones, rad2deg, sin, take, 
    unique, zeros)
from scipy import interpolate
from scipy.fftpack import fft2, fftfreq, fftshift
from scipy.misc import factorial
from scipy.signal import fftconvolve, detrend
from scipy.stats import binned_statistic

import common


def speeddir_to_vector(m, a, dtype='from'):
    """
    Converts polar vector notation to complex vector notation, assuming
    geographical angle origin convention at north.

    Parameters
    ----------
    m : array like
        Magnitude of the speed
    a : array like
        Angle in degrees.
    dtype : string, optional
        If `from`, uses convention that vector indicates direction from
        which it is comming (e.g. wind from direction). If `to`
        indicates direction to which it is pointing (e.g. currents).

    Returns
    -------
    uv : array like
        Vector in complex notation (u + 1j*v).
    
    """
    # Converts angles from degrees to radians, assuming geographical angle
    # origin convention at north rotating clockwise. Check 
    # <http://wx.gmu.edu/dev/clim301/lectures/wind/wind-uv.html> for further
    # explanations.
    if dtype == 'from':
        a = deg2rad(270 - a)
    elif dtype == 'to':
        a = deg2rad(90- a)
    else:
        raise ValueError('Invalid data type `{}`.'.format(dtype))
    # Calcultes vector coordinates.
    return m * cos(a) + 1j * m * sin(a)


def vector_to_speeddir(uv, dtype='from'):
    """
    Converts complex vector to polar vector, assuming geographical angle
    origin convention at north.
    
    Parameters
    ----------
    uv : array like
        Array of cartesian vector components in complex notation.
    dtype : string, optional
        If `from`, uses convention that vector indicates direction from
        which it is comming (e.g. wind from direction). If `to`
        indicates direction to which it is pointing (e.g. currents).
    
    Returns
    -------
    m, a : array like
        Magnitude and angle.

    """
    m = (uv.real**2 + uv.imag**2)**0.5
    a = rad2deg(arctan2(uv.imag, uv.real))
    # Converts angle from mathematical direction to meteorological direction.
    # Check <http://wx.gmu.edu/dev/clim301/lectures/wind/wind-uv.html> for 
    # further explanations.
    if dtype == 'from':
        a = common.lon360(270 - a)
    elif dtype == 'to':
        a = common.lon360(90 - a)
    else:
        raise ValueError('Invalid data type `{}`.'.format(dtype))
    #
    return m, a


def derivative(A, axis, p=1, q=3, mask=None):
    """Higher order differential.

    Calculates the p-th derivative of `A` using stencils of width n.
    
    The gradient is computed using central differences in the interior
    and first differences at the boundaries. The returned gradient hence
    has the same shape as the input array.
    
    Parameters
    ----------
    A : array like
        Data to calculate the derivate.
    axis : array like
        Axis onto which the derivative will be calculated. Must have
        same size as `A`.
    p : integer, optional
        Order of the derivative to be calculated. Default is to
        calculate the first derivative (p=1). 2*n-p+1 gives the relative
        order of the approximation.
    q (integer, optional) :
        Length of the stencil used for centered differentials. The
        length has to be odd numbered. Default is q=3.
    mask : array like, optional :
    
    Returns
    -------
    dA : array like
        The calculated derivate with same dimensions as A

    References
    ----------
    Cushman-Roisin, B. & Beckers, J.-M. Introduction to geophysical
    fluid dynamics: Physical and numerical aspects. Academic Press,
    2011, 101, 828.

    Arbic, Brian B. Scott, R. B.; Chelton, D. B.; Richman, J. G. and
    Shriver, J. F. Effects of stencil width on surface ocean geostrophic
    velocity and vorticity estimation from gridded satellite altimeter
    data. Journal of Geophysical Research, 2012, 117, C03029.
        
    """
    # Checks size of A and axis.
    if A.shape != axis.shape:
        raise ValueError('Data and axis array do not have same shape.')

    # Makes shure the length of the stencil is odd numbered.
    q += (q % 2) - 1
    # Calculate left and right stencils.
    q_left = (q - 1) / 2
    q_right = (q - 1) / 2

    # Calculate stencil coefficients
    coeffs = _derivative_stencil_coefficients(axis, p=p, q=q)

    # Calculate the p-th derivative
    I = A.size
    dA = zeros(I)
    for i in arange(q) - q_left:
        if i < 0:
            u, v = -i, I
        else:
            u, v = 0, I - i
        dA[u:v] += (coeffs[u:v, i+q_left] * A[u+i:v+i])
    
    return dA


def _derivative_stencil_coefficients(axis, p=1, q=3):
    """Calculates the coefficients needed for the derivative.
    
    Parameters
    ----------
    axis : array like
        Axis onto which the derivative will be calculated.
    p : integer, optional
        Order of the derivative to be calculated. Default is to
        calculate the first derivative (p=1). 2*n-p+1 gives the relative
        order of the approximation.
    q : integer, optional
        Length of the stencil used for centered differentials. The
        length has to be odd numbered. Default is q=3.
    
    Returns
    -------
    Coefficients (a_q) needed for the linear combination of `q` points
    to get the first derivative according to Arbic et al. (2012)
    equations (20) and (22). At the boundaries forward and backward
    differences approximations are calculated.
    
    References
    ----------
    Cushman-Roisin, B. & Beckers, J.-M. Introduction to geophysical
    fluid dynamics: Physical and numerical aspects Academic Press, 2011,
    101, 828.
    
    Arbic, Brian B. Scott, R. B.; Chelton, D. B.; Richman, J. G. &
    Shriver, J. F. Effects of stencil width on surface ocean geostrophic
    velocity and vorticity estimation from gridded satellite altimeter
    data. Journal of Geophysical Research, 2012, 117, C03029.
    
    """
    # Calculate left and right stencils.
    q_left = (q - 1) / 2
    q_right = (q - 1) / 2 + 1
    #
    I = axis.size

    # Constructs matrices according to Cushman-Roisin & Beckers (2011)
    # equations (1.25) and adapted for variable grids as in Arbic et 
    # al. (2012), equations (20), (22). The linear system of equations
    # is solved afterwards.
    coeffs = zeros((I, q))
    smart_coeffs = dict()
    for i in range(I):
        A = zeros((q, q))
        #
        if i < q_left:
            start = q_left - i
        else:
            start = 0
        if i > I - q_right:
            stop = i - (I - q_right)
        else:
            stop = 0
        #
        A[0, start:q+stop] = 1
        da = axis[i-q_left+start:i+q_right-stop] - axis[i]
        da_key = str(da)
        #
        if da_key not in smart_coeffs.keys():
            for h in range(1, q):
                A[h, start:q-stop] = da ** h
            B = zeros((q, 1))
            # This tells where the p-th derivative is calculated
            B[p] = factorial(p)
            C = linalg.solve(A[:q-start-stop, start:q-stop], 
                B[:q-(start+stop), :])
            #
            smart_coeffs[da_key] = C.flatten()
        #
        coeffs[i, start:q-stop] = smart_coeffs[da_key]
    #
    return coeffs


def continuous_timeseries(dat, dt, fill=None, max_gap=12, mask=True, skip=[],
    skip_fill=[]):
    """Creates continuous data array.

    Parameters
    ----------
    dat : structured array
        Input data.
    dt : datetime.timedelta
        Time step.
    fill : string, optional
        If set to `climatology`, fills data gaps with climatological
        data. If set to `interpolate` fills data gaps with cubic spline
        interpolated data.
    max_gap : integer, optional
        Maximum gap size to fill with either climatology or interpolated
        values, according to `fill` parameter.
    mask : boolean, optional
        If both `climatology` and `mask` are true, masks filled gaps.
    skip : list, optional
        List of fields to skip. Default is empty.
    skip_fill : list, optional
        List of fields to skip gap filling. Default is empty.

    Returns
    -------
    out : structured array
        Reordered data.
    [clima] : dictionary of arrays, optional
        If `climatology` is true, returns also the climatologies for
        each input fields.

    """
    # Some parameters and variable initialization. Determines the sampling
    # interval, the size of the new array. Then creates a bin array in which
    # original data will be fit and finally initializes new data array.
    # Creates an array of indices of the new data.
    T0 = num2date(dat['time'][0])
    T1 = num2date(dat['time'][-1])
    t = drange(T0, T1, dt)
    dt = (t[1] - t[0])
    dt2 = dt * 0.5
    N = t.size
    bins = arange(t[0] - dt2, t[-1] + 2 * dt2, dt)
    out = ma.empty(N, dtype=dat.dtype)
    out_idx = arange(N)
    if fill == 'climatology':
        clima = dict()

    # Determines to which bin each data point belongs to.
    bin_sel = digitize(dat['time'], bins)
    
    # Walks through each record and determines each bin mean.
    for i, field in enumerate(dat.dtype.names):
        if field in skip:
            continue
        if field == 'time':
            out['time'] = t
        else:
            try:
                sel = ~(dat[field].mask | isnan(dat[field]))
            except:
                sel = ~(isnan(dat[field]))
            
            is_complex = iscomplex(dat[field][sel]).any()
            
            if is_complex:
                out_real, _, bin_nr = binned_statistic(dat['time'][sel], 
                    dat[field][sel].real, statistic='mean', bins=bins)
                out_imag, _, bin_nr = binned_statistic(dat['time'][sel], 
                    dat[field][sel].imag, statistic='mean', bins=bins)
                out[field] = out_real + 1j * out_imag
                #bin_nr = list(set(bin_nr_real) | set(bin_nr_imag))
            else:
                out[field], _, bin_nr = binned_statistic(dat['time'][sel], 
                    dat[field][sel], statistic='mean', bins=bins)
            # Finds out where the gaps are and fills them either with the
            # climatological means or simply masks them. It is important to
            # note that the bin number starts counting at one.
            sel = list(set(out_idx) - set(unique(bin_nr) - 1))
            sel.sort()
            if ((fill in ['climatology', 'interpolate']) &
                                                (field not in skip_fill)):
                # Separates gaps in order to determine gap size
                _spl = [0] + [i for i in range(1, len(sel)) \
                    if (sel[i] - sel[i-1]) > 1] + [None]
                gaps = []
                for i in range(1, len(_spl)):
                    if len(sel[_spl[i-1]:_spl[i]]) <= max_gap:
                        gaps += sel[_spl[i-1]:_spl[i]]
                #
                if fill == 'climatology':
                    # Calculates dayly climatological mean.
                    clima[field] = climatology(dat['time'], dat[field], 
                        major='month', minor='hour', result='timeseries', 
                        t_out=t)
                    # Fills invalid data with climatology and masks filled gaps
                    # according to `mask` input parameter.
                    out[field].data[gaps] = clima[field][gaps]
                elif fill == 'interpolate':
                    if is_complex:
                        out_real = _interpolate(t[gaps], dat['time'], 
                            dat[field].real)
                        out_imag = _interpolate(t[gaps], dat['time'], 
                            dat[field].imag)
                        out[field].data[gaps] = out_real + 1j * out_imag
                    else:
                        out[field].data[gaps] = _interpolate(t[gaps], 
                            dat['time'], dat[field])
                # Masks all data not in continuous timeseries and sets mask in 
                # gaps according to `mask` parameter.
                out[field].mask[sel] = True
                out[field].mask[gaps] = mask
            else:
                # Mask gaps in new dataset. 
                out[field].data[sel] = 0
                out[field].mask[sel] = True
    #
    if fill == 'climatology':
        return out, clima
    else:
        return out


def circular_mean(a, A=2**0.5, dtype='mean'):
    """Calculates the average of an angle.

    Parameters
    ----------
        a : array_like
            List of angles.
        A : array_like, float, optional
            In case of vector fields, gives the modulus of each vector.
        dtype : string, optional
            Accepted values are `mean` (default), `median` to either
            calculate mean or median respectively.

    Returns
    -------
        <a> : array_like
            Average angle
        <A> : array_like
            If `A` is an array, then retunrs the average modulus.
    
    """
    # Converts array of angles in array of radians.
    a = deg2rad(a)
    # Creates array of vectors using complex notation, and calculates mean.
    z = A * cos(a) + 1j*A * sin(a)
    if dtype == 'mean':
        Z = z.mean()
    elif dtype == 'median':
        Z = median(z)
    else:
        raise ValueError('Invalid mean type `{}`.'.format(dtype))
    #
    if isinstance(A, float) | isinstance(A, int):
        return rad2deg(arctan2(Z.imag, Z.real))
    else:
        return rad2deg(arctan2(Z.imag, Z.real)), abs(Z)


def climatology(t, f, major='month', minor='hour', result='climatology',
    kind='data', t_out=None, averaging='mean', **kwargs):
    """Calculates climatology from data.

    Parameters
    ----------
        t, f : array_like
            Time and value arrays for which climatology will be
            calculated. Time has to be in matplotlib's number format.
        major, minor : string, optional
            Sets the major and minor scales for calculation. Scales can
            be either `year`, `month`, `day`, `hour`, `julian`, `week`
            or `season`.
        result : string, optional
            If `climatology` (default) returns climatology array
            according to major and minor scales. If `all`, returns in
            addition the major and minor indices for each time in
            climatology array. If `timeseries`, returns only climatology
            for every time t.
        kind : string, optional
            If set to `data`, adjusts minor and major axis to data. If
            set to `valid`, adjusts minor and major axis to valid 
            values, irrespective if they occur in the dataset.
        averaging: string, optional
            If set to `mean` (default), calculates mean values. If set
            to `sum`, sums all values.
        t_out : array_like, optional
            If result is `timeseries`, then returns climatology for time
            give in `t_out` instead of `t`.

    Returns
    -------
        clima : array_like
            Climatology of input value.
        stdev : array_like
            Standard deviation in climatology.
        count : array_like
            Number of occurences.
        ind_j, ind_i : array_like
            Major and minor indices for each time.
    
    """
    # Converts time array to year-month-day array.
    YMD = common.num2ymd(t, **kwargs)
    try:
        YMD_out = common.num2ymd(t_out, **kwargs)
    except:
        pass
    #
    def _set_column(name):
        if name == 'year':
            return 0
        elif name == 'month':
            return 1
        elif name == 'day':
            return 2
        elif name == 'hour':
            return 3
        elif name == 'julian':
            return 6
        elif name == 'week':
            return 7
        elif name == 'season':
            return 8
        else:
            raise ValueError('Invalid column name `{}`.'.format(name))
    #
    def _set_axis(name, kind, data):
        if kind == 'data':
            return unique(data)
        elif kind == 'valid':
            if name == 'year':
                return arange(data.min(), data.max()+1)
            if name == 'month':
                return arange(0, 12) + 1
            elif name == 'day':
                return arange(0, 31) + 1
            elif name == 'hour':
                return arange(0, 24)
            elif name == 'julian':
                return arange(0, 366) + 1
            elif name == 'week':
                return arange(53) + 1
            elif name == 'season':
                return arange(4) + 1
            else:
                raise ValueError('Invalid column name `{}`.'.format(name))
        else:
            raise ValueError('Invalid kind `{}`.'.format(kind))
    #
    major_col = _set_column(major)
    minor_col = _set_column(minor)
    #
    if major_col == minor_col:
        major_col = -1
        Major = array([nan])
    else:
        Major = _set_axis(major, kind, YMD[:, major_col])
    Minor = _set_axis(minor, kind, YMD[:, minor_col])
    M, N = Major.size, Minor.size
    #
    clima = zeros((M, N), dtype=f.dtype)
    count = zeros((M, N))
    stdev = zeros((M, N))
    ind_i = -ones(t.size, dtype=int)
    ind_j = -ones(t.size, dtype=int)
    try:
        ind_i_out = -ones(t_out.size, dtype=int)
        ind_j_out = -ones(t_out.size, dtype=int)
    except:
        pass
    #
    for m, major in enumerate(Major):
        for n, minor in enumerate(Minor):
            if major_col == -1:
                sel = (YMD[:, minor_col] == minor)
            else:
                sel = (YMD[:, major_col] == major) & \
                      (YMD[:, minor_col] == minor)
            if averaging == 'mean':
                clima[m, n] = f[sel].mean()
            elif averaging == 'sum':
                clima[m, n] = f[sel].sum()
            else:
                raise ValueError('Invalid averaging `{}`.'.format(averaging))
            count[m, n] = sel.sum()
            stdev[m, n] = f[sel].std()
            ind_j[sel] = m
            ind_i[sel] = n
            try:
                if major_col == -1:
                    sel = (YMD_out[:, minor_col] == minor)
                else:
                    sel = (YMD_out[:, major_col] == major) & \
                          (YMD_out[:, minor_col] == minor)
                ind_j_out[sel] = m
                ind_i_out[sel] = n
            except:
                pass
    #
    if result == 'climatology':
        return Major, Minor, clima, stdev, count
    elif result == 'all':
        return Major, Minor, clima, stdev, count, ind_j, ind_i
    elif result == 'timeseries':
        try:
            return clima[ind_j_out, ind_i_out]
        except:
            if (ind_i<0).any() | (ind_j<0).any():
                raise ValuError('Some data in input array are undetermined.')
            return clima[ind_j, ind_i]


def errors(a, b, result='dict'):
    """
    Returns the mean error, mean average error and root-mean-square
    error between `a` and `b`.

    The mean error (ME), mean average error (MAE) and root-mean-square
    error (RMSE) are defined as:

    ME = \frac{1}{N} \sum\limits_{n=1}^{N} \eta(n) - \eta_C(n)
    MAE = \frac{1}{N} \sum\limits_{n=1}^{N} \left| \eta(n) -
        \eta_C(n) \right|
    RMSE = \sqrt{\frac{1}{N} \sum\limits_{n=1}^{N} \left[ \eta(n) -
        \eta_C(n) \right]^2}
    
    Parameters
    ----------
    a, b : array_like
        Samples to compare. Note that both arrays should have the same
        shape.
    result : string, optional
        If set to `dict` (default) returns a dictionary containing
        calculated parameters.

    Returns
    """
    # Makes sure input parameters are arrays and checks if they have same
    # shape.
    a = asarray(a)
    b = asarray(b)
    if a.shape != b.shape:
        raise ValueError('Shape of arrays do not match.')

    # Calculates errors.
    ME = (a - b).sum() / a.size
    MAE = abs(a - b).sum() / a.size
    RMSE = (((a - b)**2.).sum() / a.size)**0.5

    if result == 'dict':
        return dict(ME=ME, MAE=MAE, RMSE=RMSE)
    else:
        return ME, MAE, RMSE


def fftnegativeshift(x):
    """
    Shift the zero-frequency component to the center of the spectrum,
    inverting positive with negative frequencies.

    This function swaps half-spaces for all axes listed (defaults to
    all). Note that ``y[0]`` is the Nyquist component only if ``len(x)``
    is even.

    Note that this function works only with two-dimensional input
    arrays.

    PARAMETERS
        x (array like) :
            Input array.
    
    RETURNS
        y (array like) :
            The shifted array.
    
    """
    tmp = asarray(x)
    if tmp.ndim != 2:
        raise ValueError('Input array is not two-dimensional.')
    else:
        axes = range(tmp.ndim)
    #
    y = tmp.copy()
    #
    for k in axes:
        n = tmp.shape[k]
        p2 = (n + 1) // 2
        print k, n, p2
        mylist = concatenate((arange(p2, n), arange(p2)))
        y = take(y, mylist, k)
        if k == 1:
            y[:p2, :] = fliplr(y[:p2, :])
    #
    print y[:p2, :]
    return y


def next_power_of_two(A):
    """Takes every element in A and calculates the next integer power of 
    two.
    
    PARAMETERS
        A (integer, float, array like) :
            Values to calculate the next power of two.
    
    RETURNS
        B (integer, array like) :
            Next integer power of two.
    
    """
    t = type(A)
    a = asarray(A)
    b = array([int(2**ceil(log2(i))) for i in a.flatten()])
    b.shape = a.shape
    return t(b)


def power_spectral_density(h, shape=None, delta=(1., 1.), mirror='',
    negativeshift=False, method='ET01', window=None, detrend=False,
    result='full'):
    """Calculates the two dimensional power spectral density (2D-PSD).

    The 2D-PSD is defined as the squared amplitude per unit area of the
    spectrum of a surface height map
    
    PARAMETERS
        h (array like) :
        shape (list or tuple, optional) :
        delta (list or tuple, optional) :
            List containing the y-axis and x-axis sampling interval.
        positivex, positivey (boolean, optional) :
        negative (string, optional) :
        method (string, optional) :
        windows (array like, optional) :
        detrend (boolean, optional) :
        result (string, optional) :

    RETURNS
        freqx, freqy (array like) :
        PSD (array like) :
        phase (array like) :

    REFERENCES
        Emery, W. J. & Thomson, R. E. Data analysis methods in physical
        oceanography Elsevier, 2001, 638, section 5.6.3.2
        
        Sidick, E. Power spectral density specification and analysis
        of large optical surfaces SPIE Europe Optical Metrology, 2009,
        73900L-73900L.
    
    """
    if h.ndim != 2:
        raise ValueError('This function works only with two-dimensional '
            'arrays.')
    if window != None:
        h = fftconvolve(window/window.sum(), h, mode='same')
    if shape == None:
        J, I = next_power_of_two(h.shape)
    else:
        J, I = shape
    dy, dx = delta
    #
    try:
        H = h.data * ~h.mask
        H[isnan(H)] = 0
        FFT = fft2(H, (J, I))
    except:
        FFT = fft2(h, (J, I))
    freqx = fftfreq(I, dx)
    freqy = fftfreq(J, dy)

    if negativeshift:
        j = (J + 1) // 2
        FFT[1:j, :] = fliplr(FFT[1:j, :])
    
    if mirror.find('x') >= 0:
        fft = zeros((FFT.shape[0], FFT.shape[1]/2), dtype=complex)
        fft[:, 0] = FFT[:, 0]
        fft[:, 1:] = FFT[:, 1:I/2] + fliplr(FFT[:, I/2+1:])
        freqx = freqx[:I/2]
        FFT = fft.copy()
    else:
        FFT = fftshift(FFT, axes=1)
        freqx = fftshift(freqx) 
    if mirror.find('y') >= 0:
        fft = zeros((FFT.shape[0]/2, FFT.shape[1]), dtype=complex)
        fft[0, :] = FFT[0, :]
        fft[1:, :] = FFT[1:J/2, :] + flipud(FFT[J/2+1:, :])
        freqy = freqy[:J/2]
        FFT = fft.copy()
    else:
        FFT = fftshift(FFT, axes=0)
        freqy = fftshift(freqy)
    
    if method == 'ET01':
        PSD = (FFT * FFT.conj()).real / (dx * dy * J * I)
    elif method == 'S09':
        PSD = (FFT * FFT.conj()).real * (dx * dy) / (J * I)
    else:
        raise ValueError ('Unrecognized method \'%s\'.' % (method))
    phase = arctan2(FFT.imag, FFT.real)

    if result == 'full':
        return freqx, freqy, PSD, phase
    elif result == 'psd':
        return PSD


def bin_average(x, y, dx=1., bins=None, nstd=2., interpolate=True, k=3, s=None,
    extrapolate='repeat', mode='mean'):
    """Calculates bin average from input data.

    Inside each bin, calculates the average and standard deviation, and
    selects only those values inside the confidence interval given in
    `nstd`. Finally calculates the bin average using spline
    interpolation at the middle points in each bin. Linearly
    extrapolates values outside of the data boundaries.

    Parameters
    ----------
    x : array like
        Input coordinate to be binned. It has to be 1-dimensional.
    y : array like
        The data input array.
    dx: float, optional
    bins : array like, optional
        Array of bins. It has to be 1-dimensional and strictly
        increasing.
    nstd : float, optional
        Confidence interval given as number of standard deviations.
    interpolate : bool, optional
        If `True` (default), interpolates averages in bins to central
        bin points.
    k : int, optional
        Specifies the order of the interpolation spline. Default is 3,
        `cubic`.
    s : float, optional
        Positive smoothing factor used to choose the number of knots.
    extrapolate : string, bool, optional
        Sets if averaging outside data boundaries should be
        extrapolated. If `True` or `linear`, extrapolates data linearly,
        if `repeat` (default) repeats values from nearest bin.
    mode : string, optional
        Sets averaging mode: `mean` (default), `median`.

    Returns
    -------
    bin_x : array like
        Coordinate at the center of the bins.
    bin_y : array like
        Interpolated array of bin averages.
    avg_x : array like
        Average coordinate in each bin.
    avg_y : array like
        Average values inside each bin.
    std_x : array like
        Coordinate standard deviation in each bin.
    std_y : array like
        Standard deviation in each bin.
    min_y : array like
        Minimum values in each bin.
    max_y : array like
        Maximum values in each bin.

    """
    # If no bins are given, calculate them from input data.
    if bins == None:
        x_min = floor(x.min() / dx) * dx
        x_max = 0. # numpy.ceil(x.max() / dx) * dx
        bins = arange(x_min-dx, x_max+dx, dx) + dx/2
    # Checks if bin array is strictly increasing.
    if not all(x < y for x, y in zip(bins, bins[1:])):
        raise ValueError('Bin array must be strictly increasing.')
    # Ensures that input coordinate `x` is monotonically increasing.
    _i = x.argsort()
    x = x[_i]
    y = y[_i]
    # Data types
    dtype_x = x.dtype
    dtype_y = y.dtype
    # Some variable initializations
    nbins = len(bins) - 1
    ndata = len(y)
    Sel = zeros(ndata, dtype=bool)
    bin_y = empty(nbins, dtype=dtype_y)
    avg_x = empty(nbins, dtype=dtype_x) * nan 
    avg_y = empty(nbins, dtype=dtype_y) * nan
    std_x = empty(nbins, dtype=dtype_x) * nan
    std_y = empty(nbins, dtype=dtype_y) * nan
    min_y = empty(nbins, dtype=dtype_y) * nan
    max_y = empty(nbins, dtype=dtype_y) * nan
    #
    for i in range(nbins):
        # Calculate averages inside each bin in two steps: (i) calculate
        # average and standard deviation; (ii) consider only those values
        # within selected standard deviation range.
        for step in xrange(2):
            if step == 0:
                # Selects data inside bins and calculates average and standard
                # deviation inside bin.
                sel = (x >= bins[i]) & (x < bins[i+1])
            elif (step == 1) & (sel.sum() > 0):
                # Selects data within selected standard deviation or single
                # data in current bin.
                sel = sel & ((y >= (_avg_y - nstd * _std_y)) &
                    (y < (_avg_y + nstd * _std_y)))
            #
            sel_sum = sel.sum()
            if sel_sum > 1:
                if mode == 'mean':
                    _avg_x = x[sel].mean()
                    _avg_y = y[sel].mean()
                elif mode == 'median':
                    _avg_x = median(x[sel])
                    _avg_y = median(y[sel])
                else:
                    raise ValueError('Invalid mode `{}`.'.format(mode))
                _std_x = x[sel].std()
                _std_y = y[sel].std()
                _min_y = y[sel].min()
                _max_y = y[sel].max()
                # This skips the next step if standard deviation is zero.
                if _std_y == 0:
                    break
            else:
                if sel_sum == 1:
                    _avg_x, _avg_y = x[sel][0], y[sel][0]
                    _std_x, _std_y = 0, 0
                    _min_y, _max_y = nan, nan
                else:
                    _avg_x, _avg_y = nan, nan
                    _std_x, _std_y = nan, nan
                    _min_y, _max_y = nan, nan
                # This skips the next step and goes straight to bin averaging.
                break
        
        #print i, sel_sum, _avg_x, _avg_y
        #
        avg_x[i] = _avg_x
        avg_y[i] = _avg_y
        std_x[i] = _std_x
        std_y[i] = _std_y
        min_y[i] = _min_y
        max_y[i] = _max_y

    # Interpolates selected data to central data point in bin using spline.
    if interpolate:
        bin_x = (bins[1:] + bins[:-1]) * 0.5
        #bin_y = extrapolate(bin_x, x[Sel], y[Sel], k=k, s=s)
        bin_y = _interpolate(bin_x, avg_x, avg_y, k=k, s=s,
            outside=extrapolate)
        #
        return bin_x, bin_y, avg_x, avg_y, std_x, std_y, min_y, max_y
    else:
        return avg_x, avg_y, std_x, std_y, min_y, max_y


def _interpolate(xi, xd, yd, k=3, s=None, outside='repeat'):
    """."""
    if s == None:
        s = xi.size
    yi = empty(xi.shape) * nan
    # Checks for NaN's in data.
    sel = ~(isnan(xd) | isnan(yd))
    # Checks if number of points is greater then the order of the spline.
    if sel.sum() <= 1:
        print 'Array must have at least two data points.', xd, yd
        return yi
    elif sel.sum() <= k:
        k = sel.sum() - 1
    # Removes nan from input data arrays.
    xd, yd = xd[sel], yd[sel]
    # First calculates spline inside the data domain.
    sel = (xi >= xd[0]) & (xi <= xd[-1])
    spl_k = interpolate.UnivariateSpline(xd, yd, k=k, s=s)
    yi[sel] = spl_k(xi[sel])
    # Then extrapolates the data outside data domain, if appropriate.
    # Extrapolation can be either linear, or simply repeating nearest data
    # values.
    if outside in [True, 'linear']:
        spl_1 = interpolate.UnivariateSpline(xd, yd, k=1, s=0)
        yi[~sel] = spl_1(xi[~sel])
    elif outside in ['repeat', 'exponential']:
        if sel.sum() == 0:
            yi[~sel] = _nearest_neighbour(xi[~sel], xd, yd)
        else:
            yi[~sel] = _nearest_neighbour(xi[~sel], xi[sel], yi[sel])
    if outside == 'exponential':
        sel = flatnonzero(xi >= -5.)
        yi[sel], popt, _ = _fit_exponential(xi[sel], xd, yd, right=xi[sel[0]])
        print '{:.4f}\t{:.4f}'.format(*popt),
    #
    return yi


def _nearest_neighbour(xi, xd, yd):
    """Returns data of nearest neighbor of xi in xd."""
    yi = empty(xi.shape)
    for i, x in enumerate(xi):
        d = abs(xd - x)
        j = flatnonzero(d == d.min())
        yi[i] = yd[j].mean()
    #
    return yi
