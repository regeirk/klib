# -*- coding: iso-8859-1 -*-
"""Statistics module.

This is part of the kLib Python library for scientific data analysis.
The purpose of this module is to assist in performing basic and
advanced statistical analysis of geophysical properties such as sea
surface height, for example.

AUTHOR
    Sebastian Krieger
    email: sebastian@nublia.com

REVISION
    4 (2012-02-24 20:17 -0300)
    3 (2011-09-08 13:54 -0300)
    2 (2011-05-01 22:21 -0300)
    1 (2011-04-20 14:48 -0300)

"""

__version__ = '$Revision: 4 $'
# $Source$

__all__ = ['acorr', 'basics', 'wavelet_analysis', 'polyfit2d', 'polyval2d', 
    'local_maxima', 'detect_peaks']

import os
import numpy
import pylab
import string
import warnings
import itertools
import scipy.signal

from time import time
from sys import stdout
from scipy import ndimage
from mpl_toolkits.basemap import cm

import common
import wavelet
import gis
import graphics
import file as fm


def acorr(a):
    """Discrete linear normalized auto-correlation of a 1-dimensional
    sequence.

    This function uses numpy.convolve

    PARAMETERS
        a (array like) :
            The input sequence to be analysed.

    RETURNS
        out (array like):
            Discrete linear auto-correlation of the input array.

    """
    n = a.size
    out = scipy.signal.fftconvolve(a, a[::-1], 'full')
    out /= out[n - 1]
    return out


def basics(z, dt=None, oldschool=False):
    """Performs basic statistics on given data variable z.

    Calculates the mean, standard deviation and trend along time.
    Assumes fist dimension of the array to be time and the others to be
    the coordinates. Maximum number of dimensions is three. The trend
    is calculated by least square fit of a one degree polynomial
    function.

    PARAMETERS
        z (array like) :
            Variable to be analysed.
        dt (float) :
            Temporal sampling scale to normalize the trend.
        oldschool (boolean, optional):
            If set to true, calculates the avarages and standard deviation
            using old school techniques.

    RETURNS
        mean, std, trend, alpha (array like) :
            Calculated mean, standard deviation, trends and lag-1 auto-
            correlation.
    """
    t1 = time()

    # Transforms input arrays numpy masked arrays.
    z = numpy.ma.asarray(z)
    mask = (z.mask | numpy.isnan(z.data)).any(axis=0)
    if dt == None:
        dt = 1.

    dim = len(z.shape)
    if dim == 1:
        z = numpy.reshape(z.size, 1, 1)
    elif dim == 2:
        c, b = z.shape
        z = numpy.reshape(c, b, 1)
    elif dim > 3:
        raise Warning, 'Higher dimensions than three are not implemented.'
    c, b, a = z.shape
    t = numpy.arange(c) * dt

    t2 = time()
    s = 'Calculating mean... '
    stdout.write(s)
    stdout.flush()
    if oldschool:
        zmean = numpy.ma.empty([b, a]) * numpy.nan
        zstd = numpy.ma.empty([b, a]) * numpy.nan
        for i in range(a):
            t2 = time()
            for j in range(b):
                if not mask[j, i]:
                    zmean[j, i] = z[:, j, i].mean()
                    zstd[j, i] = z[:, j, i].std()
            stdout.write(len(s) * '\b')
            s = ('Calculating mean and standard deviation... %s ' %
                (common.profiler(a, i + 1, 0, t1, t2)))
            stdout.write(s)
            stdout.flush()
        s = '\n'
    else:
        zmean = z.mean(axis=0)
        s = '%s\n' % (common.profiler(1, 1, 0, t1, t2))
    zmean[mask] = numpy.nan
    zmean.mask = mask
    stdout.write(s)
    
    if not oldschool:
        t2 = time()
        s = 'Calculating standard deviation... '
        stdout.write(s)
        stdout.flush()
        zstd = z.std(axis=0)
        s = '%s\n' % (common.profiler(1, 1, 0, t1, t2))
        stdout.write(s)
    zstd[mask] = numpy.nan
    zstd.mask = mask

    s = 'Calculating trends and lag-1 autocorrelation... '
    stdout.write(s)
    stdout.flush()
    ztrend = numpy.ma.empty([b, a]) * numpy.nan
    zalpha = numpy.ma.empty([b, a]) * numpy.nan
    for i in range(a):
        t2 = time()
        for j in range(b):
            if not mask[j, i]:
                p = numpy.polyfit(t, z[:, j, i], 1)
                ztrend[j, i] = p[0]
                #
                ac = acorr(z[:, j, i])
                zalpha[j, i] = (ac[c] + ac[c + 1] ** 0.5) / 2

        stdout.write(len(s) * '\b')
        s = ('Calculating trends and lag-1 autocorrelation... %s ' %
                (common.profiler(a, i + 1, 0, t1, t2)))
        stdout.write(s)
        stdout.flush()
    ztrend.mask = mask
    zalpha.mask = mask | numpy.isnan(zalpha)
    stdout.write('\n')

    return zmean, zstd, ztrend, zalpha


def local_maxima(z, cyclic=False, epsilon=0.):
    """Calculates the local minima and maxima from the input field.
    
    PARAMETERS
        z (array like) :
            Input signal.
        cyclic (boolean, optional) :
            If true, assumes cycles at the borders.
        epsilon (float, optional) :
            Acceptable error, default value is 0.
    
    RETURNS
        mmap (array like):
            Boolean Map with the locations of the local minima and
            maxima.
    
    """
    if numpy.ndim(z) != 2:
        raise Warning, 'Only two-dimensional mapping implemented.'

    l, k = z.shape
    mmap = numpy.empty((l, k), dtype='bool') * False

    for i in range(k):
        for j in range(l):
            # Sets north, south, east and west indices.
            n, s = j - 1, j + 2
            w, e = i - 1, i + 2
            if not cyclic:
                if (n < 0):
                    n = 0
                elif s > l:
                    s = l
                if w < 0:
                    w = 0
                elif e > k:
                    e = k
            # Checks if neighbors have higher absolute value.
            zij = abs(z[j, i]) - epsilon
            ismax = True
            for u in range(w, e):
                for v in range(n, s):
                    if (u != i) & (v != j) & (zij <= abs(z[v, u])):
                        # print '%d (%d:%d), %d (%d:%d), %s' % (i, w, e, j, n, s, 'Yeah!')
                        ismax = False
            #
            mmap[j, i] = ismax

    # Returns the local minima and maxima map
    return mmap


def wavelet_analysis(z, tm, lon=None, lat=None, mother='Morlet', alpha=0.0,
                     siglvl=0.95, loc=None, onlyloc=False, periods=None,
                     sel_periods=[], show=False, save='', dsave='', prefix='',
                     labels=dict(), title=None, name=None, fpath='', 
                     fpattern='', std=dict(), crange=None, levels=None,
                     cmap=cm.GMT_no_green, debug=False):
    """Continuous wavelet transform and significance analysis.

    The analysis is made using the methodology and statistical approach
    suggested by Torrence and Compo (1998).

    Depending on the dimensions of the input array, three different
    kinds of approaches are taken. If the input array is one-dimensional
    then only a simple analysis is performed. If the array is
    bi- or three-dimensional then spectral Hovmoller diagrams are drawn
    for each Fourier period given within a range of +/-25%.

    PARAMETERS
        z (array like) :
            Input data. The data array should have one of these forms,
            z[tm], z[tm, lat] or z[tm, lat, lon].
        tm (array like) :
            Time axis. It should contain values in matplotlib date
            format (i.e. number of days since 0001-01-01 UTC).
        lon (array like, optional) :
            Longitude.
        lat (array like, optional) :
            Latitude.
        mother (string, optional) :
            Gives the name of the mother wavelet to be used. Possible
            values are 'Morlet' (default), 'Paul' or 'Mexican hat'.
        alpha (float or dictionary, optional) :
            Lag-1 autocorrelation for background noise.  Default value
            is 0.0 (white noise). If different autocorrelation 
            coefficients should be used for different locations, then
            the input should contain a dictionary with 'lon', 'lat',
            'map' keys as for the std parameter.
        siglvl (float, optional) :
            Significance level. Default value is 0.95.
        loc (array like, optional) :
            Special locations of interest. If the input array is of
            higher dimenstions, the output of the simple wavelet
            analysis of each of the locations is output. The list
            should contain the pairs of (lon, lat) for each locations
            of interest.
        onlyloc (boolean, optional) :
            If set to true then only the specified locations are
            analysed. The default is false.
        periods (array like, optional) :
            Special Fourier periods of interest in case of analysis of
            higher dimensions (in years).
        sel_periods (array like, optional) :
            Select which Fourier periods spectral power are averaged.
        show (boolean, optional) :
            If set to true the the resulting maps are shown on screen.
        save (string, optional) :
            The path in which the resulting plots are to be saved. If
            not set, then no images will be saved.
        dsave (string, optional) :
            If set, saves the scale averaged power spectrum series to
            this path. This is especially useful if memory is an issue.
        prefix (string, optional) :
            Prefix to retain naming conventions such as basin.
        labels (dictionary, optional) :
            Sets the labels for the plot axis.
        title (string, array like, optional) :
            Title of each of the selected periods.
        name (string, array like, optional) :
            Name of each of the selected periods. Used when saving the 
            results to files.
        fpath (string, optional) :
            Path for the source files to be loaded when memory issues
            are a concern.
        fpattern (string, optional) :
            Regular expression pattern to match file names.
        std (dictionary, optional) :
            A dictionary containing a map of the standard deviation of
            the analysed time series. To set the longitude and latitude
            coordinates of the map, they should be included as
            separate 'lon' and 'lat' key items. If they are omitted,
            then the regular input parameters are assumed. Accepted
            standard deviation error is set in key 'err' (default value
            is 1e-2).
        crange (array like, optional) :
            Array of power levels to be used in average Hovmoler colour bar.
        levels (array like, optional) :
            Array of power levels to be used in spectrogram colour bar.
        cmap (colormap, optional) :
            Sets the colour map to be used in the plots. The default is
            the Generic Mapping Tools (GMT) no green.
        debug (boolean, optional) :
            If set to True then warnings are shown.

    OUTPUT
        If show or save are set, plots either on screen and or on file
        according to the specified parameters.

        If dsave parameter is set, also saves the scale averaged power
        series to files.

    RETURNS
        wave (dictionary) :
            Dictionary containing the resulting calculations from the
            wavelet analysis according to the input parameters. The
            output items might be:
                scale --
                    Wavelet scales.
                period --
                    Equivalent Fourier periods (in days).
                power_spectrum --
                    Wavelet power spectrum (in units**2).
                power_significance --
                    Relative significance of the power spectrum.
                global_power --
                    Global wavelet power spectrum (in units**2).
                scale_spectrum  --
                    Scale averaged wavelet spectra (in units**2)
                    according to selected periods.
                scale_significance --
                    Relative significance of the scale averaged wavelet
                    spectra.
                fft --
                    Fourier spectrum.
                fft_first --
                    Fourier spectrum of the first half of the 
                    time-series.
                fft_second --
                    Fourier spectrum of the second half of the 
                    time-series.
                fft_period --
                    Fourier periods (in days).
                trend --
                    Signal trend (in units/yr).
                wavelet_trend --
                    Wavelet spectrum trends (in units**2/yr).

    """
    t1 = time()
    result = {}

    # Resseting unit labels for hovmoller plots
    hlabels = dict(labels)
    hlabels['units'] = ''
    
    # Setting some titles and paths
    if name == None:
        name = title

    # Working with the std parameter and setting its properties:
    if 'val' in std.keys():
        if 'lon' not in std.keys():
            std['lon'] = lon
        std['lon180'] = common.lon180(std['lon'])
        if 'lat' not in std.keys():
            std['lat'] = lat
        if 'err' not in std.keys():
            std['err'] = 1e-2
        std['map'] = True
    else:
        std['map'] = False
    
    # Lag-1 autocorrelation parameter
    if type(alpha).__name__ == 'dict':
        if 'lon' not in alpha.keys():
            alpha['lon'] = lon
        alpha['lon180'] = common.lon180(alpha['lon'])
        if 'lat' not in alpha.keys():
            alpha['lat'] = lat
        alpha['mean'] = alpha['val'].mean()
        alpha['map'] = True
        alpha['calc'] = False
    else:
        if alpha == -1:
            alpha = {'mean': -1, 'calc': True}
        else:
            alpha = {'val': alpha, 'mean': alpha, 'map': False, 'calc': False}

    # Shows some of the options on screen.
    print ('Average Lag-1 autocorrelation for background noise: %.2f' % 
        (alpha['mean']))
    if save:
        print 'Saving result figures in \'%s\'.' % (save)
    if dsave:
        print 'Saving result data in \'%s\'.' % (dsave)

    if fpath:
        # Gets the list of files to be loaded individually extracts all the
        # latitudes and loads the first file to get the main parameters.
        flist = os.listdir(fpath)
        flist, match = common.reglist(flist, fpattern)
        if len(flist) == 0:
            raise Warning, 'No files matched search pattern.'
        flist = numpy.asarray(flist)
        lst_lat = []
        for item in match:
            y = string.atof(item[-2])
            if item[-1].upper() == 'S': y *= -1
            lst_lat.append(y)
        # Detect file type from file name
        ftype = fm.detect_ftype(flist[0])
        x, y, tm, z = fm.load_map('%s/%s' % (fpath, flist[0]),
            ftype=ftype, masked=True)
        if lon == None:
            lon = x
        lat = numpy.unique(lst_lat)
        dim = 2
    else:
        # Transforms input arrays in numpy arrays and numpy masked arrays.
        tm = numpy.asarray(tm)
        z = numpy.ma.asarray(z)
        z.mask = numpy.isnan(z)

        # Determines the number of dimensions of the variable to be plotted and
        # the sizes of each dimension.
        a = b = c = None
        dim = len(z.shape)
        if dim == 3:
            c, b, a = z.shape
        elif dim == 2:
            c, a = z.shape
            b = 1
            z = z.reshape(c, b, a)
        else:
            c = z.shape[0]
            a = b = 1
            z = z.reshape(c, b, a)
        if tm.size != c:
            raise Warning, 'Time and data lengths do not match.'
    
    # Transforms coordinate arrays into numpy arrays
    s = type(lat).__name__
    if s in ['int', 'float', 'float64']:
        lat = numpy.asarray([lat])
    elif s != 'NoneType':
        lat = numpy.asarray(lat)
    s = type(lon).__name__
    if s in ['int', 'float', 'float64']:
        lon = numpy.asarray([lon])
    elif s != 'NoneType':
        lon = numpy.asarray(lon)

    # Starts the mother wavelet class instance and determines important
    # analysis parameters
    mother = mother.lower()
    if mother == 'morlet':
        mother = wavelet.Morlet()
    elif mother == 'paul':
        mother = wavelet.Paul()
    elif mother in ['mexican hat', 'mexicanhat', 'mexican_hat']:
        mother = wavelet.Mexican_hat()
    else:
        raise Warning, 'Mother wavelet unknown.'

    t = tm / common.daysinyear        # Time array in years
    dt = tm[1] - tm[0]                # Temporal sampling interval
    try:                              # Zonal sampling interval
        dx = lon[1] - lon[0]
    except:
        dx = 1
    try:                              # Meridional sampling interval
        dy = lat[1] - lat[0]
    except:
        dy = dx
    if numpy.isnan(dt): dt = 1
    if numpy.isnan(dx): dx = 1
    if numpy.isnan(dy): dy = dx
    dj = 0.25                         # Four sub-octaves per octave
    s0 = 2 * dt                       # Smallest scale
    J = 7 / dj - 1                    # Seven powers of two with dj sub-octaves
    scales = period = None

    if type(crange).__name__ == 'NoneType':
        crange = numpy.arange(0, 1.1, 0.1)
    if type(levels).__name__ == 'NoneType':
        levels = 2. ** numpy.arange(-3, 6)

    if fpath:
        N = lat.size
        # TODO: refactoring # lon = numpy.arange(-81. - dx / 2., 290. + dx / 2, dx)
        # TODO: refactoring # lat = numpy.unique(numpy.asarray(lst_lat))
        c, b, a = tm.size, lat.size, lon.size
    else:
        N = a * b
    
    # Making sure that the longitudes range from -180 to 180 degrees and
    # setting the squared search radius R2.
    try:
        lon180 = common.lon180(lon)
    except:
        lon180 = None
    R2 = dx ** 2 + dy ** 2
    if numpy.isnan(R2):
        R2 = 65535.
    if loc != None:
        loc = numpy.asarray([[common.lon180(item[0]), item[1]] for item in 
            loc])

    # Initializes important result variables such as the global wavelet power
    # spectrum map, scale avaraged spectrum time-series and their significance,
    # wavelet power trend map.
    global_power = numpy.ma.empty([J + 1, b, a]) * numpy.nan
    try:
        C = len(periods) + 1
        dT = numpy.diff(periods)
        pmin = numpy.concatenate([[periods[0] - dT[0] / 2],
                                 0.5 * (periods[:-1] + periods[1:])])
        pmax = numpy.concatenate([0.5 * (periods[:-1] + periods[1:]),
                                 [periods[-1] + dT[-1] / 2]])
    except:
        # Sets the lowest period to null and the highest to half the time
        # series length.
        C = 1
        pmin = numpy.array([0])
        pmax = numpy.array([(tm[-1] - tm[0]) / 2])
    if type(sel_periods).__name__ in ['int', 'float']:
        sel_periods = [sel_periods]
    elif len(sel_periods) == 0:
        sel_periods = [-1.]
    try:
        if fpath:
            raise Warning, 'Process files individually'
        avg_spectrum = numpy.ma.empty([C, c, b, a]) * numpy.nan
        mem_error = False
    except:
        avg_spectrum = numpy.ma.empty([C, c, a]) * numpy.nan
        mem_error = True
    avg_spectrum_signif = numpy.ma.empty([C, b, a]) * numpy.nan
    trend = numpy.ma.empty([b, a]) * numpy.nan
    wavelet_trend = numpy.ma.empty([C, b, a]) * numpy.nan
    fft_trend = numpy.ma.empty([C, b, a]) * numpy.nan
    std_map = numpy.ma.empty([b, a]) * numpy.nan
    zero = numpy.ma.empty([c, a])
    fft_spectrum = None
    fft_spectrum1 = None
    fft_spectrum2 = None

    # Walks through each latitude and then through each longitude to perform
    # the temporal wavelet analysis.
    if N == 1:
        plural = ''
    else:
        plural = 's'
    s = 'Spectral analysis of %d location%s... ' % (N, plural)
    stdout.write(s)
    stdout.flush()
    for j in range(b):
        t2 = time()
        isloc = False  # Ressets 'is special location' flag
        hloc = []      # Cleans location list for Hovmoller plots
        zero *= numpy.nan
        if mem_error:
            # Clears average spectrum for next step.
            avg_spectrum *= numpy.nan
            avg_spectrum.mask = False
        if fpath:
            findex = pylab.find(lst_lat == lat[j])
            if len(findex) == 0:
                continue
            ftype = fm.detect_ftype(flist[findex[0]])
            try:
                x, y, tm, z = fm.load_dataset(fpath, flist=flist[findex],
                    ftype=ftype, masked=True, lon=lon, lat=lat[j:j+1],
                    verbose=True)
            except:
                continue
            z = z[:, 0, :]
            x180 = common.lon180(x)

        # Determines the first and second halves of the time-series and some
        # constants for the FFT
        fft_ta = numpy.ceil(t.min())
        fft_tb = numpy.floor(t.max())
        fft_tc = numpy.round(fft_ta + fft_tb) / 2
        fft_ia = pylab.find((t >= fft_ta) & (t <= fft_tc))
        fft_ib = pylab.find((t >= fft_tc) & (t <= fft_tb))
        fft_N = int(2 ** numpy.ceil(numpy.log2(max([len(fft_ia), 
            len(fft_ib)]))))
        fft_N2 = fft_N / 2 - 1
        fft_dt = t[fft_ib].mean() - t[fft_ia].mean()
        
        for i in range(a):
            # Some string output.
            try:
                Y, X = common.num2latlon(lon[i], lat[j], mode='each', 
                    padding=False)
            except:
                Y = X = '?'
            
            # Extracts individual time-series from the whole dataset and
            # sets or calculates its standard deviation, squared standard
            # deviation and finally the normalized time-series.
            if fpath:
                try:
                    ilon = pylab.find(x == lon[i])[0]
                    fz = z[:, ilon]
                except:
                    continue
            else:
                fz = z[:, j, i]
            if fz.mask.all():
                continue
            if std['map']:
                try:
                    u = pylab.find(std['lon180'] == lon180[i])[0]
                    v = pylab.find(std['lat'] == lat[j])[0]
                except:
                    if debug:
                        warnings.warn('Unable to locate standard deviation '
                                      'for (%s, %s)' % (X, Y), Warning)
                    continue
                fstd = std['val'][v, u]
                estd = fstd - fz.std()
                if (estd < 0) & (abs(estd) > std['err']):
                    if debug:
                        warnings.warn('Discrepant input standard deviation '
                            '(%f) location (%.3f, %.3f) will be '
                            'disregarded.' % (estd, lon180[i], lat[j]))
                    continue
            else:
                fstd = fz.std()
            fstd2 = fstd ** 2
            std_map[j, i] = fstd
            zero[:, i] = fz
            fz = (fz - fz.mean()) / fstd
            
            # Calculates the distance of the current point to any special
            # location set in the 'loc' parameter. If only special locations
            # are to be analysed, then skips all other ones. If the input
            # array is one dimensional, then do the analysis anyway.
            if dim == 1:
                dist = numpy.asarray([0.])
            else:
                try:
                    dist = numpy.asarray([((item[0] - (lon180[i])) **
                        2 + (item[1] - lat[j]) ** 2) for item in loc])
                except:
                    dist = []
            if (dist > R2).all() & (loc != 'all') & onlyloc:
                continue
            
            # Determines the lag-1 autocorrelation coefficient to be used in
            # the significance test from the input parameter
            if alpha['calc']:
                ac = acorr(fz)
                alpha_ij = (ac[c + 1] + ac[c + 2] ** 0.5) / 2
            elif alpha['map']:
                try:
                    u = pylab.find(alpha['lon180'] == lon180[i])[0]
                    v = pylab.find(alpha['lat'] == lat[j])[0]
                    alpha_ij = alpha['val'][v, u]
                except:
                    if debug:
                        warnings.warn('Unable to locate standard deviation '
                            'for (%s, %s) using mean value instead' %
                            (X, Y), Warning)
                    alpha_ij = alpha['mean']
            else:
                alpha_ij = alpha['mean']

            # Calculates the continuous wavelet transform using the wavelet
            # Python module. Calculates the wavelet and Fourier power spectrum
            # and the periods in days. Also calculates the Fourier power 
            # spectrum for the first and second halves of the timeseries.
            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(fz, dt, dj,
                s0, J, mother)
            power = abs(wave * wave.conj())
            fft_power = abs(fft * fft.conj())
            period = 1. / freqs
            fftperiod = 1. / fftfreqs
            psel = pylab.find(period <= pmax.max())
            
            # Calculates the Fourier transform for the first and the second
            # halves ot the time-series for later trend analysis.
            fft_1 = numpy.fft.fft(fz[fft_ia], fft_N)[1:fft_N/2] / fft_N ** 0.5
            fft_2 = numpy.fft.fft(fz[fft_ib], fft_N)[1:fft_N/2] / fft_N ** 0.5
            fft_p1 = abs(fft_1 * fft_1.conj())
            fft_p2 = abs(fft_2 * fft_2.conj())
            
            # Creates FFT return array and stores the spectrum accordingly
            try:
                fft_spectrum[:, j, i] = fft_power * fstd2
                fft_spectrum1[:, j, i] = fft_p1 * fstd2
                fft_spectrum2[:, j, i] = fft_p2 * fstd2
            except:
                fft_spectrum = (numpy.ma.empty([len(fft_power), b, a]) *
                    numpy.nan)
                fft_spectrum1 = (numpy.ma.empty([fft_N2, b, a]) *
                    numpy.nan)
                fft_spectrum2 = (numpy.ma.empty([fft_N2, b, a]) *
                    numpy.nan)
                #
                fft_spectrum[:, j, i] = fft_power * fstd2
                fft_spectrum1[:, j, i] = fft_p1 * fstd2
                fft_spectrum2[:, j, i] = fft_p2 * fstd2

            # Performs the significance test according to the article by
            # Torrence and Compo (1998). The wavelet power is significant
            # if the ratio power/sig95 is > 1.
            signif, fft_theor = wavelet.significance(1., dt, scales, 0,
                alpha_ij, significance_level=siglvl, wavelet=mother)
            sig95 = (signif * numpy.ones((c, 1))).transpose()
            sig95 = power / sig95

            # Calculates the global wavelet power spectrum and its
            # significance. The global wavelet spectrum is the average of the
            # wavelet power spectrum over time. The degrees of freedom (dof)
            # have to be corrected for padding at the edges.
            glbl_power = power.mean(axis=1)
            dof = c - scales
            glbl_signif, tmp = wavelet.significance(1., dt, scales, 1,
                alpha_ij, significance_level=siglvl, dof=dof, wavelet=mother)
            global_power[:, j, i] = glbl_power * fstd2

            # Calculates the average wavelet spectrum along the scales and its
            # significance according to Torrence and Compo (1998) eq. 24. The
            # scale_avg_full variable is used multiple times according to the
            # selected periods range.
            #
            # Also calculates the average Fourier power spectrum.
            Cdelta = mother.cdelta
            scale_avg_full = (scales * numpy.ones((c, 1))).transpose()
            scale_avg_full = power / scale_avg_full
            for k in range(C):
                if k == 0:
                    sel = pylab.find((period >= pmin[0]) &
                        (period <= pmax[-1]))
                    pminmax = [period[sel[0]], period[sel[-1]]]
                    les = pylab.find((fftperiod >= pmin[0]) &
                        (fftperiod <= pmax[-1]))
                    fminmax = [fftperiod[les[0]], fftperiod[les[-1]]]
                else:
                    sel = pylab.find((period >= pmin[k - 1]) &
                        (period < pmax[k - 1]))
                    pminmax = [pmin[k-1], pmax[k-1]]
                    les = pylab.find((fftperiod >= pmin[k - 1]) &
                        (fftperiod <= pmax[k - 1]))
                    fminmax = [fftperiod[les[0]], fftperiod[les[-1]]]
                
                scale_avg = numpy.ma.array((dj * dt / Cdelta *
                    scale_avg_full[sel, :].sum(axis=0)))
                scale_avg_signif, tmp = wavelet.significance(1., dt, scales,
                    2, alpha_ij, significance_level=siglvl, 
                    dof=[scales[sel[0]], scales[sel[-1]]], wavelet=mother)
                scale_avg.mask = (scale_avg < scale_avg_signif)
                if mem_error:
                    avg_spectrum[k, :, i] = scale_avg
                else:
                    avg_spectrum[k, :, j, i] = scale_avg
                avg_spectrum_signif[k, j, i] = scale_avg_signif

                # Trend analysis using least square polynomial fit of one
                # degree of the original input data and scale averaged
                # wavelet power. The wavelet power trend is calculated only
                # where the cone of influence spans the highest analyzed
                # period. In the end, the returned value for the trend is in
                # units**2.
                #
                # Also calculates the trends in the Fourier power spectrum.
                # Note that the FFT power spectrum is already multiplied by
                # the signal's standard deviation.
                incoi = pylab.find(coi >= pmax[-1])
                if len(incoi) == 0:
                    incoi = numpy.arange(c)
                polyw = numpy.polyfit(t[incoi], scale_avg[incoi].data, 1)
                wavelet_trend[k, j, i] = polyw[0] * fstd2
                fft_trend[k, j, i] = (fft_spectrum2[les, j, i] -
                    fft_spectrum1[les, j, i]).mean() / fft_dt
                if k == 0:
                    polyz = numpy.polyfit(t, fz * fstd, 1)
                    trend[j, i] = polyz[0]

                # Plots the wavelet analysis results for the individual
                # series. The plot is only generated if the dimension of the
                # input variable z is one, if a special location is within a
                # range of the search radius R and if the show or save
                # parameters are set.
                if (show | (save != '')) & ((k in sel_periods)):
                    if (dist < R2).any() | (loc == 'all') | (dim == 1):
                        # There is an interesting spot within the search
                        # radius of location (%s, %s).' % (Y, X)
                        isloc = True
                        if (dist < R2).any():
                            try:
                                hloc.append(loc[(dist < R2)][0, 0])
                            except:
                                pass                            
                        if save:
                            try:
                                sv = '%s/tz_%s_%s_%d' % (save, prefix, 
                                    common.num2latlon(lon[i], lat[j]), k)
                            except:
                                sv = '%s' % (save)
                        else:
                            sv = ''
                        graphics.wavelet_plot(tm, period[psel], fz,
                            power[psel, :], coi, glbl_power[psel],
                            scale_avg.data, fft=fft, fft_period=fftperiod,
                            power_signif=sig95[psel, :],
                            glbl_signif=glbl_signif[psel],
                            scale_signif=scale_avg_signif, pminmax=pminmax,
                            labels=labels, normalized=True, std=fstd,
                            ztrend=polyz, wtrend=polyw, show=show, save=sv,
                            levels=levels, cmap=cmap)

        # Saves and/or plots the intermediate results as zonal temporal
        # diagrams.
        if dsave:
            for k in range(C):
                if k == 0:
                    sv = '%s/%s/%s_%s.xt.gz' % (dsave, 'global', prefix,
                        common.num2latlon(lon[i], lat[j], mode='each')[0])
                else:
                    sv = '%s/%s/%s_%s.xt.gz' % (dsave, name[k - 1].lower(),
                        prefix,
                        common.num2latlon(lon[i], lat[j], mode='each')[0])
                if mem_error:
                    fm.save_map(lon, tm, avg_spectrum[k, :, :].data,
                        sv, lat[j])
                else:
                    fm.save_map(lon, tm, avg_spectrum[k, :, j, :].data,
                        sv, lat[j])
        
        if ((dim > 1) and (show or (save != '')) & (not onlyloc) and 
                len(hloc) > 0):
            hloc = common.lon360(numpy.unique(hloc))
            if save:
                sv = '%s/xt_%s_%s' % (save, prefix,
                    common.num2latlon(lon[i], lat[j], mode='each')[0])
            else:
                sv = ''
            if mem_error:
                # To include overlapping original signal, use zz=zero
                gis.hovmoller(lon, tm, avg_spectrum[1:, :, :],
                    zo=avg_spectrum_signif[1:, j, :], title=title,
                    crange=crange, show=show, save=sv, labels=hlabels,
                    loc=hloc, cmap=cmap, bottom='avg', right='avg',
                    std=std_map[j, :])
            else:
                gis.hovmoller(lon, tm, avg_spectrum[1:, :, j, :],
                    zo=avg_spectrum_signif[1:, j, :], title=title,
                    crange=crange, show=show, save=sv, labels=hlabels,
                    loc=hloc, cmap=cmap, bottom='avg', right='avg',
                    std=std_map[j, :])

        # Flushing profiling text.
        stdout.write(len(s) * '\b')
        s = 'Spectral analysis of %d location%s (%s)... %s ' % (N, plural, Y,
            common.profiler(b, j + 1, 0, t1, t2))
        stdout.write(s)
        stdout.flush()

    stdout.write('\n')

    result['scale'] = scales
    result['period'] = period
    if dim == 1:
        result['power_spectrum'] = power * fstd2
        result['power_significance'] = sig95
    result['global_power'] = global_power
    result['scale_spectrum'] = avg_spectrum
    if fpath:
        result['lon'] = lon
        result['lat'] = lat
    result['scale_significance'] = avg_spectrum_signif
    result['trend'] = trend
    result['wavelet_trend'] = wavelet_trend
    result['fft_power'] = fft_spectrum
    result['fft_first'] = fft_spectrum1
    result['fft_second'] = fft_spectrum2
    result['fft_period'] = fftperiod
    result['fft_trend'] = fft_trend
    return result


def polyfit2d(x, y, z, order=3, mode='full', debug=False):
    """Two-dimensional polynomial fit. Based uppon code provided by 
    Joe Kington.
    
    PARAMETERS
        mode (string, optional) :
            'full' (default), 'linear', 'diagonal'
    
    References:
        http://stackoverflow.com/questions/7997152/
            python-3d-polynomial-surface-fit-order-dependent/7997925#7997925

    """
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        if (mode=='linear') & (i != 0.) & (j != 0.):
            G[:, k] = 0
        elif (mode=='diagonal') & (i + j > order):
            G[:, k] = 0
        else:
            G[:,k] = x**i * y**j
        #
        if debug:
            if ((mode=='linear') & (i + j > order)) |\
                ((mode=='diagonal') & (i + j > order)):
                print 'G[%d] = 0' % (k)
            else:
                print 'G[%d] = x**%d * y**%d' % (k, i, j)
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m, debug=False):
    """Values to two-dimensional polynomial fit. Based uppon code 
        provided by Joe Kington.
    """
    order = int(numpy.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = numpy.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
        if debug:
            print 'z += %.2f * x**%d * y**%d' % (a, i, j)
    return z

def detect_peaks(image, threshold=0.):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    References:
        http://stackoverflow.com/questions/3684484/
            peak-detection-in-a-2d-array/3689710#3689710

    """

    # define an 8-connected neighborhood
    neighborhood = ndimage.morphology.generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value 
    # in their neighborhood are set to 1
    local_max = ndimage.filters.maximum_filter(image, 
        footprint=neighborhood)==image
    local_min = ndimage.filters.minimum_filter(image, 
        footprint=neighborhood)==image
    # local_max and local_min are masks that contains the peaks we are 
    # looking for, but also the background. In order to isolate the peaks we
    # must remove the background from the mask.

    # we create the mask of the background
    background = (abs(image) <= threshold)

    # a little technicality: we must erode the background in order to 
    # successfully subtract it form local_max and local_min, otherwise a line 
    # will appear along the background border (artifact of the local maximum 
    # and minimum filters)
    eroded_background = ndimage.morphology.binary_erosion(background, 
        structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_max mask
    detected_peaks = local_max + local_min - eroded_background

    return detected_peaks
