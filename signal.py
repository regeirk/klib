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

__all__ = ['fftnegativeshift', 'next_power_of_two', 'power_spectral_density']

from numpy import (arange, arctan2, array, asarray, ceil, concatenate,
    fliplr, flipud, log2, take, take, zeros, isnan)
from scipy.fftpack import fft2, fftfreq, fftshift
from scipy.signal import fftconvolve, detrend


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
