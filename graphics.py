# -*- coding: iso-8859-1 -*-
"""Graphics module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to offer a framework to generate high quality
plots easily.

AUTHOR
    Sebastian Krieger
    email: solutions@nublia.com

REVISION
    1 (2011-12-19 17:21)

"""
from __future__ import division

__version__ = '$Revision: 4 $'
# $Source$

__all__ = ['figure', 'plot_ts', 'wavelet_plot']

import numpy
import pylab

from time import time
from sys import stdout
from mpl_toolkits.basemap import cm

import common


def __init__(show=False):
    if show:
        pylab.ion()


def figure(fp=dict(), ap=dict(left=0.15, bottom=0.12, right=0.95, top=0.95, 
    wspace=0.10, hspace=0.10), orientation='portrait'):
    """Creates a standard figure.
    
    PARAMETERS
        fp (dictionary, optional) :
            Figure properties.
        ap (dictionary, optional) :
            Adjustment properties.
    
    RETURNS
        fig : Figure object
    
    """

    __init__()
    golden = (5 ** 0.5 + 1.0) / 2.0    # The golden ratio
    
    if 'figsize' not in fp.keys():
        if orientation == 'landscape':
            fp['figsize'] = [11, 8]
        elif orientation == 'portrait':
            fp['figsize'] = [8, 11]
        elif orientation == 'squared':
            fp['figsize'] = [8, 8]
        elif orientation == 'worldmap':
            fp['figsize'] = [9, 5.0625] # Widescreen aspect ratio 16:9
        else:
            raise Warning, 'Orientation \'%s\' not allowed.' % (orientation, )
    
    fig = pylab.figure(**fp)
    fig.subplots_adjust(**ap)
    
    return fig


def timeformat(ax, dt=7, axis='x'):
    """Formats time axis.

    """
    if dt <= 90:
        major = pylab.matplotlib.dates.DayLocator(range(1, 30, 10))
        minor = pylab.matplotlib.dates.DayLocator()
        fmt = u'%d/%m'
    else:
        major = pylab.matplotlib.dates.YearLocator(1)
        minor = pylab.matplotlib.dates.MonthLocator(range(1, 13), 
            bymonthday=1)
        fmt = u'%Y'
    if axis == 'x':
        Ax = ax.xaxis
    elif axis == 'y':
        Ax = ax.yaxis
    Ax.set_major_locator(major)
    Ax.set_minor_locator(minor)
    Ax.set_major_formatter(pylab.matplotlib.dates.DateFormatter(fmt))
    
    if axis == 'x':
        ax.format_xdata = pylab.matplotlib.dates.DateFormatter((u'%Y-%m-%d'
            ' %H:%M'))
        pylab.setp(ax.get_xticklabels()[1::2], visible=False)
    elif axis == 'y':
        ax.format_ydata = pylab.matplotlib.dates.DateFormatter((u'%Y-%m-%d'
            ' %H:%M'))
        pylab.setp(ax.get_yticklabels()[1::2], visible=False)


def dropspines(ax, dist=7):
    """Drops some spines from plot axis ax.
    
    """
    for loc, spine in ax.spines.iteritems():
        if loc in ['left', 'bottom']:
            spine.set_position(('outward', dist)) # outward by 10 points
        elif loc in ['right', 'top']:
            spine.set_color('none') # don't draw spine
        else:
            raise ValueError('unknown spine location: %s' % loc)
    # Turning off ticks where there's no spine 
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return


def plot_ts(x, y, title='', xlabel='Time', xunits='', ylabel='', yunits='',
    label='', format='-', color='k', linewidth=1.5, markersize=7, fig=None,
    subplot=(1, 1, 1), sharex=None, xlim=None, ylim=None, xscale='time',
    yscale='linear', nospines=True):
    """Plots time-series.
    
    PARAMETERS
        x (array like) :
        y (array like) :

    RETURNS
        ax : axis
    """
    if fig == None:
        fig = figure()
    if type(y).__name__ in ['ndarray', 'MaskedArray']:
        x, y = [x], [y]
        format = [format]
        color = [color]
        linewidth = [linewidth]
        markersize = [markersize]
        n = 1
    else:
        n = len(y)
        if type(x).__name__ == 'ndarray':
            x = [x] * n
        if type(format).__name__ == 'str':
            format = [format] * n
        if type(color).__name__ == 'str':
            color = [color] * n
        if type(linewidth).__name__ in ['float', 'int']:
            linewidth = [linewidth] * n
        if type(markersize).__name__ in ['float', 'int']:
            markersize = [markersize] * n

    if len(subplot) == 3:
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], sharex=sharex)
    elif len(subplot) == 4:
        ax = fig.add_axes(subplot, sharex=sharex)

    if nospines:
        dropspines(ax)

    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    xmin, xmax = 9e9, 0
    for i in range(n):
        if xscale == 'log2':
            xs = numpy.log2(x[i])
        else:
            xs = x[i]
        if yscale == 'log2':
            ys = numpy.log2(y[i])
        else:
            ys = y[i]
        
        if numpy.iscomplex(ys).any():
            q = ax.quiver(xs, xs * 0, ys.real, ys.imag)
            qk = ax.quiverkey(q, 0.1, 0.1, 1., ur'%d %s' % (1, xunits), 
                labelpos='E')
        else:
            ax.plot(xs, ys, format[i], color=color[i], linewidth=linewidth[i], 
                markersize=markersize[i])
        xmin, xmax = min(xmin, xs.min()), max(xmax, xs.max())

    if xscale == 'log2':
        xmin, xmax = numpy.floor([-3., xmax])
        xticks = 2 ** numpy.arange(xmin, xmax)
        ax.set_xticks(numpy.log2(xticks))
        ax.set_xticklabels(xticks)
        pylab.setp(ax.get_xticklabels()[::2], visible=False)

    if xlim == None:
        ax.set_xlim([xmin, xmax])
    else:
        ax.set_xlim(xlim)
    if not (ylim == None):
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    if label:
        ax.text(0.02, 0.95, label, ha='left', va='top', transform=ax.transAxes,
            bbox=bbox)
    if xunits != '':
        ax.set_xlabel(ur'\textbf{%s} $\left[%s\right]$' % (xlabel, xunits))
    elif xlabel != '':
        ax.set_xlabel(ur'\textbf{%s}' % xlabel)
    else:
        pylab.setp(ax.get_xticklabels(), visible=False)
    if yunits:
        ax.set_ylabel(ur'\textbf{%s} $\left[%s\right]$' % (ylabel, yunits))
    elif ylabel:
        ax.set_ylabel(ur'\textbf{%s}' % (ylabel))
    if xscale == 'time':
        timeformat(ax, dt=xmax-xmin)

    pylab.draw()
    return ax


def wavelet_plot(tm, period, z, power, coi, glbl, scale_avg, fft=None,
                 fft_period=None, power_signif=None, glbl_signif=None,
                 scale_signif=None, pminmax=[], labels=dict(), normalized=True,
                 std=1., ztrend=None, wtrend=None, show=False, save='', 
                 ftype='png', levels=None, cmap=cm.GMT_no_green):
    """Plots results from wavelet analysis.

    PARAMETERS
        tm (array like) :
            Time.
        period (array like) :
            Gives the Fourier periods of the wavelet analysis.
        z (array like) :
            Variable (first plot).
        power (array like) :
            Wavelet power spectrum (second plot).
        coi (array like) :
            Cone of influence as returned by the wavelet analysis module.
        glbl (array like) :
            Global wavelet power spectrum (third plot).
        scale_avg (array like) :
            Scale averaged power spectrum (fourth plot).
        fft (array like, optional) :
            Fast Fourier Transform (FFT) power spectrum. If given, the
            fft_periods parameter has also to be set.
        fft_period (aray like, optional) :
            If fft parameter is set, the FFT periods have to be given.
        power_signif (array like, optional) :
            Normalized wavelet power spectrum significance level. If
            set then draws a contour line where the significance level
            equals one.
        glbl_signif (array like, optional) :
            Significance of the global wavelet spectrum.
        scale_signif: (float, optional) :
            Scale average significance level.
        pminmax (array like, optional) :
            Dictionary containing scale averaging upper and lower limits.
        labels (dictionary, optional) :
            Sets the labels for the plot axis. Should be a dictionary
            with the following keys:
                - name (name or symbol of the variable)
                - units (units of the variable)
                - power (power axis label)
                - period (period axis label)
        normalizes (boolean, optional) :
            Tells wether the time-series is normalized by its standard
            deviation or not.
        std (float, optional) :
            Standard deviation of normalized time-series. Usefull for
            plotting the original time-series.
        ztrend (array like, optional) :
            Polynomial coefficients from least square fit for trend
            plot of the variable z.
        wtrend (array like, optional) :
            Polynomial coefficients from least square fit for trend
            plot of the scale averaged wavelet power.
        show (boolean, optional) :
            If set to true the the resulting maps are explicitly shown
            on screen.
        save (string, optional) :
            The path in which the resulting plots are to be saved. If
            not set, then no images will be saved.
        ftype (string, optional) :
            The image file type. Most backends support png, pdf, ps,
            eps and svg.
        levels (array like, optional) :
            Array of power levels to be used in colorbar.
        cmap (colormap, optional) :
            Sets the colormap to be used in the plots. The default is
            the Generic Mapping Tools (GMT) no green.

    OUTPUT
        Wavelet analysis series plot on screen and/or on file.

    RETURNS
        Nothing.

    """
    t1 = time()
    __init__()
    
    # Some constants
    grey = [0.6, 0.6, 0.6]

    # Turning interactive mode on or off according to show parameter.
    if show == False:
        pylab.ioff()
    elif show == True:
        pylab.ion()
    else:
        raise Warning, 'Invalid show option.'        

    # Setting undefined label strings.
    if 'name' not in labels.keys():
        labels['name'] = ''
    if 'units' not in labels.keys():
        labels['units'] = ''
    if 'Power' not in labels.keys():
        labels['Power'] = 'Power'
    if 'Period' not in labels.keys():
        labels['Period'] = 'Period'
    if 'months' not in labels.keys():
        labels['months'] = r'\textnormal{months}'
    if 'Year' not in labels.keys():
        labels['Year'] = 'Year'

    # Calculating trended values from ztrend and wtrend parameters using time
    # array in years.
    t = tm / 365.2421896698
    try:
        ztrend = numpy.polyval(ztrend, t)
    except:
        ztrend = None
    try:
        wtrend = numpy.polyval(wtrend, t)
    except:
        wtrend = None

    # Setting up the figure.
    x0 = 0.11  # Left margin
    if len(levels) > 5:
        w0 = 1 - (2 * x0) - 0.2 - 0.01
    else:
        w0 = 1 - (1.25 * x0) - 0.2 - 0.01
    figprops = dict(figsize=(11, 8))
    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    
    fig = figure(fp=figprops)

    # Temporal sampling interval, colorbar levels, its extend, the period
    # ticks, the cone of influence fill coordinates.
    dt = tm[1] - tm[0]
    if type(levels).__name__ == 'NoneType':
        #levels = numpy.array([2, 5, 10])
        levels = 2. ** numpy.arange(-3, 6)
    extend = 'both'
    periodY = period / 365.25
    Yticks = 2 ** numpy.arange(numpy.ceil(numpy.log2(periodY.min())),
                               numpy.ceil(numpy.log2(periodY.max())))
    yticks = Yticks * 365.25
    coix = numpy.concatenate([tm[:1] - dt, tm, tm[-1:] + dt, tm[-1:] + dt,
                             tm[:1] - dt, tm[:1] - dt])
    coiy = numpy.concatenate([[1e-9], coi, [1e-9], period[-1:], period[-1:],
                             [1e-9]])

    # First plot, the original time-series, its trends and some formatting.
    ax = fig.add_axes([x0, 0.7, w0, 0.25])
    if type(ztrend).__name__ == 'ndarray':
        ax.plot(tm, ztrend, '-', color=grey, linewidth=1.5)
        ax.plot(tm, ztrend + 2 * std, '--', color=grey, linewidth=1)
        ax.plot(tm, ztrend - 2 * std, '--', color=grey, linewidth=1)
    ax.plot(tm, z * std, 'k-', linewidth=1.5)
    pylab.setp(ax.get_xticklabels(), visible=False)
    ystep, ystep1 = common.step(z * std, 3)
    ymajor = pylab.matplotlib.ticker.MultipleLocator(ystep)
    yminor = pylab.matplotlib.ticker.MultipleLocator(ystep1)
    ax.yaxis.set_major_locator(ymajor)
    ax.yaxis.set_minor_locator(yminor)
    ax.text(0.02, 0.93, 'i)', ha='left', va='top', transform=ax.transAxes,
            bbox=bbox)
    if labels['units'] != '':
        ax.set_ylabel(r'\textbf{%s} $\left[%s\right]$' % (labels['name'],
            labels['units']))
    elif labels['name']:
        ax.set_ylabel(r'\textbf{%s}' % (labels['name'],))
    
    # Second subplot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area.
    bx = fig.add_axes([x0, 0.34, w0, 0.32], sharex=ax)
    pylab.contourf(tm, numpy.log2(period), numpy.log2(power),
                   numpy.log2(levels), cmap=cmap, extend=extend)
    if type(power_signif).__name__ == 'ndarray':
        bx.contour(tm, numpy.log2(period), power_signif, [-99, 1], colors='k',
            linewidths=1.)
    bx.fill(coix, numpy.log2(coiy), 'k', alpha='0.3', hatch='x')
    try:
        bx.axhline(numpy.log2(min(pminmax)), linewidth=2, color='w', alpha=0.8)
        bx.axhline(numpy.log2(max(pminmax)), linewidth=2, color='w', alpha=0.8)
    except:
        pass
    pylab.setp(bx.get_xticklabels(), visible=False)
    bx.text(0.02, 0.95, 'ii)', ha='left', va='top', transform=bx.transAxes,
            bbox=bbox)
    bx.set_ylabel(r'\textbf{%s} $\left[%s \right]$' % (labels['Period'],
        labels['months']))
    bx.invert_yaxis()

    if len(levels) > 5:
        cax = fig.add_axes([0.7, 0.345, 0.015, 0.31])
        pylab.colorbar(cax=cax, ax=bx, orientation='vertical', extend=extend,
                       ticks=numpy.log2(levels[0::2]))
        cax.set_yticklabels(levels[0::2])

    # Third subplot, the global wavelet and Fourier power spectra and
    # theoretical noise spectra.
    cx = fig.add_axes([0.79, 0.34, 0.4 - (2 * x0), 0.32], sharey=bx)
    if type(fft).__name__ == 'ndarray':
        cx.plot(fft, numpy.log2(fft_period), '-', color=grey,
                linewidth=1.)
    cx.plot(glbl, numpy.log2(period), 'k-', linewidth=1.5)
    if type(glbl_signif).__name__ == 'ndarray':
        cx.plot(glbl_signif, numpy.log2(period), 'k:')
    try:
        cx.set_xlim([0, glbl.max() + glbl_signif.mean()])
    except:
        pass
    xstep, xstep1 = common.step(glbl, 2)
    xmajor = pylab.matplotlib.ticker.MultipleLocator(xstep)
    xminor = pylab.matplotlib.ticker.MultipleLocator(xstep1)
    cx.xaxis.set_major_locator(xmajor)
    cx.xaxis.set_minor_locator(xminor)
    cx.text(0.05, 0.95, 'iii)', ha='left', va='top', transform=cx.transAxes,
            bbox=bbox)
    if normalized:
        cx.set_xlabel(r'\textbf{%s}' % (labels['Power']))
    else:
        cx.set_xlabel(r'\textbf{%s} $\left[%s^2\right]$' % (labels['Power'],
            labels['units'], ))
    cx.set_ylim(numpy.log2([period.min(), period.max()]))
    cx.set_yticks(numpy.log2(yticks))
    cx.set_yticklabels(Yticks * 12)
    pylab.setp(cx.get_yticklabels(), visible=False)
    cx.invert_yaxis()

    # Fourth subplot, the scale averaged wavelet spectrum.
    dx = fig.add_axes([x0, 0.1, w0, 0.20], sharex=ax)
    if type(wtrend).__name__ == 'ndarray':
        dx.plot(tm, wtrend, '-', color=grey, linewidth=1.5)
    dx.plot(tm, scale_avg, 'k-', linewidth=1.5)
    if scale_signif:
        dx.axhline(scale_signif, color='k', linestyle=':', linewidth=1.)
    if scale_signif > scale_avg.max():
        ystep, ystep1 = common.step(numpy.array([0, scale_signif]), 3)
    else:
        ystep, ystep1 = common.step(scale_avg, 3)
    ymajor = pylab.matplotlib.ticker.MultipleLocator(ystep)
    yminor = pylab.matplotlib.ticker.MultipleLocator(ystep1)
    dx.yaxis.set_major_locator(ymajor)
    dx.yaxis.set_minor_locator(yminor)
    dx.text(0.02, 0.93, 'iv)', ha='left', va='top', transform=dx.transAxes,
        bbox=bbox)
    dx.set_xlabel(r'\textbf{%s}' % (labels['Year']))
    if normalized:
        dx.set_ylabel(r'\textbf{%s}' % (labels['Power']))
    else:
        dx.set_ylabel(r'\textbf{%s} $\left[%s^2\right]$' % (labels['Power'],
            labels['units']))
    tickFmt = pylab.matplotlib.dates.DateFormatter("%Y")
    xmajor = pylab.matplotlib.dates.YearLocator(1)
    xminor = pylab.matplotlib.dates.MonthLocator(range(1, 13), bymonthday=15)
    dx.xaxis.set_major_locator(xmajor)
    dx.xaxis.set_minor_locator(xminor)
    dx.xaxis.set_major_formatter(tickFmt)
    dx.format_xdata = pylab.matplotlib.dates.DateFormatter('%Y-%m-%d')

    dx.set_xlim([tm.min(), tm.max()])
    pylab.setp(dx.get_xticklabels()[1::2], visible=False)

    # Drawing and saving the figure if appropriate.
    pylab.draw()
    if save:
        pylab.savefig('%s.%s' % (save, ftype), dpi=150)
    if show == False:
        pylab.close(fig)
