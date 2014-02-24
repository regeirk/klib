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

__all__ = ['figure', 'plot', 'plot_ts', 'wavelet_plot']

import numpy
import pylab

from time import time
from sys import stdout
from string import atof
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from mpl_toolkits import axisartist
from mpl_toolkits.basemap import cm
from mpl_toolkits.axes_grid1 import host_subplot

import common
import cm as custom_cm


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
        orientation (string, optional) :
            Adjusts figure size according to selectec orientation. Valid
            options are 'landscape' (8 x 5.8), 'portrait' (8 x 11),
            'squared' (8 x 8), 'worldmap' (8, 4.5), 'landscape.golden'
            (8 x 4.9), 'portrait.golden' (8 x 12.9), 'landscape.letter'
            (11 x 8), portrait.letter (8 x 11).
    
    RETURNS
        fig : Figure object
    
    """

    __init__()
    golden = (5 ** 0.5 + 1.0) / 2.0    # The golden ratio
    letter = 11./8.
    
    if 'figsize' not in fp.keys():
        if orientation == 'landscape':
            fp['figsize'] = [8, 8/letter]
        elif orientation == 'portrait':
            fp['figsize'] = [8, 8*letter]
        elif orientation == 'squared':
            fp['figsize'] = [8, 8]
        elif orientation == 'worldmap':
            fp['figsize'] = [8, 4.5] # Widescreen aspect ratio 16:9
        elif orientation == 'landscape.golden':
            fp['figsize'] = [8, 8/golden]
        elif orientation == 'portrait.golden':
            fp['figsize'] = [8, 8*golden]
        elif orientation == 'landscape.letter':
            fp['figsize'] = [11, 8]
        elif orientation == 'portrait.letter':
            fp['figsize'] = [8, 11]
        else:
            raise Warning, 'Orientation \'%s\' not allowed.' % (orientation, )
    
    fig = pylab.figure(**fp)
    fig.subplots_adjust(**ap)
    
    return fig


def legend(labels, ax=None, im=None, bbox=None):
    """Adds legend to plot.

    """
    if ax == None:
        ax = pylab.gca()

    fontP = FontProperties()
    fontP.set_size('small')
    #
    if bbox == None:
        bbox = (0.5, -0.05)
    if im == None:
        ax.legend(labels, loc='upper center', bbox_to_anchor=bbox,
            ncol=int(round(len(labels)/2)), prop=fontP)
    else:
        _proxy, _legend = [], []
        for lc, pc in zip(labels, im.collections):
            if lc != None:
                _proxy.append(pylab.Rectangle((0, 0), 1, 1,
                    fc=pc.get_facecolor()[0], hatch=pc.get_hatch()))
                _legend.append(lc)
        ax.legend(_proxy, _legend, loc='upper center', 
            bbox_to_anchor=bbox, ncol=int(round(len(labels)/2)), 
            prop=fontP)
    #
    return


def axis_degree(ax, axis='x'):
    """
    """
    if axis == 'x':
        ax.set_xticklabels([common.num2latlon(i, 0, mode='each', x180=True,
            dtype='label')[1] for i in ax.get_xticks()])
    elif axis == 'y':
        ax.set_yticklabels([common.num2latlon(0, i, mode='each', x180=True,
            dtype='label')[0] for i in ax.get_yticks()])
    else:
        raise ValueError('Invalid \'%s\' axis.' % (axis))


def timeformat(ax, dt=7, axis='x', orientation='portrait'):
    """Formats time axis.

    """
    try:
        if axis == 'x':
            ax.xaxis_date()
            #if orientation == 'portrait':
            #    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
            #elif orientation == 'landscape':
            #    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        elif axis == 'y':
            ax.yaxis_date()
            #if orientation == 'portrait':
            #    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
            #elif orientation == 'landscape':
            #    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.minorticks_on()
        return
    except:
        pass
    
    if dt <= 90:
        major = pylab.matplotlib.dates.DayLocator(range(1, 30, 10))
        minor = pylab.matplotlib.dates.DayLocator()
        fmt = u'%d/%m'
    elif dt <= 9131: # 25 years!
        major = pylab.matplotlib.dates.YearLocator(1)
        minor = pylab.matplotlib.dates.MonthLocator(range(1, 13), 
            bymonthday=1)
        fmt = u'%Y'
    else:
        major = pylab.matplotlib.dates.YearLocator(10)
        minor = pylab.matplotlib.dates.YearLocator(1)
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

    return True


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
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return
    
def plot_ts(*args, **kwargs):
    """Plots time-series.
    
    RETURNS
        ax : axis
    
    """
    kwargs['xscale'] = 'time'
    return plot(*args, **kwargs)


def plot(x, y, title='', xlabel='', xunits='', ylabel='', yunits='', label='', 
    format='-', color='k', linewidth=1.5, markersize=7, fig=None,
    subplot=(1, 1, 1), sharex=None, sharey=None, xlim=None, ylim=None, 
    xscale='linear', yscale='linear', xaxis='same', yaxis='same', 
    scale=1., scale_label=None, nospines=False, xtick='auto', ytick='auto',
    legend_label=None, orientation='portrait'):
    """Plot lines and/or markers.
    
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
        ylabel = [ylabel]
        yunits = [yunits]
        yscale = [yscale]
        n = 1
    else:
        n = len(y)
        if type(x).__name__ == 'ndarray':
            x = [x] * n
        if type(format) == str:
            format = [format] * n
        if type(color) == str:
            color = [color] * n
        if type(linewidth) in [float, int]:
            linewidth = [linewidth] * n
        if type(markersize) in [float, int]:
            markersize = [markersize] * n
        if type(ylabel) == str:
            ylabel = [ylabel] * n
        if type(yunits) == str:
            yunits = [yunits] * n
        if type(yscale) == str:
            yscale = [yscale] * n
    
    if len(subplot) == 3:
        if xaxis == 'twin':
            ax = host_subplot(subplot[0], subplot[1], subplot[2],
                sharex=sharex, sharey=sharey, axes_class=axisartist.Axes)
        if yaxis == 'twin':
            ax = host_subplot(subplot[0], subplot[1], subplot[2],
                sharex=sharex, sharey=sharey, axes_class=axisartist.Axes)
        else:
            ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                sharex=sharex, sharey=sharey)
    elif len(subplot) == 4:
        ax = fig.add_axes(subplot, sharex=sharex, sharey=sharey)

    if not (xlim == None):
        ax.set_xlim(xlim)
    if not (ylim == None):
        ax.set_ylim(ylim)
    
    if nospines:
        dropspines(ax)

    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    xmin, xmax = 9e9, 0
    for i in range(n):
        if i == 0:
            bx = ax
        else:
            if xaxis == 'twin':
                bx = ax.twinx()

                offset = (i - 1) * 100
                new_fixed_axis = bx.get_grid_helper().new_fixed_axis
                bx.axis["right"] = new_fixed_axis(loc="right", axes=bx,
                    offset=(offset, 0))
                bx.axis["right"].toggle(all=True)
                
            elif yaxis == 'twin':
                raise Warning, 'Y-axis twin not implemented yet.'

        if xscale == 'log2':
            xs = numpy.log2(x[i])
        else:
            xs = x[i]
        if yscale == 'log2':
            ys = numpy.log2(y[i])
        else:
            ys = y[i]

        if ((n == 1) | (xaxis == 'twin')) & (scale == None):
            std = numpy.log10(ys.std())
            if (std > 3) | (std <=-1):
                std = numpy.round(std)
                scale = 10 ** std
                scale_label = r'\times 10^{%d}' % (std)
            else:
                scale = 1.
                scale_label = ''
        else:
            scale = 1
            scale_label = ''

        # Sets scale label if not set
        if (scale != 1) & (scale_label == None):
            scale_label = r'\times %s' %s (scale)
        
        args = dict(color=color[i], linewidth=linewidth[i], 
            markersize=markersize[i])
        if numpy.iscomplex(ys).any():
            q = bx.quiver(xs, xs * 0, ys.real, ys.imag)
            qk = bx.quiverkey(q, 0.1, 0.1, 1., ur'%d %s' % (1, xunits), 
                labelpos='E')
        elif (xscale != 'log') & (yscale[i] == 'log'):
            bx.semilogy(xs, ys/scale, format[i], **args) 
        else:
            bx.plot(xs, ys/scale, format[i], **args)
        xmin, xmax = min(xmin, xs.min()), max(xmax, xs.max())
        #
        if ytick[:4] == 'auto':
            n = ytick.find(':')
            if n >= 0:
                n = atof(ytick[n+1:])
            else:
                n = 5
            if yscale[i] == 'linear':
                bx.yaxis.set_major_locator(ticker.MaxNLocator(n))
        if (xaxis == 'twin') | (i == 0):
            if yunits[i] or scale_label:
                bx.set_ylabel(ur'\textbf{%s} $\left[%s %s\right]$' % (ylabel[i], 
                    scale_label, yunits[i]))
            elif ylabel[i]:
                bx.set_ylabel(ur'\textbf{%s}' % (ylabel[i]))
            if yscale[i] == 'deg':
                bx.set_yticklabels([common.num2latlon(0, tk, mode='each', 
                    x180=False, dtype='label')[0] for tk in bx.get_yticks()])
    
    if xscale == 'log2':
        xmin, xmax = numpy.floor([-3., xmax])
        xticks = 2 ** numpy.arange(xmin, xmax)
        bx.set_xticks(numpy.log2(xticks))
        bx.set_xticklabels(xticks)
        pylab.setp(bx.get_xticklabels()[::2], visible=False)

    if xlim == None:
        ax.set_xlim([xmin, xmax])
    else:
        ax.set_xlim(xlim)
    if not (ylim == None):
        ax.set_ylim(ylim)
    if xtick[:4] == 'auto':
        n = xtick.find(':')
        if n >= 0:
            n = atof(xtick[n+1:])
        else:
            n = 10
        if xscale == 'linear':
            ax.xaxis.set_major_locator(ticker.MaxNLocator(n))
    if title:
        ax.set_title(title)
    if label:
        ax.text(0.02, 0.95, label, ha='left', va='top', transform=bx.transAxes,
            bbox=bbox)
    if xunits != '':
        ax.set_xlabel(ur'\textbf{%s} $\left[%s\right]$' % (xlabel, xunits))
    elif xlabel != None:
        ax.set_xlabel(ur'\textbf{%s}' % xlabel)
    else:
        pylab.setp(bx.get_xticklabels(), visible=False)
    if xscale == 'time':
        timeformat(ax, dt=xmax-xmin, orientation=orientation)
    if xscale == 'deg':
        ax.set_xticklabels([common.num2latlon(i, 0, mode='each', x180=False,
            dtype='label')[1] for i in bx.get_xticks()])
    #
    ax.minorticks_on()
    ax.grid(True, zorder=0)

    if legend_label != None:
        # Draws legend
        legend(legend_label, ax=ax)
    
    pylab.draw()
    return ax


def contour(x, y, z, title='', xlabel='', xunits='', ylabel='', yunits='',
    zunits='', label='', fig=None, subplot=(1, 1, 1), sharex=None, xlim=None, 
    ylim=None, xscale='linear', yscale='linear', zscale='linear', 
    nospines=False, scale=1., scale_label=None, crange=None, cticks=None,
    cmap=cm.GMT_no_green, colorbar=True, cbarpos=None, 
    orientation='horizontal', extend='both'):
    """
    """
    if fig == None:
        fig = figure()

    if len(subplot) == 3:
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], sharex=sharex)
    elif len(subplot) == 4:
        ax = fig.add_axes(subplot, sharex=sharex)

    if nospines:
        dropspines(ax)
    ax.minorticks_on()
    
    norm = None
    # Base 10 logarithmic scale
    if zscale == 'log':
        z = numpy.log10(z)
        crange = numpy.log10(crange)
    # The chlorophyll-a color scale as described at
    # http://oceancolor.gsfc.nasa.gov/DOCS/standard_chlorophyll_colorscale.txt
    # Chl-a concentration are converted from mg m-3 to a log like scale, i.e.
    #   pix = (log10(chlor_a) + 2) / 0.015
    #   chlor_a = 10 ** (0.015 * pix - 2)
    elif zscale == 'chla':
        cmap = custom_cm.custom_chla
        z = (numpy.log10(z) + 2) / 0.015
        zrange = numpy.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 60])
        #crange = numpy.arange(0, 256)
        crange = (numpy.log10(zrange) + 2) / 0.015
        cticks = dict(values=(numpy.log10(zrange) + 2) / 0.015, text=zrange)
    
    # Setting the color ranges
    if (crange == None) & (cticks == None):
        cmajor, cminor, crange, cticks, extend = common.step(z / scale,
            returnrange=True)
        cticks = dict(values = cticks)
    if cticks == None:
        if len(crange) < 15:
            cticks = dict(values = crange[::2])
        else:
            cticks = dict(values = crange[::5])
    if zscale == 'log':
        cticks['text'] = ['10$^%d$' % (tick) for tick in cticks['values']]
    
    # Sets scale label according to scale
    if scale_label == None:
        log = int(numpy.log10(scale))
        scale = 10 ** log
        if log <> 0:
            scale_label = r'\times 10^{%d}' % (log)
            crange /= scale
        else:
            scale_label = ''
    
    xmin, xmax = 9e9, 0
    xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
    ymin, ymax = 9e9, 0
    ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)

    try:
        xmask = z.mask.any(axis=0)
        sel = pylab.find(~xmask)
        xmin, xmax = x[sel[0]], x[sel[-1]]
    except:
        pass
    try:
        ymask = z.mask.any(axis=1)
        sel = pylab.find(~ymask)
        ymin, ymax = y[sel[0]], y[sel[-1]]
    except:
        pass
    
    # The contour!
    im = ax.contourf(x, y, z / scale, crange, extend=extend, cmap=cmap, 
        norm=norm)
    
    # Draws colorbar
    if colorbar:
        corners = ax.get_position().corners()
        if orientation == 'squared':
            co = 'horizontal'
        elif orientation  in ['landscape', 'worldmap', 'horizontal']:
            if cbarpos == None:
                cbarpos = [0.05, -0.08, -0.1, 0.02]
            position = numpy.array([corners[0, 0], corners[0, 1], 
                corners[2, 0] - corners[0, 0], 0]) + numpy.array(cbarpos)
            co = 'horizontal'
        elif orientation in ['portrait', 'vertical']:
            if cbarpos == None:
                cbarpos = [0.03, 0.025, 0.017, -0.05]
            position = numpy.array([corners[2, 0], corners[2, 1], 0, 
                corners[3, 1] - corners[2, 1]]) + numpy.array(cbarpos)
            co = 'vertical'
        else:
            raise Warning('Invalid orientation %s.' % orientation)
        cax = fig.add_axes(position)
        pylab.colorbar(im, cax=cax, orientation=co, ticks=cticks['values'], 
            extend=extend)
        if 'text' in cticks.keys():
            if co == 'horizontal':
                cax.set_xticklabels(cticks['text'])
            else:
                cax.set_yticklabels(cticks['text'])
    
    if title:
        ax.set_title(title)
    if xunits != '':
        ax.set_xlabel(ur'\textbf{%s} $\left[%s\right]$' % (xlabel, xunits))
    elif xlabel != '':
        ax.set_xlabel(ur'\textbf{%s}' % xlabel)
    if yunits:
        ax.set_ylabel(ur'\textbf{%s} $\left[%s\right]$' % (ylabel, yunits))
    elif ylabel:
        ax.set_ylabel(ur'\textbf{%s}' % (ylabel))
    if xscale == 'time':
        timeformat(ax, dt=xmax-xmin)
    if xscale == 'deg':
        ax.set_xticklabels([common.num2latlon(i, 0, mode='each', x180=True,
            dtype='label')[1] for i in ax.get_xticks()])
    if yscale == 'time':
        timeformat(ax, dt=ymax-ymin, axis='y')
    if yscale == 'deg':
        ax.set_yticklabels([common.num2latlon(0, i, mode='each', x180=False,
            dtype='label')[0] for i in ax.get_yticks()])
    if (zunits != '') | (scale_label != ''):
        if co == 'horizontal':
            ci, cj, ha, va = 1.05, 0.5, 'left', 'center'
        else:
            ci, cj, ha, va = 0.5, -0.15, 'left', 'top'
        cax.text(ci, cj, r'$\left[%s %s\right]$' % (scale_label, zunits), ha=ha, 
            va=va, transform=cax.transAxes)
    if label:
        ax.text(0.02, 0.95, label, ha='left', va='top', transform=ax.transAxes,
            bbox=bbox)
    
    if xlim == None:
        xlim = [xmin, xmax]
    if ylim == None:
        ylim = [ymin, ymax]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.minorticks_on()
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
    grey = (0.6, 0.6, 0.6)

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
    x0 = 0.13  # Left margin
    if len(levels) > 5:
        w0 = 1 - (2 * x0) - 0.2 - 0.01
    else:
        w0 = 1 - (1.25 * x0) - 0.2 - 0.01
    y1, y2, y3, y4 = (0.75, 0.355, 0.34, 0.11)
    h1, h2, h3, h4 = (0.2, 0.35, 0.32, 0.2)
    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    fig = figure(fp=dict(), orientation='landscape')

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
    coix = numpy.concatenate([[tm[0]], tm, [tm[-1]], [tm[-1]],
        [tm[0]], [tm[0]]])
    coiy = numpy.concatenate([[0.75], coi, [0.75], [period[-1]], 
        [period[-1]], [0.75]])

    # First plot, the original time-series, its trends and some formatting.
    ax = fig.add_axes([x0, y1, w0, h1])
    if type(ztrend).__name__ == 'ndarray':
        ax.plot(tm, ztrend, '-', color=grey, linewidth=1.5)
        ax.plot(tm, ztrend + 2 * std, '--', color=grey, linewidth=1)
        ax.plot(tm, ztrend - 2 * std, '--', color=grey, linewidth=1)
    ax.plot(tm, z * std, 'k-', linewidth=1.5)
    pylab.setp(ax.get_xticklabels(), visible=False)
    #ystep, ystep1 = common.step(z * std, 3)
    #ymajor = pylab.matplotlib.ticker.MultipleLocator(ystep)
    #yminor = pylab.matplotlib.ticker.MultipleLocator(ystep1)
    #ax.yaxis.set_major_locator(ymajor)
    #ax.yaxis.set_minor_locator(yminor)
    ax.text(0.02, 0.93, 'i)', ha='left', va='top', transform=ax.transAxes,
            bbox=bbox)
    if labels['units'] != '':
        ax.set_ylabel(r'\textbf{%s} $\left[%s\right]$' % (labels['name'],
            labels['units']))
    elif labels['name']:
        ax.set_ylabel(r'\textbf{%s}' % (labels['name'],))
    timeformat(ax, dt=tm[-1] - tm[0])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.minorticks_on()
    ax.grid(True, zorder=0)
    
    # Second subplot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area.
    bx = fig.add_axes([x0, y2, w0, h2], sharex=ax)
    pylab.contourf(tm, numpy.log2(period), numpy.log2(power),
                   numpy.log2(levels), cmap=cmap, extend=extend)
    if type(power_signif).__name__ == 'ndarray':
        bx.contour(tm, numpy.log2(period), power_signif, [-99, 1], colors='k',
            linewidths=1.)
    bx.fill_between(tm, numpy.log2(period[-1]), numpy.log2(coi), color='k', 
        alpha='0.3', hatch='x')
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
    bx.yaxis.set_major_locator(ticker.MaxNLocator(5))
    bx.minorticks_on()
    bx.grid(True, zorder=0)

    if len(levels) > 5:
        cax = fig.add_axes([x0+w0+0.02, y2+0.005, 0.015, h2-0.01])
        pylab.colorbar(cax=cax, ax=bx, orientation='vertical', extend=extend,
                       ticks=numpy.log2(levels[0::2]))
        cax.set_yticklabels(levels[0::2])

    # Third subplot, the global wavelet and Fourier power spectra and
    # theoretical noise spectra.
    cx = fig.add_axes([x0+w0+0.15, y2, 0.42 - (2 * x0), h2], sharey=bx)
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
    #xstep, xstep1 = common.step(glbl, 2)
    #xmajor = pylab.matplotlib.ticker.MultipleLocator(xstep)
    #xminor = pylab.matplotlib.ticker.MultipleLocator(xstep1)
    #cx.xaxis.set_major_locator(xmajor)
    #cx.xaxis.set_minor_locator(xminor)
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
    cx.xaxis.set_major_locator(ticker.MaxNLocator(3))
    cx.yaxis.set_major_locator(ticker.MaxNLocator(5))
    cx.minorticks_on()
    cx.grid(True, zorder=0)

    # Fourth subplot, the scale averaged wavelet spectrum.
    dx = fig.add_axes([x0, y4, w0, h4], sharex=ax)
    if type(wtrend).__name__ == 'ndarray':
        dx.plot(tm, wtrend, '-', color=grey, linewidth=1.5)
    dx.plot(tm, scale_avg, 'k-', linewidth=1.5)
    if scale_signif:
        dx.axhline(scale_signif, color='k', linestyle=':', linewidth=1.)
    if scale_signif > scale_avg.max():
        ystep, ystep1 = common.step(numpy.array([0, scale_signif]), 3)
    else:
        ystep, ystep1 = common.step(scale_avg, 3)
    #ymajor = pylab.matplotlib.ticker.MultipleLocator(ystep)
    #yminor = pylab.matplotlib.ticker.MultipleLocator(ystep1)
    #dx.yaxis.set_major_locator(ymajor)
    #dx.yaxis.set_minor_locator(yminor)
    dx.text(0.02, 0.93, 'iv)', ha='left', va='top', transform=dx.transAxes,
        bbox=bbox)
    dx.set_xlabel(r'\textbf{%s}' % (labels['Year']))
    if normalized:
        dx.set_ylabel(r'\textbf{%s}' % (labels['Power']))
    else:
        dx.set_ylabel(r'\textbf{%s} $\left[%s^2\right]$' % (labels['Power'],
            labels['units']))
    #tickFmt = pylab.matplotlib.dates.DateFormatter("%Y")
    #xmajor = pylab.matplotlib.dates.YearLocator(1)
    #xminor = pylab.matplotlib.dates.MonthLocator(range(1, 13), bymonthday=15)
    #dx.xaxis.set_major_locator(xmajor)
    #dx.xaxis.set_minor_locator(xminor)
    #dx.xaxis.set_major_formatter(tickFmt)
    #dx.format_xdata = pylab.matplotlib.dates.DateFormatter('%Y-%m-%d')
    dx.set_xlim([tm.min() - dt, tm.max() + dt])
    #pylab.setp(dx.get_xticklabels()[1::2], visible=False)
    dx.xaxis.set_major_locator(ticker.MaxNLocator(7))
    dx.yaxis.set_major_locator(ticker.MaxNLocator(4))
    dx.minorticks_on()
    dx.grid(True, zorder=0)

    # Drawing and saving the figure if appropriate.
    pylab.draw()
    if save:
        pylab.savefig('%s.%s' % (save, ftype), dpi=150)
    if show == False:
        pylab.close(fig)


def add_second_axis(ax, axis='x', tick='top', xtick=None, ytick=None,
    xlim=None, ylim=None, hide_tick_labels=''):
    """Adds a second axis with different units.
    
    """
    if xtick == None:
        _xtick = [0.5, 1, 2.5, 5, 25]
        xtick = dict(
            value=1./numpy.asarray(_xtick),
            label=[r'%s$^{\circ}$' % (item) for item in _xtick]
        )
    if ytick == None:
        ytick = xtick
    if xlim == None:
        xlim = ax.get_xlim()
    if ylim == None:
        ylim = ax.get_ylim()
    
    corners = ax.get_position().corners()
    position = [corners[0, 0], corners[0, 1], corners[2, 0]-corners[0, 0],
        corners[3, 1]-corners[0, 1]]
    aax = ax.figure.add_axes(position, xscale=ax.get_xscale(),
        yscale=ax.get_yscale(), frameon=False)

    tick = tick.split()
    if axis.find('x') >= 0:
        if 'top' in tick:
            aax.xaxis.tick_top()
            ax.xaxis.tick_bottom()
        if 'bottom' in tick:
            aax.xaxis.tick_bottom()
            ax.xaxis.tick_top()
        #
        aax.set_xlim(xlim)
        aax.set_xticks(xtick['value'])
        aax.set_xticklabels(xtick['label'], fontsize='x-small')
    else:
        aax.xaxis.set_major_locator(pylab.NullLocator())
    if axis.find('y') >= 0:
        if 'left' in tick:
            aax.yaxis.tick_left()
            ax.yaxis.tick_right()
        if 'right' in tick:
            aax.yaxis.tick_right()
            ax.yaxis.tick_left()
        #
        aax.set_ylim(ylim)
        aax.set_yticks(ytick['value'])
        aax.set_yticklabels(ytick['label'], fontsize='x-small')
    else:
        aax.yaxis.set_major_locator(pylab.NullLocator())
    if hide_tick_labels.find('x') >= 0:
        pylab.setp(aax.get_xticklabels(), visible=False)
    if hide_tick_labels.find('y') >= 0:
        pylab.setp(aax.get_yticklabels(), visible=False)

    return aax
