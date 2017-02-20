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
import gsw

from time import time
from sys import stdout
from string import atof
from matplotlib import ticker, pyplot
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse
from mpl_toolkits import axisartist
from mpl_toolkits.basemap import cm
from mpl_toolkits.axes_grid1 import host_subplot

from atlantis.astronomy import Compass

import common
import cm as custom_cm


def __init__(show=False):
    if show:
        pylab.ion()


def axes_label(s, ax, x=0.02, y=0.95,
          bbox=dict(edgecolor='w', facecolor='w', alpha=0.9)):
    ax.text(x, y, s, ha='left', va='top', transform=ax.transAxes,
            bbox=bbox, zorder=99)


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
            'squared' (8 x 8), 'squared.half' (4 x 4),
            'worldmap' (8, 4.5), 'landscape.golden' (8 x 4.9),
            'portrait.golden' (8 x 12.9), 'landscape.letter' (11 x 8),
            portrait.letter (8 x 11).

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
        elif orientation == 'squared.half':
            fp['figsize'] = [4, 4]
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


def legend(labels, ax=None, im=None, handles=None, bbox=None,
    loc='upper center', ncol=None):
    """Adds legend to plot.

    """
    if ax == None:
        ax = pylab.gca()

    fontP = FontProperties()
    fontP.set_size('small')
    #
    if bbox == None:
        bbox = (0.5, -0.05)
    if ncol == None:
        ncol = int(round(len(labels)/2))
    if im == None:
        if handles == None:
            ax.legend(labels, loc=loc, bbox_to_anchor=bbox,
                ncol=ncol, prop=fontP)
        else:
            ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox,
                ncol=ncol, prop=fontP)
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


def diagram_ts(T, S, p=0, lon=None, lat=None, use_teos10=True, is_state=True, dsigma=1., result='default', debug=False, **kwargs):
    """Plots T-S diagram.

    Under TEOS-10, the observed values of practical salinity and in situ
    temperature t need to be converted into absolute salinity and
    conservative temperature.

    Parameters
    ----------
    T : array like
        In situ or conservative temperature [degC].
    S : array like
        Practical or absolute salinity [unitless, g kg-1], according.
    p : array like
        Pressure [dbar]. If not given, assumes sea surface.
    lon, lat: float, array like
        To plot state diagram, longitude and latitude have to be given
        in decimal degrees.
    use_teos10 : boolean, optional
        If true (default), uses conservative temperature and absolute
        salinity according to the Thermodynamic Equation of SeaWater
        2010 (TEOS-10). If longitude and latitude are not given,
        assumes that T and S have already been converted according to
        TEOS10.
    is_state : boolean, optional
        If true (default), plots the state diagram: density
        anomalies referenced to surface.
    dsigma : float, optional
        Sets the interval for each isopycnal in the state diagram.
        Default is 1.
    result : string, optional
        If `default` returns axis and handles objects. If `results` also
        returns converted absolute salinity and conservative temperature.
    debug : boolean, optional
        If true prints some statistics on screen.

    Returns
    -------
    ax : axis
    hs : handles
    [CA, CT] : array_like

    """
    keys = kwargs.keys()
    if 'format' not in keys:
        kwargs['format'] = '.'
    if 'zorder' not in keys:
        kwargs['zorder'] = 99
    kwargs['return_handles'] = True

    # Calculates absolute salinity and conservative temperature.
    if use_teos10 and (lon is not None) and (lat is not None):
        SA = gsw.SA_from_SP(S, p, lon, lat)
        CT = gsw.CT_from_t(SA, T, p)
    else:
        SA = S
        CT = T

    if debug == True:
        dump = ['Mean differences', '----------------']
        dT, dS = T - CT, S - SA
        dump.append('Temperature: {:.4f} � {:.4f}'.format(dT.mean(), dT.std()))
        dump.append('Salinity: {:.4f} � {:.4f}'.format(dS.mean(), dS.std()))
        dump.append('')
        print '\n'.join(dump)

    # Plots Theta - SA diagram.
    ax, hs = plot(SA, CT, **kwargs)

    # Calculates in-situ density from absolute salinity (SA) and conservative
    # temperature using `gsw` module.
    if is_state:
        #if (lon == None) | (lat == None):
        #       raise ValueError('Missing longitude and latitude.')
        SA_lim = ax.get_xlim()
        CT_lim = ax.get_ylim()
        SA_range = numpy.linspace(SA_lim[0], SA_lim[1], 100)
        CT_range = numpy.linspace(CT_lim[0], CT_lim[1], 100)
        sigma_range = numpy.arange(0, 50.5, dsigma)
        SA_grid, CT_grid = numpy.meshgrid(SA_range, CT_range)
        #
        sigma_grid = gsw.rho(SA_grid, CT_grid, 0) - 1000
        #
        cs = ax.contour(SA_grid, CT_grid, sigma_grid, sigma_range,
            colors='k', alpha=0.5, zorder=-98)
        cs.clabel(colors='k', alpha=0.5, fmt='%1.1f')

    if result == 'default':
        return ax, hs
    elif result == 'results':
        return ax, hs, SA, CT
    else:
        raise ValueError('Invalid return type `{}`'.format(result))


def plot_ts(*args, **kwargs):
    """Plots time-series.

    RETURNS
        ax : axis

    """
    kwargs['xscale'] = 'time'
    return plot(*args, **kwargs)


def plot(x, y, title='', xlabel='', xunits='', ylabel='', yunits='', label='',
    format='-', color='k', linewidth=1.5, markersize=7, fig=None, ax=None,
    subplot=(1, 1, 1), sharex=None, sharey=None, xlim=None, ylim=None,
    xscale='linear', yscale='linear', xaxis='same', yaxis='same',
    scale=1., scale_label='', nospines=False, xtick='auto', ytick='auto',
    legend_label=None, orientation='portrait', style=None, alpha=1.,
    label_pos=[0.02, 0.95], new_line=False, return_handles=False, err=None,
    **kwargs):
    """Plot lines and/or markers.

    PARAMETERS
        x (array like) :
        y (array like) :
        style (string, optional) :
            Barb, quiver, scatter, ...
        return_handles (boolean, optional) :
            If true returns ax and plot handles.

    RETURNS
        ax[, handles] : axis

    """
    if fig == None:
        fig = figure()
    if type(y).__name__ in ['ndarray', 'MaskedArray']:
        x, y = [x], [y]
        format = [format]
        color = [color]
        alpha = [alpha]
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
        if type(format) in [str, unicode]:
            format = [format] * n
        if type(color) in [str, unicode]:
            color = [color] * n
        if type(alpha) in [float, int]:
            alpha = [alpha] * n
        if type(linewidth) in [float, int]:
            linewidth = [linewidth] * n
        if type(markersize) in [float, int]:
            markersize = [markersize] * n
        if type(ylabel) in [str, unicode]:
            ylabel = [ylabel] * n
        if type(yunits) in [str, unicode]:
            yunits = [yunits] * n
        if type(yscale) in [str, unicode]:
            yscale = [yscale] * n

    if ax == None:
        if len(subplot) == 3:
            if xaxis == 'twin':
                ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                    sharex=sharex, sharey=sharey)
            if yaxis == 'twin':
                ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                    sharex=sharex, sharey=sharey)
            else:
                ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                    sharex=sharex, sharey=sharey)
        elif len(subplot) == 4:
            ax = fig.add_axes(subplot, sharex=sharex, sharey=sharey)
    else:
        ax.hold('on')

    if nospines:
        dropspines(ax)

    # Adds a line between labels and units. Makes sure that character is
    # UTF8 encoded.
    if new_line:
        new_line = u'\n'
    else:
        new_line = u''

    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    xmin, xmax = 9e9, 0
    handles = []
    Ax = []
    for i in range(n):
        if i == 0:
            bx = ax
        else:
            if xaxis == 'twin':
                bx = ax.twinx()
                offset = 1 + (i - 1) * 1.1
                bx.spines['right'].set_position(('axes', offset))

            elif yaxis == 'twin':
                bx = ax.twiny()
                offset = 1 + (i - 1) * 2.1
                bx.spines['bottom'].set_position(('axes', 0))
                print 'Ahhhh!!!!'
        #
        Ax.append(bx)
        #

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

        # Sets label for legend
        try:
            _label = legend_label[i]
        except:
            _label = None

        # Sets scale label if not set
        if (scale != 1) & (scale_label == ''):
            scale_label = r'\times %s' % (scale)

        args = kwargs.copy()
        args.update(dict(color=color[i], markerfacecolor=color[i],
            linewidth=linewidth[i], markersize=markersize[i], alpha=alpha[i]),
            label=_label)
        quiver = False
        if numpy.iscomplex(ys).any():
            quiver = True
            if style == 'barbs':
                handle = bx.barbs(xs, xs * 0, ys.real, ys.imag, **kwargs)
            else:
                # Normalize vectors!
                ysN = numpy.sqrt(ys.real**2 + ys.imag**2)
                ys = ys / ysN
                #
                q = bx.quiver(xs, xs * 0, ys.real, ys.imag, units='y',
                    scale_units='y', scale=scale, **kwargs)
                handle = q
                if yunits[i] == '':
                    qk = bx.quiverkey(q, 0.1, 0.1, 1., labelpos='E')
                else:
                    qk = bx.quiverkey(q, 0.1, 0.1, 1., ur'%d $%s$' % (1,
                        yunits[i]), labelpos='E')
        elif (xscale != 'log') & (yscale[i] == 'log'):
            handle, = bx.semilogy(xs, ys/scale, format[i], **args)
        elif (xscale == 'log') & (yscale[i] != 'log'):
            handle, = bx.semilogx(xs, ys/scale, format[i], **args)
        elif err != None:
            _draw_ellipse = False
            if 'ellipse' in err.keys():
                if err['ellipse']:
                    _draw_ellipse = True
            if _draw_ellipse:
                for _x, _y, _w, _h in zip(xs, ys/scale, err['x'], err['y']):
                    _e = Ellipse(xy=(_x, _y), width=_w, height=_h, alpha=0.5, color='#333333')
                    handle = bx.add_patch(_e)
            else:
                handle, _, _= bx.errorbar(xs, ys/scale, xerr=err['x'],
                    yerr=err['y'], fmt=format[i], **args)
        elif style == 'scatter':
            handle = bx.scatter(xs, ys/scale, marker=format[i],
                s=args['markersize'], c=args['color'], cmap=args['cmap'],
                alpha=args['alpha'], zorder=args['zorder'], vmin=args['vmin'],
                vmax=args['vmax'])
        else:
            handle = bx.plot(xs, ys/scale, format[i], **args)
        handles.append(handle)
        try:
            xmin, xmax = min(xmin, xs.min()), max(xmax, xs.max())
        except:
            xmin, xmax = 0, 1
        #
        if not (xlim == None):
            try:
                if len(xlim[i]) == 2:
                    bx.set_xlim(xlim[i])
                else:
                    raise ValueError()
            except:
                bx.set_xlim(xlim)
        if not (ylim == None):
            try:
                if len(ylim[i]) == 2:
                    bx.set_ylim(ylim[i])
                else:
                    raise ValueError()
            except:
                bx.set_ylim(ylim)
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
            if not quiver:
                if yunits[i] or scale_label:
                    bx.set_ylabel(ur'\textbf{%s}' % (ylabel[i]) + new_line +
                        ur'$\left[%s %s\right]$' % (scale_label, yunits[i]))
                elif ylabel[i]:
                    bx.set_ylabel(ur'\textbf{%s}' % (ylabel[i]))
            else:
                pylab.setp(bx.get_yticklabels(), visible=False)
                if ylabel[i]:
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
    if yscale == 'log2':
        ymin, ymax = numpy.floor([-3., ymax])
        yticks = 2 ** numpy.arange(ymin, ymax)
        bx.set_yticks(numpy.log2(yticks))
        bx.set_yticklabels(yticks)
        pylab.setp(bx.get_yticklabels()[::2], visible=False)

    if xlim == None:
        ax.set_xlim([xmin, xmax])
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
        axes_label(label, ax, label_pos[0], label_pos[1], bbox)
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

    if (1 == 2) & (legend_label is not None):
        # Draws legend
        legend(legend_label, ax=ax)

    pylab.draw()
    #
    if len(Ax) == 1:
        if return_handles:
            return Ax[0], handles
        else:
            return Ax[0]
    else:
        if return_handles:
            return Ax, handles
        else:
            return Ax


def contour(x, y, z, title='', xlabel='', xunits='', ylabel='', yunits='',
    zunits='', label='', label_pos=[0.02, 0.95], fig=None, ax=None,
    subplot=(1, 1, 1), sharex=None, sharey=None, xlim=None, ylim=None,
    xscale='linear', yscale='linear', zscale='linear', nospines=False,
    scale=1., scale_label=None, crange=None, cticks=None,
    cmap=custom_cm.custom_viridis, colorbar=True, cbarpos=None,
    orientation='horizontal', extend='both', **kwargs):
    """
    """
    # OLD: cmap=cm.GMT_no_green
    #
    if fig == None:
        fig = figure()

    if ax == None:
        if len(subplot) == 3:
            ax = fig.add_subplot(subplot[0], subplot[1], subplot[2],
                sharex=sharex, sharey=sharey)
        elif len(subplot) == 4:
            ax = fig.add_axes(subplot, sharex=sharex, sharey=sharey)

    if nospines:
        dropspines(ax)
    ax.minorticks_on()
    ax.tick_params(direction='out', which='both')

    norm = None
    # Base 10 logarithmic scale
    if zscale == 'log':
        z = numpy.log10(z)
        crange = numpy.log10(crange)
    elif zscale == 'log2':
        z = numpy.log2(z)
        crange = numpy.log2(crange)
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
            cticks = dict(values = crange[::4])
    if zscale == 'log':
        # Checks if tick values are all integers.
        cticks['text'] = ['10$^{{{}}}$'.format(tick) for tick in
            cticks['values']]
    elif zscale == 'log2':
        cticks['text'] = ['2$^{{{%d}}}$' % (tick) for tick in cticks['values']]

    # Sets scale label according to scale
    if scale_label == None:
        log = int(numpy.log10(scale))
        scale = 10 ** log
        if log != 0:
            scale_label = r'\times 10^{%d}' % (log)
            crange /= scale
        else:
            scale_label = ''

    xmin, xmax = 9e9, 0
    xmin, xmax = min(xmin, x.min()), max(xmax, x.max())
    ymin, ymax = 9e9, 0
    ymin, ymax = min(ymin, y.min()), max(ymax, y.max())
    #bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    bbox = dict(boxstyle='square, pad=0.3', facecolor='w', edgecolor='none',
        alpha=0.9)

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
        elif orientation  in ['landscape', 'landscape.golden', 'worldmap',
            'horizontal']:
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
		try:
			pyplot.locator_params(axis='x', nbins=3)
		except:
			pass
		ax.set_xticklabels([common.num2latlon(i, 0, mode='each',
			x180=True, dtype='label')[1] for i in ax.get_xticks()])
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
        ax.text(label_pos[0], label_pos[1], label, ha='left', va='top',
            transform=ax.transAxes, bbox=bbox)

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


def reset():
    """Resets all graphical capabilities."""
    pylab.close('all')


def windrose(wind=None, speed=None, direction=None, npoints=32, bins=None,
    dt=None, mode='from', fig=None, subplot=(1, 1, 1), cmap=pyplot.cm.YlGn,
    legend_ncol=2, title='', label='', ylim=None, sharex=None, sharey=None):
    """
    Plots windrose for a series of measurements.

    Parameters
    ----------
    t : array like
        Time
    wind : complex array like
        Wind vector in complex form (u + 1j*v).
    speed : array like
        Array of speed (modulos of velocity).
    direction : array like
        Direction of velocity vector in cardinal degrees, i.e.
        zero is heading north.
    npoints : integer
        Number of cardinal points in compass. Values can be
        either 4, 8, 16 or 32 (default).
    bins : array like
        Bins of speed ranges to plot.
    dt : float, optional
        Time interval for each measurements. If set calculates
        frequency instead of percentage
    mode : string, optional
        Sets whether orientation is 'from' (default) or 'to'
        direction when input is the wind vector field.
    fig :
    cmap :
    legend_ncol : int, optional
        Sets the number of columns for the legend. If set to zero,
        then no legend is plotted. In this case it also returns the
        reference object to the for later use.

    Returns
    -------
    stats, ax, p_list, titles
    stats, ax, titles


    """
    # Checks input parameters
    if wind is not None:
        # Converts velocity vector to speed and direction arrays, checking
        # first if all data is masked
        try:
            if wind.mask.all():
                wind = numpy.zeros((wind.shape), dtype=complex)
        except:
            pass
        if not numpy.iscomplexobj(wind):
            raise ValueError('Wind vector data is not in complex format.')
        speed = abs(wind)
        if mode == 'from':
            direction = numpy.arctan2(-wind.real, -wind.imag)
        elif mode == 'to':
            direction = numpy.arctan2(wind.real, wind.imag)
        else:
            raise ValueError('Invalid mode `{}`.'.format(mode))
    elif direction != None:  # direction.max() > 2*numpy.pi:
        # Assumes that if direction is greater than 2 \pi, it is given
        # in degrees. If so, converts it to radians.
        try:
            if direction.mask.all():
                speed = direction = numpy.zeros((direction.shape))
        except:
            pass
        direction = numpy.deg2rad(direction)
    if bins is None:
        bins = [1, 5, 10, 15]
    if dt is None:
        dt = 1
        use_percent = True
    else:
        use_percent = False
    if fig is None:
        # Creates figure instance for the plot.
        fig = figure()
    # Initializes compass
    compass = Compass(N=npoints)
    delta = numpy.deg2rad(compass._delta)
    delta_half = delta / 2.
    rows = compass.list_cardinal_points(mode='radian')
    cardinal_angles = numpy.array([r[2] for r in rows])

    # Distributes along direction and speed.
    cols = len(bins) + 1
    titles = []
    stats = numpy.zeros((npoints+1, cols+1))
    stats[0, 0] = numpy.nan
    stats[0, 1:-1] = bins
    stats[0, -1] = numpy.inf
    direction_point = compass.from_angle_to_point(direction, mode='radian')
    calm = (speed == 0).sum()
    legend_labels = [0]
    for row in range(npoints):
        titles.append(rows[row][1])
        stats[row+1, 0] = numpy.rad2deg(rows[row][2])
        start = 1e-9
        for col in range(cols):
            try:
                stop = bins[col]
            except:
                stop = speed.max()
            #
            counter = ((direction_point == row) & (speed >= start) &
                (speed < stop)).sum()
            stats[row+1, col+1] = counter
            #
            if row == 0:
                legend_labels.append(stop)
            start = stop
    #
    total = calm + stats[1:, 1:].sum()
    stats[1:, 1:] /= total

    # Creates the plot!
    if len(subplot) == 3:
        ax = fig.add_subplot(subplot[0], subplot[1], subplot[2], polar=True,
            sharex=sharex, sharey=sharey)
    elif len(subplot) == 4:
        ax = fig.add_axes(subplot, polar=True, sharex=sharex, sharey=sharey)

    # Creates a blank bar, just in case there is no data to plot.
    p = ax.bar(0, 0.1, bottom=0, width=numpy.pi, linewidth=0, alpha=0)

    p_list = []
    bottom = numpy.zeros(npoints)
    for col in range(cols):
        if col < cols-1:
            if legend_ncol == 0:
                legend_label = r'${:.1f}$--${:.1f}$'.format(legend_labels[col],
                    legend_labels[col+1])
            else:
                legend_label = r'${:.1f}$--${:.1f}$: $({:.1f}\;\%)$'.format(
                    legend_labels[col], legend_labels[col+1],
                    stats[1:, col+1].sum()*100.)
        else:
            if legend_ncol == 0:
                legend_label = r'$>{:.1f}$'.format(legend_labels[col])
            else:
                legend_label = r'$>{:.1f}$: $({:.1f}\;\%)$'.format(
                    legend_labels[col], stats[1:, col+1].sum()*100.)
        #
        p = ax.bar(_from_cardinal_to_angle(cardinal_angles) - delta_half,
            stats[1:, col+1], bottom=bottom, width=delta, linewidth=0.25,
            color=cmap((col+1.)/cols), label=legend_label, alpha=0.8)
        p_list.append(p)
        #
        bottom += stats[1:, col+1]

    # Updates
    cardinal_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    cardinal_angles = numpy.arange(0, 360, 45)
    ax.set_thetagrids(angles=_from_cardinal_to_angle(cardinal_angles,
        mode='degree'), labels=cardinal_labels)
    if ylim != None:
        ax.set_ylim(ylim)
    if legend_ncol > 0:
        ax.legend(loc='upper center', fontsize='small', ncol=legend_ncol,
            bbox_to_anchor=(0.5, -0.1))
    if title:
        ax.text(0.5, 1.2, title, fontsize='large', ha='center', va='top',
            transform=ax.transAxes)
    if label:
        ax.text(0.5, 0.15, label, fontsize='large', ha='center', va='baseline',
            alpha=0.6, transform=ax.transAxes)

    if legend_ncol == 0:
        return stats, ax, p_list, titles
    else:
        return stats, ax, titles


def _from_cardinal_to_angle(a, mode='radian'):
    """Converts to cardinal direction to angle."""
    if mode == 'degree':
        return (90 - a) % 360.
    elif mode == 'radian':
        return (numpy.pi /2 - a) % (2 * numpy.pi)
