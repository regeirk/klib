# -*- coding: iso-8859-1 -*-
"""Mapping module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to assist in generating and manipulating
high quality maps and general plots (i.e. Hovmoller plots).

AUTHOR
    Sebastian Krieger
    email: solutions@nublia.com

REVISION
    4 (2011-11-18 01:57)
    3 (2011-08-31 23:09)
    2 (2011-04-14 14:48)

"""
from __future__ import division

__version__ = '$Revision: 4 $'
# $Source$

__all__ = ['map', 'hovmoller']

import numpy
import pylab

from time import time
from sys import stdout
from matplotlib import dates
from matplotlib import rcParams
from matplotlib.patches import Polygon
from scipy.stats import nanmean, nanstd
from mpl_toolkits.basemap import Basemap, pyproj, cm, shiftgrid

import common
import graphics


def __init__(show=False):
    if show:
        pylab.ion()


class Basemap(Basemap):
    def ellipse(self, x0, y0, a, b, n, ax=None, **kwargs):
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.
        
        For a description of the properties of ellipsis, please refer to [1].
        
        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].
        
        Extra keyword ``ax`` can be used to override the default axis instance.
        
        Other \**kwargs passed on to matplotlib.patches.Polygon
        
        RETURNS
            poly : a maptplotlib.patches.Polygon object.
        
        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse
            [2] : http://www.mail-archive.com/matplotlib-users@
                      lists.sourceforge.net/msg07606.html
        
        
        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b
        
        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = [self(x0+a, y0)]
        AZ = numpy.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if numpy.allclose(0., y0) and (numpy.allclose(90., az) or
                numpy.allclose(270., az)):
                continue

            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * numpy.pi / 360. * (az + 90.)
            A = dist[0] * numpy.sin(azr)
            B = dist[1] * numpy.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)
            
            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))
        
        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)
        
        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
        
        return poly


def map(lon, lat, z, z2=None, tm=None, projection='cyl', save='', ftype='png',
        crange=None, crange2=None, cmap=cm.GMT_no_green, show=False,
        shiftgrd=0., orientation='landscape', title='', label='', units='',
        subplot=None, adjustprops=None, loc=[], xlim=None, ylim=None,
        xstep=None, ystep=None, etopo=False, profile=True, hook=None, 
        **kwargs):
    """Generates maps.

    The maps can be either saved as image files or simply showed on
    screen.

    PARAMETERS
        lon, lat (array like) :
            Longitude and latitude arrays.
        z (array like) :
            Variable data array. For bi-dimensional MxN arrays, then a
            single map is plotted where M and N should have the same
            lengths as the latitude and the longitude respectively.

            For tri-dimensional TxMxN arrays, eather a sequence of maps
            is generated if T has the same length as tm or, in case tm
            is not set, T maps are plotted on the save figure.
        z2 (array like, optional) :
            Second variable to be plotted using simple line contours.
        t (array like, optional) :
            Time array. It should contain values in matplotlib date
            format (i.e. number of days since 0001-01-01 UTC).
        projection (text, optional) :
            Sets the map projection. Implemented projections are:
                cyl -- Equidistant cylindrical
                ortho -- Orthographic
                robin -- Robinson
                moll -- Mollweide
                eqdc -- Equidistant conic
                poly -- Polyconic
                omerc -- Oblique mercator
            Default is the equidistant cylindrical projection (cyl).
        save (string, optional) :
            The path in which the resulting plots are to be saved. If
            not set, then no images will be saved.
        ftype (string, optional) :
            The image file type. Most backends support png, pdf, ps,
            eps and svg.
        crange (array like, optional) :
            Sets the color range of the maps. If not given then the
            range is calculated from the input data.
        crange2 (array like, optional) :
            Sets the contour line interval.
        cmap (colormap, optional) :
            Sets the colormap to be used in the plots. The default is
            the Generic Mapping Tools (GMT) no green.
        show (boolean, optional) :
            If set to true the the resulting maps are explicitly shown
            on screen.
        shiftgrd (float, optional) :
            Shifts the longitude and variable data arrays east or west.
            Its value determines the starting longitude for the shifted
            grid.
            TODO: update functionality
        orientation (string, optional) :
            Sets the orientation of the figure. Allowed options are
            'landscape' (default), 'portrait', 'squared'.
        title (string, array like, optional) :
            Sets the map title. If array like, each element of the
            array becomes the title for each map. If the title is set
            to '%date%' then the ISO formated date is written.
        label (string, array like, optional) :
            Sets the label for each plot. If array like, each element
            of the array becomes the label for each plot.
        units (string, array like, optional) :
            Determines the units for all the maps of for each map
            sepparetely if a text array is given.
        subplot (array like, optional) :
            Two item list containing the number of rows and columns for
            subplots.
        adjustprops (dict, optional) :
            Dictionary containing the subplot parameters.
        loc (list, optional) :
            Lists the longitude of locations to be marked in map.
        xlim, ylim (array like, optional) :
            List containing the upper and lower zonal and meridional
            limits, respectivelly.
        xstep, ystep (float, optional) :
            Determines the parallel and meridian spacing.
        etopo (boolean, optional) :
            If true, overlays ETOPO contour lines on map.
        profile (boolean, optional) :
            Turns profiler on/off. If set to true (default) outputs the
            ETA and other information on screen.
        hook (function, optional) :
            Executes a hook function after the plot. The map instance
            is passed along as parameter.

    OUTPUT
        Map plots either on screen and or on file according to the
        specified parameters.

    RETURNS
        Nothing.

    """
    t1 = time()
    __init__()

    # Transforms input arrays in numpy arrays and numpy masked arrays.
    lat = numpy.asarray(lat)
    lon = numpy.asarray(lon)
    if type(tm).__name__ != 'NoneType':
        tm = numpy.asarray(tm)
    if type(z).__name__ != 'MaskedArray':
        z = numpy.ma.asarray(z)
        z.mask = numpy.isnan(z)

    # Determines the number of dimensions of the variable to be plotted and
    # the sizes of each dimension.
    dim = len(z.shape)
    if dim == 3:
        c, b, a = z.shape
    elif dim == 2:
        b, a = z.shape
        c = 1
        z = z.reshape(c, b, a)
    else:
        raise Warning, ('Map plots require either bi-dimensional or tri-'
                        'dimensional data.')
    if lon.size != a:
        raise Warning, 'Longitude and data lengths do not match.'
    if lat.size != b:
        raise Warning, 'Latitude and data lengths do not match.'
    #if type(tm).__name__ != 'NoneType':
    #    if tm.size != c:
    #        raise Warning, 'Time and data lengths do not match.'

    # Shifts the longitude and data grid if applicable and determines central
    # latitude and longitude for the map.
    lon180 = common.lon180(lon)
    if xlim == None:
        try:
            mask = ~z.mask.all(axis=0).all(axis=0)
            xlim = [lon180[mask].min(), lon180[mask].max()]
        except:
            xlim = [lon.min(), lon.max()]
    if ylim == None:
        try:
            mask = ~z.mask.all(axis=0).all(axis=1)
            ylim = [lat[mask].min(), lat[mask].max()]
        except:
            ylim = [lat.min(), lat.max()]
    lon0 = numpy.mean(xlim)
    lat0 = numpy.mean(ylim)
    if (shiftgrd != 0): # | (projection in ['ortho', 'robin', 'moll']):
        dx, dy = lon[1] - lon[0], lat[1] - lat[0]
        lon = lon180
        shift = pylab.find(pylab.diff(lon) < 0) + 1
        try:
          lon = numpy.roll(lon, -shift)
          z = numpy.roll(z, -shift)
        except:
          pass
        #z, lon = shiftgrid(shiftgrd, z, lon0)
        
        # Pad borders with NaN's to avoid distorsions
        #lon = numpy.concatenate([[lon[0] - dx], lon, [lon[-1] + dx]])
        #lat = numpy.concatenate([[lat[0] - dy], lat, [lat[-1] + dy]])
        #nan = numpy.ma.empty((c, 1, a)) * numpy.nan
        #nan.mask = True
        #z = numpy.ma.concatenate([nan, z, nan], axis=1)
        #nan = numpy.ma.empty((c, b+2, 1)) * numpy.nan
        #nan.mask = True
        #z = numpy.ma.concatenate([nan, z, nan], axis=2)
    
    # Loads topographic data, if appropriate.
    if etopo:
        ez = common.etopo.z
        ex = common.etopo.x
        ey = common.etopo.y
        er = -numpy.arange(1000, 12000, 1000)

    # Setting the color ranges
    if crange == None:
        cmajor, cminor, crange, cticks, extend = common.step(z,
            returnrange=True)
    else:
        crange = numpy.asarray(crange)
        cminor = numpy.diff(crange).mean()
        if crange.size > 11:
            cmajor = 2 * cminor
        if len(crange) < 15 :
            cticks = crange[::2]
        else:
            cticks = crange[::5]

        xmin, xmax = z.min(), z.max()
        rmin, rmax = crange.min(), crange.max()
        
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
    if type(z2).__name__ != 'NoneType' and crange2 == None:
        cmajor2, cminor2, crange2, cticks2, extend2 = common.step(z2,
            returnrange=True)

    # Turning interactive mode on or off according to show parameter.
    if show == False:
        pylab.ioff()
    elif show == True:
        pylab.ion()
    else:
        raise Warning, 'Invalid show option.'

    # Sets the figure properties according to the orientation parameter and to
    # the data dimensions.
    if adjustprops == None:
        if projection in ['cyl', 'eqdc', 'poly', 'omerc', 'vandg', 'nsper']:
            adjustprops = dict(left=0.1, bottom=0.15, right=0.95, top=0.9,
                               wspace=0.05, hspace=0.5)
        else:
            adjustprops = dict(left=0.05, bottom=0.15, right=0.95, top=0.9,
                               wspace=0.05, hspace=0.2)

    # Sets the meridian and the parallel coordinates and necessary parameters
    # depending on the chosen projection.
    if xstep == None:
        xstep = int(common.step(xlim, 5, kind='polar')[0])
    if ystep == None:
        ystep = int(common.step(ylim, 3, kind='polar')[0])
    merid = numpy.arange(10 * int(min(xlim) / 10 - 2),
                         10 * int(max(xlim) / 10 + 3), xstep)
    if (max(ylim) - min(ylim)) > 130 | (projection in ['ortho', 'robin', 
        'moll']):
        #paral = numpy.array([-(66. + 33. / 60. + 38. / (60. * 60.)),
        #                     -(23. + 26. / 60. + 22. / (60. * 60.)), 0.,
        #                      (23. + 26. / 60. + 22. / (60. * 60.)),
        #                      (66. + 33. / 60. + 38. / (60. * 60.))])
        #paral = numpy.round(paral)
        paral = numpy.array([-60, -30, 0, 30, 60])
    else:
        paral = numpy.arange(numpy.floor(min(ylim) / ystep) * ystep,
                             numpy.ceil(max(ylim) / ystep) * ystep + ystep,
                             ystep)
    if projection == 'eqdc':
        if not (('lat_0' in kwargs.keys()) and ('lat_1' in kwargs.keys())):
            kwargs['lat_0'] = min(ylim) + (max(ylim) - min(ylim)) / 3.
            kwargs['lat_1'] = min(ylim) + 2 * (max(ylim) - min(ylim)) / 3.
        if not ('lon_0' in kwargs.keys()):
            kwargs['lon_0'] = lon0
    elif projection == 'poly':
        if not ('lat_0' in kwargs.keys()):
            kwargs['lat_0'] = (max(ylim) - min(ylim)) / 2.
        if not ('lon_0' in kwargs.keys()):
            kwargs['lon_0'] = lon0
    elif projection == 'omerc':
        if not (('lat_0' in kwargs.keys()) and ('lat_1' in kwargs.keys())):
            kwargs['lat_1'] = min(ylim) + (max(ylim) - min(ylim)) / 4.
            kwargs['lat_2'] = min(ylim) + 3 * (max(ylim) - min(ylim)) / 4.
        if not (('lon_0' in kwargs.keys()) and ('lon_1' in kwargs.keys())):
            kwargs['lon_1'] = min(xlim) + (max(ylim) - min(ylim)) / 4.
            kwargs['lon_2'] = min(xlim) + 3 * (max(ylim) - min(ylim)) / 4.
        kwargs['no_rot'] = False
    elif projection == 'vandg':
        kwargs['lon_0'] = lon0
    elif projection == 'nsper':
        kwargs['lon_0'] = lon0
        kwargs['lat_0'] = lat0
    elif projection in ['aea', 'lcc']:
        kwargs['lon_0'] = lon0
        kwargs['lat_0'] = (min(ylim) + max(ylim)) / 2.
        kwargs['lat_1'] = max(ylim) - (max(ylim) - min(ylim)) / 4.
        kwargs['lat_2'] = min(ylim) + (max(ylim) - min(ylim)) / 4.

    # Setting the subplot parameters in case multiple maps per figure.
    try:
        plrows, plcols = subplot
    except:
        if type(tm).__name__ in ['NoneType', 'float']:
            if orientation in ['landscape', 'worldmap']:
                plcols = min(3, c)
                plrows = numpy.ceil(float(c) / plcols)
            elif orientation == 'portrait':
                plrows = min(3, c)
                plcols = numpy.ceil(float(c) / plrows)
            elif orientation == 'squared':
                plrows = plcols = numpy.ceil(float(c) ** 0.5)
        else:
            plcols = plrows = 1

    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)

    # Starts the plotting routines
    if profile:
        if c == 1:
            plural = ''
        else:
            plural = 's'
        s = 'Plotting %d map%s... ' % (c, plural)
        stdout.write(s)
        stdout.flush()

    fig = graphics.figure(fp=dict(), ap=adjustprops, orientation=orientation)
    for n in range(c):
        t2 = time()
        if plcols * plrows > 1:
            ax = pylab.subplot(plrows, plcols, n + 1)
        else:
            fig.clear()
            ax = pylab.subplot(plcols, plrows, 1)
        
        if (projection in ['ortho', 'robin', 'moll']):
            m = Basemap(projection=projection, lat_0=lat0, lon_0=lon0, *kwargs)
            xoffset = (m.urcrnrx - m.llcrnrx) / 50.
        elif projection in ['aea', 'cyl', 'eqdc', 'poly', 'omerc', 'vandg', 
                            'nsper', 'lcc']:
            m = Basemap(projection=projection, llcrnrlat=min(ylim),
                        urcrnrlat=max(ylim), llcrnrlon=min(xlim),
                        urcrnrlon=max(xlim), **kwargs)
            xoffset = None
        else:
            raise Warning, 'Projection \'%s\' not implemented.' % (projection)

        x, y = m(*numpy.meshgrid(lon, lat))
        dat = z[n, :, :]
        
        # Set the merdians' and parallels' labels
        if plcols * plrows > 1:
            if (n % plcols) == 0:
                plabels =  [1, 0, 0, 0]
            else:
                plabels = [0, 0, 0, 0]
            if (n >= c - plcols):
                mlabels = [0, 0, 0, 1]
            else:
                mlabels = [0, 0, 0, 0]
        else:
            mlabels = [0, 0, 0, 1]
            plabels = [1, 0, 0, 0]
        if projection in ['ortho']:
            plabels = [0, 0, 0, 0]
        if projection in ['geos', 'ortho', 'aeqd', 'moll']:
            mlabels = [0, 0, 0, 0]

        # Plots locations
        for item in loc:
            m.scatter(item[0], item[1], s=24, c='w', marker='o', alpha=1, 
                      zorder=99)

        # Plot contour
        im = m.contourf(x, y, dat, crange, cmap=cmap, extend=extend, hold='on')

        if type(z2).__name__ != 'NoneType':
            dat2 = z2[n, :, :]
            im2 = m.contour(x, y, dat2, crange2, colors='k', hatch='x',
                hold='on', linewidths=numpy.linspace(0.25, 2., len(crange2)),
                alpha=0.6)
            #pylab.clabel(im2, fmt='%.1f')

        # Plot topography, if appropriate
        if etopo:
            xe, ye = m(*numpy.meshgrid(ex, ey))
            cs = m.contour(xe, ye, ez, er, colors='k', linestyles='-',
                alpha=0.3, hold='on')

        # Run hook function, if appropriate
        try:
            hook(m)
        except:
            pass

        m.drawcoastlines()
        m.fillcontinents()
        m.drawcountries()
        if projection != 'nsper':
            m.drawmapboundary(fill_color='white')
        m.drawmeridians(merid, linewidth=0.5, labels=mlabels)
        m.drawparallels(paral, linewidth=0.5, labels=plabels, xoffset=xoffset)
        
        # Draws colorbar
        if orientation == 'squared':
            cx = pylab.axes([0.25, 0.07, 0.5, 0.03])
        elif orientation  in ['landscape', 'worldmap']:
            cx = pylab.axes([0.2, 0.05, 0.6, 0.03])
        elif orientation == 'portrait':
            cx = pylab.axes([0.25, 0.05, 0.5, 0.02])
        pylab.colorbar(im, cax=cx, orientation='horizontal', ticks=cticks,
                       extend=extend)

        # Titles, units and other things
        ttl = None
        if title.__class__ == str:
            ttl = title
        else:
            try:
                ttl = title[n]
            except:
                pass
        if ttl:
            if ttl == '%date%':
                try:
                    ttl = dates.num2date(tm[n]).isoformat()[:10]
                except:
                    try:
                        ttl = dates.num2date(tm).isoformat()[:10]
                    except:
                        ttl = ''
                        pass
            ax.text(0.5, 1.05, ttl, ha='center', va='baseline',
                transform=ax.transAxes)
        
        lbl = None
        if label.__class__ == str:
            lbl = label
        else:
            try:
                lbl = label[n]
            except:
                pass
        if lbl:
            if lbl == '%date%':
                try: 
                    ttl = dates.num2date(tm[n]).isoformat()[:10]
                except:
                    try:
                        ttl = dates.num2date(tm).isoformat()[:10]
                    except:
                        ttl = ''
                        pass
            ax.text(0.04, 0.83, lbl, ha='left', va='bottom', 
                transform=ax.transAxes, bbox=bbox)

        unt = None
        if units.__class__ == str:
            unt = units
        else:
            try:
                unt = units[n]
            except:
                pass
        if unt:
            cx.text(1.05, 0.5, r'$\left[%s\right]$' % (unt), ha='left',
                va='center', transform=cx.transAxes)

        # Drawing and saving the figure if appropriate.
        pylab.draw()
        if save:
            if (c == 1) | (plcols * plrows > 1):
                pylab.savefig('%s.%s' % (save, ftype), dpi=150)
            else:
                pylab.savefig('%s%06d.%s' % (save, n+1, ftype), dpi=150)

        if profile:
            stdout.write(len(s) * '\b')
            s = 'Plotting %d map%s... %s ' % (c, plural, common.profiler(c, 
                n + 1, 0, t1, t2),)
            stdout.write(s)
            stdout.flush()

    #
    if profile:
        stdout.write('\n')
    if show == False:
        pylab.close(fig)
    else:
        return fig


def hovmoller(lon, tm, z, zo=None, zz=None, title=None, label=None,
              labels=dict(), crange=None, cmap=cm.GMT_no_green,
              orientation='landscape', show=False, save='', ftype='png',
              adjustprops=None, bottom=None, right=None, loc=[], std=None,
              xunits='deg', draft=False, hookx=None, hooky=None):
    """Hovmoller plots.

    PARAMETERS
        lon (array like) :
            Longitude axis.
        tm (array like) :
            Time axis.
        z (array like) :
            Filled contour variable.
        zo (array like) :
            Overlapping contour variable (e.g. relative significance of
            wavelet analysis) to be ploted with a thick solid black line.
        zz (array like) :
            Another overlapping contour variagle (e.g. original data) to be
            ploted with a thin solid white line.
        title (string, array like, optional) :
            Sets the contour plot title. If array like, each element of
            the array becomes the title for plot.
        label (string, array like, optional) :
            Sets the label for each plot. If array like, each element
            of the array becomes the label for each plot.
        labels (dictionary, optional) :
            Sets the labels for the plot axis.
        units (string, array like, optional) :
            Determines the units for all the contours together or
            sepparatelly.
        crange (array like, optional) :
            Sets the color range of the maps. If not given then the
            range is calculated from the input data.
        cmap (colormap, optional) :
            Sets the colormap to be used in the plots. The default is
            the Generic Mapping Tools (GMT) no green.
        orientation (string, optional) :
            Sets the orientation of the figure. Allowed options are
            'landscape' (default), 'portrait', 'squared'.
        show (boolean, optional) :
            If set to true the the resulting maps are explicitly shown
            on screen.
        save (string, optional) :
            The path in which the resulting plots are to be saved. If
            not set, then no images will be saved.
        ftype (string, optional) :
            The image file type. Most backends support png, pdf, ps,
            eps and svg.
        adjustprops (dict, optional) :
            Dictionary containing the subplot parameters.
        bottom (string, optional) :
            If set to ether 'std' or 'avg' plots respectively the
            standard deviation or mean of the signal at the bottom.
        loc (list, optional) :
            Lists the longitude of locations to be marked in plot.
        xunit (string, optional) :
            Determines the x-axis unit. Valid options are either 'deg'
            for degrees (default) or 'km' for kilometers.
        draft (boolean, optional) :
            If set to true, then reduces the size of the colorbar to
            approximatelly two colors to save time. Default is false.
        hookx, hooky (function, optional) :
            Executes a hook function after the plot in the x and y
            axes, respectivelly.

    OUTPUT
        Hovmoller contour plots plots either on screen and or on file
        according to the specified parameters.

    RETURNS
        Nothing.

    """
    t1 = time()
    __init__()

    # Setting undefined label strings.
    if 'units' not in labels.keys():
        labels['units'] = ''
    if 'Year' not in labels.keys():
        labels['Year'] = 'Year'
    if 'std' not in labels.keys():
        labels['std'] = 'Std'
    if 'avg' not in labels.keys():
            labels['avg'] = 'Avg'

    # Transforms input arrays in numpy arrays and numpy masked arrays.
    lon = numpy.asarray(lon)
    tm = numpy.asarray(tm)
    if type(z).__name__ != 'MaskedArray':
        z = numpy.ma.asarray(z)
        z.mask = numpy.isnan(z)
    else:
        z.mask = z.mask | numpy.isnan(z.data)

    # Determines the number of dimensions of the variable to be plotted and
    # the sizes of each dimension.
    dim = len(z.shape)
    if dim == 3:
        c, b, a = z.shape
    elif dim == 2:
        b, a = z.shape
        c = 1
        z = z.reshape(c, b, a)
    else:
        raise Warning, ('Hovmoller plots require either bi-dimensional or tri-'
                        'dimensional data.')
    if lon.size != a:
        raise Warning, 'Longitude and data lengths do not match.'
    if tm.size != b:
        raise Warning, 'Time and data lengths do not match.'

    if type(zo).__name__ != 'NoneType':
        dimo = len(zo.shape)
        if dimo == 2:
            co, ao = zo.shape
            bo = b
            zo = zo * numpy.ones([bo, co, ao])
        if (co != c) | (ao != a):
                raise Warning ('Overlapping array dimensions do not match')
        overlap = True
    else:
        overlap = False
    if type(zz).__name__ != 'NoneType':
        dimz = len(zz.shape)
        if dimz == 2:
            bz, az = zz.shape
            cz = 0
        elif dimz == 3:
            cz, bz, az = zz.shape
        else:
            cz = bz = az = 0
        if (bz != b) | (az != a):
                raise Warning ('Overlapping array dimensions do not match')
        zero = True
    else:
        zero = False

    # Verifies if title, label, unit and std parameters have the same number
    # of items as the number of plots to be drawn.
    if type(title).__name__ == 'str':
        title = [title] * c
    elif type(title).__name__ in ['list', 'tuple', 'ndarray']:
        C = len(title)
        if c > C:
            title = list(title) * int(numpy.ceil(float(c) / C))
    if type(label).__name__ == 'str':
        label = [label] * c
    elif type(label).__name__ in ['list', 'tuple', 'ndarray']:
        C = len(label)
        if c > C:
            label = list(label) * int(numpy.ceil(float(c) / C))

    # If the edges contain only NaN's, then slice them out.
    sel = pylab.find(~numpy.isnan(z.data).all(axis=0).all(axis=0))
    if len(sel) != a:
        a = len(sel)
        if a == 0:
            return
        lon = lon[sel[0]:sel[-1]]
        z = z[:, :, sel[0]:sel[-1]]
        if overlap:
            zo = zo[:, :, sel[0]:sel[-1]]
        if zero:
            if dimz == 2:
                zz = zz[:, sel[0]:sel[-1]]
            elif dimz == 3:
                zz = zz[:, :, sel[0]:sel[-1]]

    # Setting the color ranges
    bbox = dict(edgecolor='w', facecolor='w', alpha=0.9)
    if crange == None:
        cmajor, cminor, crange, cticks, extend = common.step(z,
            returnrange=True)
    else:
        crange = numpy.asarray(crange)
        cminor = numpy.diff(crange).mean()
        if crange.size > 11:
            cmajor = 2 * cminor
        if len(crange) < 15 :
            cticks = crange[::2]
        else:
            cticks = crange[::5]

        xmin, xmax = z.min(), z.max()
        rmin, rmax = crange.min(), crange.max()
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
    if draft:
        crange = [min(crange), crange[len(crange) / 2], max(crange)]

    # Turning interactive mode on or off according to show parameter.
    if show == False:
        pylab.ioff()
    elif show == True:
        pylab.ion()
    else:
        raise Warning, 'Invalid show option.'

    # Sets the figure properties according to the orientation parameter and to
    # the data dimensions including the subplot number of rows and columns.
    if orientation == 'landscape':
        #figprops = dict(figsize=(7.33, 5.33), dpi=96)
        plcols = c
        plrows = 1
    elif orientation == 'portrait':
        #figprops = dict(figsize=(5.33, 7.33), dpi=96)
        plcols = 1
        plrows = c
    elif orientation == 'squared':
        #figprops = dict(figsize=(5.33, 5.33), dpi=96)
        plrows = plcols = numpy.ceil(c ** 0.5)
    else:
        raise Warning, 'Orientation \'%s\' not allowed.' % (orientation, )
    if adjustprops == None:
        adjustprops = dict(left=0.1, bottom=0.15, right=0.99, top=0.9,
                           wspace=0.05, hspace=0.02)
    
    fig = graphics.figure(ap=adjustprops, orientation=orientation)

    # Some figure parameters definitions and initializations
    if bottom:
        bottommin, bottommax = [0, -65535]
        baxes = []
    if right:
        rightmin, rightmax = [0, -65535]
        grey = [0.66, 0.66, 0.66]
        lfmt = ['-', '-', '-', '-']
        lclr = ['k', 'k', grey, grey]
        lwth = [2., 1., 2., 1.]
        c += 1 # Adds one more sub-plot for size calculations only.

    # Subplot width and height parameters
    w = ((adjustprops['right'] - adjustprops['left']) / c -
        adjustprops['hspace'])
    if bottom:
        y = 0.25 + adjustprops['bottom'] - adjustprops['wspace']
    else:
        y = adjustprops['bottom']
    if bottom:
        h = 0.75 - adjustprops['bottom']
    else:
        h = adjustprops['top'] - y + adjustprops['wspace']

    if right:
        c -= 1 # Adjusts to original number of Hovmoller sub-plots.

    for k in range(c):
        x = (w + adjustprops['hspace']) * k + adjustprops['left']

        if k == 0:
            ax = pylab.axes([x, y, w, h])
            bx = ax
            if right:
                xx = ((w + adjustprops['hspace']) * c + adjustprops['left'])
                rx = pylab.axes([xx, y, w, h], sharey=ax)
                pylab.setp(rx.get_yticklabels(), visible=False)
                pylab.axes(bx)
        else:
            bx = pylab.axes([x, y, w, h], sharex=ax, sharey=ax)

        if zero:
            if dimz == 2:
                oz = zz
            elif dimz == 3:
                oz = zz[k, :, :]
            pylab.contour(lon, tm, oz, [-1e10, 0, 1e10],
                colors=[[0.9, 0.9, 0.9]], linestyles='-', linewidths=0.5,
                alpha=0.9)

        if overlap:
            if dimo == 2:
                o = (z[k, :, :].data >= zo[:, k, :])
            elif dimo == 3:
                o = (z[k, :, :].data >= zo[k, :, :])
            pylab.contour(lon, tm, o, [0, 1],
                colors='k', linestyles='-', linewidths=1.)

        # Plots the contour. Uses assigned data for power hovmollers.
        pylab.contourf(lon, tm, z[k, :, :].data, crange, cmap=cmap,
            extend=extend)

        # Running x and y hooks on current axis.
        try:
            hookx(bx)
        except:
            pass
        try:
            hooky(bx)
        except:
            pass

        if right:
            if right == 'std':
                rz = nanstd(z[k, :, :].data, axis=1)
            elif right == 'avg':
                rz = nanmean(z[k, :, :].data, axis=1)
            rx.plot(rz, tm, lfmt[k], color=lclr[k], linewidth=lwth[k])

            # Running y hook on right axis.
            try:
                hooky(rx)
            except:
                pass

            rightmin = min([numpy.nanmin(rz), rightmin])
            rightmax = max([numpy.nanmax(rz), rightmax])

        for i in loc:
            pylab.plot([i, i], [tm.min(), tm.max()], 'D', markersize=14,
                color='w', alpha=1)

        if bottom:
            yb, hb = adjustprops['bottom'], 0.175
            if k == 0:
                cx = pylab.axes([x, yb, w, hb], sharex=ax)
                dx = cx
            else:
                dx = pylab.axes([x, yb, w, hb], sharex=ax, sharey=cx)
            baxes.append(dx)
            if bottom == 'avg':
                bz = nanmean(z[k, :, :].data, axis=0)
            elif bottom == 'std':
                bz = nanstd(z[k, :, :].data, axis=0)
            pylab.plot(lon, bz, 'k-')
            
            # Running x hook on bottom axis.
            try:
                hookx(dx)
            except:
                pass

            bottommin = min([numpy.nanmin(bz), bottommin])
            bottommax = max([numpy.nanmax(bz), bottommax])
            #
            pylab.setp(bx.get_xticklabels(), visible=False)
            if k > 0:
                pylab.setp(dx.get_yticklabels(), visible=False)

        if k == 0:
            if orientation == 'landscape':
                corientation = 'horizontal'
                cax = pylab.axes([adjustprops['left'] + 0.15, 0.05,
                    adjustprops['right'] - adjustprops['left'] - 0.3, 0.03])
                ci, cj, ha, va = 1.05, 0.5, 'left', 'center'
            elif orientation == 'portrait':
                corientation = 'vertical'
                cax = pylab.axes([adjustprops['right'] + 0.02, y + 0.05, 
                    0.03, h - 0.1])
                ci, cj, ha, va = 0.5, -0.05, 'center', 'baseline'
            pylab.colorbar(cax=cax, ax=ax, orientation=corientation,
                extend=extend, ticks=cticks)
            if labels['units']:
                cax.text(ci, cj, r'$\left[%s\right]$' % (labels['units']),
                         ha=ha, va=va, transform=cax.transAxes)
        else:
            pylab.setp(bx.get_yticklabels(), visible=False)

        if title:
            bx.set_title('%s' % (title[k]), va='baseline', fontsize='medium')
        if label:
            bx.text(0.07, 0.97, '%s' % (label[k]), ha='left', va='top',
                transform=bx.transAxes, bbox=bbox)

    # Formatting the plot axis.
    if bottom:
        ystep, ystep1 = common.step([bottommin, bottommax], 1.5)
        bottommax = pylab.ceil(bottommax / ystep) * ystep

        for dx in baxes:
            for i in loc:
                dx.plot([i, i], [bottommin, bottommax], 'D', markersize=10,
                    color='w', alpha=1)

        cx.set_ylim([bottommin, bottommax])
        ymajor = pylab.matplotlib.ticker.MultipleLocator(ystep)
        yminor = pylab.matplotlib.ticker.MultipleLocator(ystep1)
        cx.yaxis.set_major_locator(ymajor)
        cx.yaxis.set_minor_locator(yminor)
        if labels['units']:
            cx.set_ylabel(r'\textbf{%s} $\left[%s\right]$' % (labels[bottom],
                labels['units']))
        else:
            cx.set_ylabel(r'\textbf{%s}' % (labels[bottom]))
        if xunits == 'km':
            cx.set_xlabel(r'\textbf{%s}' % (xunits))
    else:
        if xunits == 'km':
            ax.set_xlabel(r'\textbf{%s}' % (xunits))
    if right:
        xstep, xstep1 = common.step([rightmin, rightmax], 1.5)
        rightmax = pylab.ceil(rightmax / xstep) * xstep

        rx.set_xlim([rightmin, rightmax])
        xmajor = pylab.matplotlib.ticker.MultipleLocator(xstep)
        xminor = pylab.matplotlib.ticker.MultipleLocator(xstep1)
        rx.xaxis.set_major_locator(xmajor)
        rx.xaxis.set_minor_locator(xminor)
        if labels['units']:
            rx.set_title(r'%s $\left[%s\right]$' % (labels[right],
                labels['units']))
        else:
            rx.set_title(r'%s' % (labels[right]))

    ax.set_xlim([lon.min(), lon.max()])
    ax.set_ylim([tm.min(),  tm.max()])
    if xunits == 'deg':
        xstep, xstep1 = common.step(lon, 2)
    elif xunits == 'km':
        xstep, xstep1 = common.step(lon, 5)
    xmajor = pylab.matplotlib.ticker.MultipleLocator(xstep)
    xminor = pylab.matplotlib.ticker.MultipleLocator(xstep1)
    ax.xaxis.set_major_locator(xmajor)
    ax.xaxis.set_minor_locator(xminor)
    graphics.timeformat(ax, dt=tm[-1]-tm[0], axis='y')
    if xunits == 'deg':
        ax.set_xticklabels([common.num2latlon(i, 0, mode='each', x180=False,
            dtype='label')[1] for i in ax.get_xticks()])
    ax.set_ylabel(r'\textbf{%s}' % (labels['Year']))

    # Drawing and saving the figure if appropriate.
    pylab.draw()
    if save:
        pylab.savefig('%s.%s' % (save, ftype), dpi=150)
    if show == False:
        pylab.close(fig)
