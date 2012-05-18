# -*- coding: iso-8859-1 -*-
"""File management module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to assist in manipulating large data
sets of geophysical data.

AUTHOR
    Sebastian Krieger
    email: solutions@nublia.com

REVISION
    3 (2012-02-17 22:19)
    2 (2011-10-16 01:46)
    1 (2011-01-12 00:16)

"""

__version__ = '$Revision: 3 $'
# $Source$

__all__ = ['load_map', 'load_dataset', 'save_map', 'save_dataset']

import os
import numpy
import pylab
import warnings
import common
import interpolate

from time import time
from string import atof

import common


def detect_ftype(fname):
    """Extracts the type of file saved from string
    
    PARAMETERS
        fname (string):
            Name of the file to be analysed.
    
    RETURNS
        ftype (string):
            Type of file.

    """
    for s in ['xy', 'xt', 'ty']:
        i = fname.find(s)
        if i >= 0:
            return fname[i:i+len(s)]
    
    return False


def load_map(fullpath, ftype='xy', delimiter='\t', masked=False, topomask=None,
    xlim=None, ylim=None, lon=None, lat=None, tm=None, lon180=False, 
    pad=False):
    """Loads an individual data file.

    PARAMETERS
        fullpath (string) :
        ftype (string, optional) :
        delimiter (string, optional) :
        masked (boolean, optional) :
        topomask (string, optional) :
            Topography mask.
        xlim, ylim (array like, optional) :
            List containing the upper and lower zonal and meridional
            limits, respectivelly.
        lon, lat, tm (array like, optional):
        lon180 (boolean, optional):
        pad (boolean, optional):
            Pads edges with NaN's for maps without distorsions.

    RETURNS
        lon, lat, tm, z
    """
    if topomask != None:
        masked = True

    dat = numpy.loadtxt('%s' % (fullpath), delimiter=delimiter)
    if ftype == 'xy':
        x = dat[0, 1:]
        y = dat[1:, 0]
        t = dat[0, 0]
    elif ftype == 'xt':
        x = dat[0, 1:]
        y = dat[0, 0]
        t = dat[1:, 0]
    elif ftype == 'ty':
        x = dat[0, 0]
        y = dat[1:, 0]
        t = dat[0, 1:]
    else:
        raise Warning, 'Type \'%s\' not recognised' % (ftype)

  
    # Pads edges with NaN's to avoid distortions when generating maps.
    if pad:
        dx, dy = x[1] - x[0], y[1] - y[0]
        if (lon == None) and (ftype in ['xt', 'xt']):
            lon = numpy.concatenate([[x[0] - dx], x, [x[-1] + dx]])
        if (lat == None) and (ftype in ['xy', 'ty']):
            lat = numpy.concatenate([[y[0] - dy], y, [y[-1] + dy]])

    if masked:
        z = numpy.ma.asarray(dat[1:, 1:])
        z.mask = numpy.isnan(z)
    else:
        z = dat[1:, 1:]
    
    # Put data in appropriate place.
    if (lon != None) or (lat != None) or (tm != None):
        if lon == None:
            lon = x
        if lat == None:
            lat = y
        if tm == None:
            tm = t
        
        lon180 = common.lon180(lon)
        x180 = common.lon180(x)
        
        if ftype == 'xy':
            k, l, m = len(lon), len(lat), 1
            if masked:
                Z = numpy.ma.empty((l, k)) * numpy.nan
                Z.mask = True
            else:
                Z = numpy.empty((l, k)) * numpy.nan
            #
            u = numpy.unique(set(lon180) & set(x180))
            v = numpy.unique(set(lat) & set(y))
            if (len(u) == 0) or (len(v) == 0):
                return False
            i = [pylab.find(x180 == a)[0] for a in u]
            j = [pylab.find(y == b)[0] for b in v]
            mx, my = numpy.meshgrid(i, j)
            k = [pylab.find(lon180 == a)[0] for a in u]
            l = [pylab.find(lat == b)[0] for b in v]
            nx, ny = numpy.meshgrid(k, l)
            Z[ny, nx] = z[my, mx]
        
        if ftype == 'xt':
            k, l, m = len(lon), 1, len(tm)
            if masked:
                Z = numpy.ma.empty((m, k)) * numpy.nan
                Z.mask = True
            else:
                Z = numpy.empty((m, k)) * numpy.nan
            #
            u = numpy.unique(set(lon180) & set(x180))
            if len(u) == 0:
                return False
            i = [pylab.find(x180 == a)[0] for a in u]
            k = [pylab.find(lon180 == a)[0] for a in u]
            v = set(tm) & set(t)
            if len(v) == 0:
                return False
            j = [pylab.find(t == b)[0] for b in v]
            l = [pylab.find(tm == b)[0] for b in v]
            
            mx, my = numpy.meshgrid(i, j)
            nx, ny = numpy.meshgrid(k, l)
            Z[ny, nx] = z[my, mx]
        
        if ftype == 'ty':
            raise Warning, ('Loading of temporal-meridional files not '
                'implemented yet.')
    else:
        lon = x
        lat = y
        tm = t
        Z = z
    
    if (xlim != None) or (ylim != None):
        if xlim == None:
            xlim = [lon.min(), lon.max()]
        if ylim == None:
            ylim = [lat.min(), lat.max()]
        selx = pylab.find((lon >= min(xlim)) & (lon <= max(xlim)))
        sely = pylab.find((lat >= min(ylim)) & (lat <= max(ylim)))
        lon = lon[selx]
        lat = lat[sely]
        i, j = numpy.meshgrid(selx, sely)
        Z = Z[j, i]
    
    if topomask != None:
        # Interpolates topography into data grid.
        ezi, _, _ = interpolate.nearest([common.etopo.x, 
            common.etopo.y], common.etopo.z, [lon, lat])
        if topomask == 'ocean':
            Z.mask = Z.mask | (ezi > 0)
        elif topomask == 'land':
            Z.mask = Z.mask | (ezi < 0)

    if masked:
        Z.data[Z.mask] = 0

    return lon, lat, tm, Z

def load_dataset(path, pattern='(.*)', ftype='xy', flist=None, delimiter='\t',
                 var_from_name=False, masked=False, xlim=None, ylim=None, 
                 lon=None, lat=None, tm=None, topomask=None, verbose=False):
    """Loads an entire dataset.

    It uses the numpy.loadtxt function and therefore accepts regular
    ASCII files or GZIP compressed ones.

    PARAMETERS
        path (string) :
            The path in which the data files are located.
        pattern (string, optional) :
            Regular expression pattern correspondig to valid file names
            to be loaded.
        ftype (string, optional) :
            Specifies the file type that is loaded. The accepted values
            are 'xy', 'xt' and 'ty'.

            For 'xy', or map, files, the first line contains the
            longitude coordinates, the first column contains the
            latitude coordinates and the rest contains the data in
            matrix style. If var_from_name is set to True, it assumes
            that the time is given at the upper left cell.

            For 'xt', or zonal-temporal, files, the first line contains
            the longitude coordinates, the first column contains the
            time and the rest contains the data in matrix style. If
            var_from_name is set to True, it assumes that the latitude
            is given at the upper left cell.

            For 'ty', or temporal-meridional, files, the first line
            contains the time, the first column contains the longitude
            and the rest contains the data in matrix style. If
            var_from_name is set to True, it assumes that the latitude
            is given at the upper left cell.
        flist (array like, optional) :
            Lists the files to be loaded in path. If set, it ignores the
            pattern.
        delimiter (string, optional) :
            Specifies the data delimiter used while loading the data.
            The default value is '\t' (tab)
        var_from_name (boolean, optional) :
            If set to true, it tries to infer eather the time, latitude
            or longitude from the first match in pattern according to
            the chosen file type. If set to true, the pattern has to be
            set in such a way that the last matches contain the value
            and the hemisphere ('N', 'S', 'E' or 'W') if appropriate.
        masked (boolean, optional) :
            Returnes masked array. Default is False.
        xlim, ylim (array like, optional) :
            List containing the upper and lower zonal and meridional
            limits, respectivelly.
        lon, lat, tm (array like, optional):
        topomask (string, optional) :
            Topography mask.
        verbose (boolean, optional) :
            If set to true, does not print anything on screen.
            

    RETURNS
        lon (array like) :
            Longitude.
        lat (array like) :
            Latitude.
        t (array like) :
            Time.
        z (array like) :
            Loaded variable.

    """
    t0 = time()
    
    if topomask != None:
        masked = True

    S = 'Preparing data'
    s = '%s...' % (S)
    if not verbose:
        os.sys.stdout.write(s)
        os.sys.stdout.flush()

    # Generates list of files and tries to match them to the pattern
    if flist == None:
        flist = os.listdir(path)
        flist, match = common.reglist(flist, pattern)
    
    # Loads all the data from file list to create arrays
    N = len(flist)
    if N == 0:
        raise Warning, 'No files to be loaded.'

    # Initializes the set of array limits
    Lon = set()
    Lat = set()
    Tm = set()
    # Walks through the file loading process twice. At the first step loads
    # all the files to get all the geographical and temporal boundaries. At the
    # second step, reloads all files and fits them to the initialized data 
    # arrays
    for step in range(2):
        t1 = time()
        for n, fname in enumerate(flist):
            t2 = time()
            
            if (lon != None) and (lat != None) and (tm !=None):
                continue

            x, y, t, z = load_map('%s/%s' % (path, fname), ftype=ftype,
                delimiter=delimiter, lon=lon, lat=lat, tm=tm, masked=masked,
                topomask=topomask)

            if var_from_name:
                if (ftype == 'xt') | (ftype == 'ty'):
                    var = atof(match[n][-2])      # Gets coordinate out of ...
                    rav = match[n][-1].upper()    # ... match and also its ...
                    if (rav == 'S' | rav == 'W'): # ... hemisphere.
                        var *= -1
                    if ftype == 'xt':
                        y = var
                    else:
                        x = var
                elif ftype == 'xy':
                    t = atof(match[n][-1])       # Gets time out of last match.
            
            if numpy.isnan(t).all():
                t = 0
            
            if type(x).__name__ in ['int', 'long', 'float', 'float64']:
                x = [x]
            if type(y).__name__ in ['int', 'long', 'float', 'float64']:
                y = [y]
            if type(t).__name__ in ['int', 'long', 'float', 'float64']:
                t = [t]
            
            ###################################################################
            # FIRST STEP
            ###################################################################
            if step == 0:
                Lon.update(x)
                Lat.update(y)
                Tm.update(t)
            ###################################################################
            # SECOND STEP
            ###################################################################
            elif step == 1:
                selx = [pylab.find(Lon == i)[0] for i in x]
                sely = [pylab.find(Lat == i)[0] for i in y]
                selt = [pylab.find(Tm == i)[0] for i in t]
                
                i, j, k = common.meshgrid2(selx, sely, selt)
                
                if ftype == 'xt':
                    a, b, c = i.shape
                    z = z.reshape((a, 1, c))
                
                # Makes sure only to overwrite values not previously assigned.
                if masked:
                    Z[k, j, i] = numpy.ma.where(~Z[k, j, i].mask, 
                        Z[k, j, i], z)
                else:
                    Z[k, j, i] = numpy.where(~numpy.isnan(Z[k, j, i]), 
                        Z[k, j, i], z)
            
            ###################################################################
            # PROFILING
            ###################################################################
            if not verbose:
                os.sys.stdout.write(len(s) * '\b')
            s = '%s (%s)... %s ' % (S, fname, common.profiler(N, n + 1, t0, t1,
                t2))
            if not verbose:
                os.sys.stdout.write(s)
                os.sys.stdout.flush()
        #
        if not verbose:
            os.sys.stdout.write('\n')
    
        # Now creates data array based on input parameters xlim, ylim and
        # the loaded coordinate sets.
        if step == 0:
            if lon == None:
                Lon = numpy.asarray(list(Lon))
            else:
                Lon = lon
            if lat == None:
                Lat = numpy.asarray(list(Lat))
            else:
                Lat = lat
            if tm == None:
                Tm = numpy.asarray(list(Tm))
            else:
                Tm = tm
            
            Lon.sort()
            Lat.sort()
            Tm.sort()

            # Makes sure that all the coordinates are continuous, equally
            # spaced and that they are inside the coordinate limits.
            dx, dy, dt = numpy.diff(Lon), numpy.diff(Lat), numpy.diff(Tm)
            
            if len(dx) == 0: dx = numpy.array([1.])
            if len(dy) == 0: dy = numpy.array([1.])
            if len(dt) == 0: dt = numpy.array([1.])
            
            #if ((not (dx == dx[0]).all()) or (not (dy == dy[0]).all()) or 
            #    (not (dt == dt[0]).all())):
            #    raise Warning, 'One or more coordinates are not evenly spaced.'
            
            dx = dx[0]
            dy = dy[0]
            dt = dt[0]
            
            if xlim == None:
                xlim = [Lon.min(), Lon.max()]
            if ylim == None:
                ylim = [Lat.min(), Lat.max()]

            selx = pylab.find((Lon >= min(xlim)) & (Lon <= max(xlim)))
            Lon = Lon[selx]
            sely = pylab.find((Lat >= min(ylim)) & (Lat <= max(ylim)))
            Lat = Lat[sely]
            
            # Pads edges with NaN's to avoid distortions when generating maps.
            if lon == None:
                Lon = numpy.concatenate([[Lon[0] - dx], Lon, [Lon[-1] + dx]])
            if lat == None:
                Lat = numpy.concatenate([[Lat[0] - dy], Lat, [Lat[-1] + dy]])
            
            # Initializes data arrays
            a, b, c = Lon.size, Lat.size, Tm.size
            if masked:
                Z = numpy.ma.empty([c, b, a], dtype=float) * numpy.nan
                Z.mask = True
            else:
                Z = numpy.empty([c, b, a], dtype=float) * numpy.nan
            lon, lat = numpy.array(Lon), numpy.array(Lat)
            
            # Now everything might be ready for the second step in the loop,
            # filling in the data array.
            S, s = 'Loading data', ''

    # Interpolates topography into data grid.
    if topomask != None:
        if not verbose:
            print 'Masking topographic features...'
        ezi, _, _ = interpolate.nearest([common.etopo.x, 
            common.etopo.y], common.etopo.z, [Lon, Lat])
        if topomask == 'ocean':
            tmask = (ezi > 0)
        elif topomask == 'land':
            tmask = (ezi < 0)
        #
        tmask = tmask.reshape([1, b, a])
        tmask = tmask.repeat(c, axis=0)
        #
        Z.mask = Z.mask | tmask

    if masked:
        Z.mask = Z.mask | numpy.isnan(Z.data)
        Z.data[Z.mask] = 0
    
    return Lon, Lat, Tm, Z


def save_map(lon, lat, z, fullpath, tm=None, fmt='%.3f', nonan=True, 
    lon180=False):
    """Saves a single map to file.

    PARAMETERS
        lon, lat (array like) :
            Longitude and latitude coordinates.
        z (array like) :
            Data to be saved.
        fullpath:
            Full file path including its name and extension. As the
            numpy.savetxt function is used, if the file name ends with
            the extension .gz, it is automatically saved in compressed
            gzip format.
        tm (float, optional) :
            Time or other relevant information (i.e. mean, minimum) for
            the map.
        fmt (string, optional) :
            Format string for the values saved in the map. Default is a
            floating point number with three digits precision ('%.3f').
        nonan (boolean, optional) :
            If true, slices unnecessary NaN's at the borders.
        lon180 (boolean, optional) :
            If true, saves longitude ranging from -180 to 180 degrees.
            Default value is false.


    RETURNS
        Nothing.

    """
    # Continuous longitudes and latitudes.
    if lon180:
        lon = common.lon180(lon)
        i = numpy.argsort(lon)
        lon = lon[i]
        z = z[:, i]

    # Detects invalid values from mask or NaN's.
    if ((type(z).__name__ == 'MaskedArray') and 
        (type(z.mask).__name__ != 'bool_')):
        mask = ~z.mask
    else:
        mask = ~numpy.isnan(z)

    # If the edges contain only NaN's, then slice them out.
    if nonan:
        selx = pylab.find(mask.any(axis=0))
        sely = pylab.find(mask.any(axis=1))
        if (len(selx) == 0) | (len(sely) == 0):
            warnings.warn('No valid data to save.', Warning)
            return
        else:
            lon = lon[selx]   # lon[selx[0]:selx[-1] + 1]
            lat = lat[sely]   # lat[sely[0]:sely[-1] + 1]
            selx, sely = numpy.meshgrid(selx, sely)
            z = z[sely, selx] # z[sely[0]:sely[-1] + 1, selx[0]:selx[-1] + 1]
            mask = mask[sely, selx]

    b, a = z.shape
    if lon.size != a:
        raise Warning, 'Longitude and data lengths do not match.'
    if lat.size != b:
        raise Warning, 'Latitude and data lengths do not match.'
    
    # Sets mask to NaN where appropriate
    nmask = numpy.ones((b, a))
    nmask[~mask] = numpy.nan

    dat = numpy.zeros((b + 1, a + 1))
    dat[0, 0] = tm
    dat[0, 1:] = lon
    dat[1:, 0] = lat
    if type(z).__name__ == 'MaskedArray':
        dat[1:, 1:] = z.data * nmask
    else:
        dat[1:, 1:] = z * nmask
    numpy.savetxt('%s' % (fullpath), dat, fmt=fmt, delimiter='\t')

def save_dataset(lon, lat, tm, z, path, fname=None, prefix='', fmt='%.3f'):
    """Saves an entire dataset of maps to files.

    Function accepts only three-dimensional data variables, for now.

    PARAMTERS
        lon, lat (array like) :
            Longitude and latitude coordinates.
        tm (floag) :
            Time or other relevant information (i.e. period) to append
            to the upper left cell.
        z (array like) :
            Variable data.
        path (string) :
            Path to the dataset directory.
        fnames (string, array like, optional) :
            Forces the file name of the data. If omitted then the
            default 'xy%s_%d' % (prefix, tm[i]), where i is a counter
            starting at zero.
        prefix (string, optional) :
                Prefix to retain naming conventions such as basin.
        fmt (string, optional) :
            Format string for the values saved in the map. Default is a
            floating point number with three digits precision ('%.3f').

    OUTPUTS
        Saved map files to directory specified in path.

    RETURNS
        Nothing.

    """
    t1 = time()

    c, b, a = z.shape
    if lon.size != a:
        raise Warning, 'Longitude and data lengths do not match.'
    if lat.size != b:
        raise Warning, 'Latitude and data lengths do not match.'
    if tm.size != c:
        raise Warning, 'Time and data lengths do not match.'

    if type(fname).__name__ == 'str':
        fname = ['%s%d' % (fname, i) for i in range(c)]
    elif type(fname).__name__ in ['list', 'tuple', 'ndarray']:
        C = len(fname)
        if c > C:
            for i in range(int(numpy.ceil(float(c) / C))):
                for j in range(C):
                    fname = '%s%d' % (fname[j], i)
    else:
        fname = ['%s_%06d.xy' % (prefix, tm[i]) for i in range(c)]

    # Starts saving the maps to gziped files.
    if c == 1:
        plural = ''
    else:
        plural = 's'
    s = 'Saving %d file%s... ' % (c, plural)
    os.sys.stdout.write(s)
    os.sys.stdout.flush()
    for i in range(c):
        t2 = time()

        f = '%s/%s.gz' % (path, fname[i])
        save_map(lon, lat, z[i, :, :], f, tm[i], fmt)

        os.sys.stdout.write(len(s) * '\b')
        s = 'Saving %d file%s... %s ' % (c, plural, common.profiler(c, i + 1, 
            0, t1, t2),)
        os.sys.stdout.write(s)
        os.sys.stdout.flush()
    #
    os.sys.stdout.write('\n')
