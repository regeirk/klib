# -*- coding: iso-8859-1 -*-
"""File management module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to assist in manipulating large data
sets of geophysical data.

AUTHOR
    Sebastian Krieger
    email: naitsabes@regeirk.com

REVISION
    1 (2011-01-12 00:16)

"""

__version__ = '$Revision: 1 $'
# $Source$

__all__ = ['load_map', 'load_dataset', 'save_map', 'save_dataset']

import os
import numpy
import pylab
import warnings

from time import time
from string import atof

import common

def load_map(fullpath, ftype='xy', delimiter='\t', masked=False):
    """Loads an idividual data file.

    PARAMETERS
        fullpath (string) :
        ftype (string, optional) :
        delimiter (string, optional) :
        masked (boolean, optional) :

    RETURNS
        lon, lat, tm, z
    """
    dat = numpy.loadtxt('%s' % (fullpath), delimiter=delimiter)
    if ftype == 'xy':
        lon = dat[0, 1:]
        lat = dat[1:, 0]
        tm = dat[0, 0]
    elif ftype == 'xt':
        lon = dat[0, 1:]
        lat = dat[0, 0]
        tm = dat[1:, 0]
    elif ftype == 'ty':
        lon = dat[0, 0]
        lat = dat[1:, 0]
        tm = dat[0, 1:]
    else:
        raise Warning, 'Type \'%s\' not recognised' % (ftype)
    if masked:
        z = numpy.ma.asarray(dat[1:, 1:])
        z.mask = numpy.isnan(z)
        z.data[z.mask] = 0
    else:
        z = dat[1:, 1:]
    return lon, lat, tm, z

def load_dataset(path, pattern='(.*)', ftype='xy', delimiter='\t',
                 var_from_name=False, masked=False, xlim=None, ylim=None):
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
    t0 = 0
    t1 = time()

    s = 'Loading data...'
    os.sys.stdout.write(s)
    os.sys.stdout.flush()

    # Generates list of files and tries to match them to the pattern
    flist = os.listdir(path)
    flist, match = common.reglist(flist, pattern)

    # Loads all the data from the file list
    N = len(flist)
    if N == 0:
        raise Warning, 'No files to be loaded.'
    Lon, Lat, Tm, Z, Sh = [], [], [], [], []
    i = 0
    for n, fname in enumerate(flist):
        t2 = time()

        dat = numpy.loadtxt('%s/%s' % (path, fname), delimiter=delimiter)
        if ftype == 'xy':
            lon = dat[0, 1:]
            lat = dat[1:, 0]
            tm = dat[0, 0]
        elif ftype == 'xt':
            lon = dat[0, 1:]
            lat = dat[0, 0]
            tm = dat[1:, 0]
        elif ftype == 'ty':
            lon = dat[0, 0]
            lat = dat[1:, 0]
            tm = dat[0, 1:]
        z = dat[1:, 1:]

        if var_from_name:
            if (ftype == 'xt') | (ftype == 'ty'):
                var = atof(match[n][-2])    # Gets coordinate out of match ...
                rav = match[n][-1].upper()  # ... and also its hemisphere.
                if (rav == 'S' | rav == 'W'):
                    var *= -1
                if ftype == 'xt':
                    lat = var
                else:
                    lon = var
            elif ftype == 'xy':
                tm = atof(match[n][-1])     # Gets time out of last match.

        Lon.append(numpy.asarray(lon))
        Lat.append(numpy.asarray(lat))
        Tm.append(numpy.asarray(tm))
        Z.append(numpy.asarray(z))
        Sh.append(numpy.asarray(z.shape))

        os.sys.stdout.write(len(s) * '\b')
        s = 'Loading data (%s)... %s ' % (fname, common.profiler(N, n + 1, t0,
            t1, t2),)
        os.sys.stdout.write(s)
        os.sys.stdout.flush()
    #
    os.sys.stdout.write('\n')

    # Reshaping and rearranging the arrays to form an uniform data matrix.
    t1 = time()
    s = 'Reshaping arrays...'
    os.sys.stdout.write(s)
    os.sys.stdout.flush()

    try:
        lon = numpy.unique(numpy.concatenate(Lon))
    except:
        lon = numpy.unique(numpy.asarray(Lon))
    try:
        lat = numpy.unique(numpy.concatenate(Lat))
    except:
        lat = numpy.unique(numpy.asarray(Lat))
    try:
        tm = numpy.unique(numpy.concatenate(Tm))
    except:
        tm = numpy.unique(numpy.asarray(Tm))
    if numpy.isnan(tm).all():
        tm = numpy.array([numpy.nan])
    #elif ftype == 'ty':
    #    raise Warning, 'Loading of temporal-meridional files not implemented yet.'
    #
    dx = numpy.diff(lon).mean()
    dy = numpy.diff(lat).mean()
    dt = numpy.diff(tm).mean()

    # To ensure that the edges are padded with NaN's to avoid distortions when
    # generating maps.
    lon = numpy.concatenate([[lon[0] - dx], lon, [lon[-1] + dx]])
    lat = numpy.concatenate([[lat[0] - dy], lat, [lat[-1] + dy]])

    a, b, c = lon.size, lat.size, tm.size
    if masked:
        z = numpy.ma.empty([c, b, a], dtype=float) * numpy.nan
    else:
        z = numpy.empty([c, b, a], dtype=float) * numpy.nan

    for n in range(N):
        t2 = time()

        if ftype == 'xt':
            i = [pylab.find(lon == x)[0] for x in Lon[n]]
            j = pylab.find(lat == Lat[n])[0]
            z[:, j, i] = Z[n]
        elif ftype == 'xy':
            i = [pylab.find(lon == x)[0] for x in Lon[n]]
            j = [pylab.find(lat == y)[0] for y in Lat[n]]
            if numpy.isnan(Tm[n]) :
                k = pylab.find(numpy.isnan(tm))[0]
            else:
                k = pylab.find(tm == Tm[n])
            i, j = numpy.meshgrid(i, j)
            z[k, j, i] = Z[n]
        elif ftype == 'ty':
            i = pylab.find(lon == Lon[n])[0]
            j = [pylab.find(lat == y)[0] for y in Lat[n]]
            i, j = numpy.meshgrid(i, j)
            z[:, j, i] = Z[n]

        os.sys.stdout.write(len(s) * '\b')
        s = 'Reshaping arrays... %s ' % (common.profiler(N, n + 1, t0, t1, t2))
        os.sys.stdout.write(s)
        os.sys.stdout.flush()
    #
    os.sys.stdout.write('\n')
    
    # Finds the upper and lower zonal and meridional limits to return only the
    # selected ranges.
    if masked and ((xlim != None) or (ylim != None)):
        if xlim != None:
            xsel = pylab.find((lon < min(xlim)) | (lon > max(xlim)))
        else:
            xsel = range(a)
        if ylim != None:
            ysel = pylab.find((lat < min(ylim)) | (lat > max(ylim)))
        else:
            ysel = range(b)
        xsel, ysel = numpy.meshgrid(xsel, ysel)
        z[:, ysel, xsel] = numpy.nan
    else:
        if xlim != None:
            xsel = pylab.find((lon >= min(xlim)) & (lon <= max(xlim)))
        else:
            xsel = range(a)
        if ylim != None:
            ysel = pylab.find((lat >= min(ylim)) & (lat <= max(ylim)))
        else:
            ysel = range(b)
        lon = lon[xsel]
        lat = lat[ysel]
        xsel, ysel = numpy.meshgrid(xsel, ysel)
        z = z[:, ysel, xsel]

    if c == 1:
        z = z[0, :, :]
    if masked:
        z.mask = numpy.isnan(z)
        z.data[z.mask] = 0

    return lon, lat, tm, z

def save_map(lon, lat, z, fullpath, tm=None, fmt='%.3f', nonan=True):
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

    RETURNS
        Nothing.
    """
    # If the edges contain only NaN's, then slice them out.
    if nonan:
        selx = pylab.find(~numpy.isnan(z).all(axis=0))
        sely = pylab.find(~numpy.isnan(z).all(axis=1))
        if (len(selx) == 0) & (len(sely) == 0):
            warnings.warn('No valid data to save.', Warning)
            return
        else:
            lon = lon[selx[0]:selx[-1] + 1]
            lat = lat[sely[0]:sely[-1] + 1]
            z = z[sely[0]:sely[-1] + 1, selx[0]:selx[-1] + 1]

    b, a = z.shape
    if lon.size != a:
        raise Warning, 'Longitude and data lengths do not match.'
    if lat.size != b:
        raise Warning, 'Latitude and data lengths do not match.'

    dat = numpy.zeros((b + 1, a + 1))
    dat[0, 0] = tm
    dat[0, 1:] = lon
    dat[1:, 0] = lat
    dat[1:, 1:] = z
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
        fname = ['xy%s_%06d' % (prefix, tm[i]) for i in range(c)]

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

        s = '%s/%s.gz' % (path, fname[i])
        save_map(lon, lat, z[i, :, :], s, tm[i], fmt)

        os.sys.stdout.write(len(s) * '\b')
        s = 'Saving %d file%s... %s ' % (c, plural, common.profiler(c, i + 1, 
            0, t1, t2),)
        os.sys.stdout.write(s)
        os.sys.stdout.flush()
    #
    os.sys.stdout.write('\n')
