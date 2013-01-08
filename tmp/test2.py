import os
import numpy
import pylab

import klib

from matplotlib import dates
from scipy.io import netcdf_file as netcdf
from scikits.vectorplot import lic_internal

# Parameters
dpath = '/home/sebastian/academia/data/ecco2'
dsets = ['UVEL', 'VVEL', 'SSH']
pattern = '^(.*).(1440x720(?:x50)?).(.*).nc'

# Generates list of files based on velocities dataset and tries to match them
# to the pattern
flist = os.listdir('%s/%s.nc' % (dpath, dsets[0]))
Flist, match = klib.common.reglist(flist, pattern)
N = len(flist)

# Processes first file!
n = 0

# 1. Zonal velocity
f = '%s/%s.nc/%s.%s.%s.nc' % (dpath, dsets[0], dsets[0], '1440x720x50', 
    match[n][2])
data = netcdf(f, 'r')
u = numpy.ma.asarray(data.variables['UVEL'][0, 0, :, :])
u.mask = (u.data <= data.variables['UVEL'].missing_value)
setattr(u, 'units', data.variables['UVEL'].units)
#
T0 = dates.datestr2num(data.variables['TIME'].time_origin)
t = data.variables['TIME'].data[0] + T0
lon = data.variables['LONGITUDE_T'][:]
lat = data.variables['LATITUDE_T'][:]
z = -data.variables['DEPTH_T'][:]
data.close()

# 2. Meridional velocity
f = '%s/%s.nc/%s.%s.%s.nc' % (dpath, dsets[1], dsets[1], '1440x720x50', 
    match[n][2])
data = netcdf(f, 'r')
v = numpy.ma.asarray(data.variables['VVEL'][0, 0, :, :])
v.mask = (v.data <= data.variables['VVEL'].missing_value)
setattr(v, 'units', data.variables['VVEL'].units)
data.close()

# 3. Sea surface height
f = '%s/%s.nc/%s.%s.%s.nc' % (dpath, dsets[2], dsets[2], '1440x720', 
    match[n][2])
data = netcdf(f, 'r')
ssh = numpy.ma.asarray(data.variables['SSH'][0, :, :])
ssh.mask = (ssh.data <= data.variables['SSH'].missing_value)
setattr(ssh, 'units', data.variables['SSH'].units)
data.close()
u.mask = ssh.mask
v.mask = ssh.mask

# 4. Selecting specific data range
xlim = [-80.875, 19.875]                         # Atlantic Ocean
lon180 = klib.common.lon180(lon)
les = pylab.find((lon180 >= min(xlim)) & (lon180 <= max(xlim)))
idx = lon180[les].argsort()                      # Makes sure lon is ascending
sel = les[idx]
X = lon180[sel]
Y = lat
U = u[:, sel].astype(numpy.float32)
V = v[:, sel].astype(numpy.float32)
SSH = ssh[:, sel]

# 5. Data visualization
kernellen=31
texture = numpy.random.rand(SSH.shape[0], SSH.shape[1]).astype(numpy.float32)
kernel = numpy.sin(numpy.arange(kernellen) * numpy.pi / 
    kernellen).astype(numpy.float32)
image = lic_internal.line_integral_convolution(U, V, texture, kernel)


