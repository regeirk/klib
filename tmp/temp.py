import numpy
import pylab

try:
    reload(klib.interpolate)
except:
    import klib

p = '../'
ez = numpy.loadtxt('%s/etopo20data.gz' % p)
ex = numpy.loadtxt('%s/etopo20lons.gz' % p)
ey = numpy.loadtxt('%s/etopo20lats.gz' % p)

lon = numpy.arange(-180., 180., 1.)
lat = numpy.arange(-90., 91., 1.)

zi = klib.interpolate.nearest([ex, ey], ez, [lon, lat])


pylab.ion()
crange = numpy.arange(-10000, 8100, 100.)
pylab.subplot(2, 1, 1)
pylab.contourf(ex, ey, ez, crange)
pylab.subplot(2, 1, 2)
pylab.contourf(lon, lat, zi)
