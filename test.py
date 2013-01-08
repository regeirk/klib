import numpy
import pylab
try:
    reload(klib)
except:
    import klib

e, g, n = 1, 1, 1
u0, v0 = 10, 0
# Regular grid
x, y = numpy.meshgrid(numpy.arange(0, 11), numpy.arange(0, 11))

u = e*x - g*y + u0                     # Linear velocity field
v = g*x - e*y + v0
phi, psi = klib.ocean.flowfun(u, v);   # Here comes the potential and streamfun

pylab.close('all')
pylab.ion()
pylab.figure(figsize=[8, 8])
pylab.contour(x, y, phi, 20, colors='k', linestyles='--', linewidth=1.0)
pylab.contour(x, y, psi, 20, colors='k', linestyles='-', linewidth=2.0)
pylab.quiver(x, y, u, v, angles='uv', scale_units='xy', scale=10)
