import numpy
import pylab

import klib

# Parameters
e, g = 1, 1
u0, v0 = 10, 0


# Regular grid coordineates, velocity field and stream function
x = numpy.arange(0, 21)
y = numpy.arange(0, 11)
xx, yy = numpy.meshgrid(x, y)
u = e * xx - g * yy + u0
v = g * xx - e * yy + v0
phi, psi = klib.dynamics.flowfun(u, v)

# The plots!
pylab.close('all')
pylab.ion()
pylab.figure(figsize=[16, 8])
pylab.contour(x, y, psi, 20, colors='k', linestyles='-', linewidths=2.0)
pylab.contour(x, y, phi, 20, colors='k', linestyles='-.', linewidths=1.0)
pylab.quiver(x, y, u, v, angles='xy', scale_units='xy')

ax = pylab.axes()
ax.set_aspect(1.)

pylab.draw()
