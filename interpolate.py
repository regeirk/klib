# -*- coding: iso-8859-1 -*-
"""Interpolation module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to provide useful and fast interpolation
routines.

AUTHOR
    Sebastian Krieger
    email: naitsabes@regeirk.com

REVISION
    1 (2011-11-11 11:11)

"""

__version__ = '$Revision: 3 $'
# $Source$

__all__ = ['nearest']

import numpy
from scipy.spatial import cKDTree
from common import lon180

def nearest(x, data, xi):
    """Nearest neighbour interpolation.
    
       
    """
    # Starts kd-tree instance
    if type(x).__name__ == 'list':
        xx, yy = numpy.meshgrid(x[0], x[1])
        X = numpy.array([lon180(xx.flatten()), yy.flatten()])
        zz = data.flatten()

    a, b = data.shape
    v, u = numpy.mgrid[0:a, 0:b]
    V, U = v.flatten(), u.flatten()

    k = cKDTree(X.transpose())

    if type(xi).__name__ == 'list':
        xi, yi = numpy.meshgrid(xi[0], xi[1])
        Xi = numpy.array([lon180(xi.flatten()), yi.flatten()]).transpose()
    else:
        Xi = xi

    D, I, UV = [], [], []
    for item in Xi:
        d, i = k.query(item)
        D.append(d)
        I.append(i)
        UV.append((U[i], V[i]))

    zi = zz[I]
    try:
        zi = zi.reshape(xi.shape)
    except:
        pass

    return zi, I, UV
