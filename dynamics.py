# -*- coding: iso-8859-1 -*-
"""Ocean module.

This is part of the kLib Python library for scientific data analysis.
The purpouse of this module is to give a set of functions relevant to the
oceanographic community.

AUTHOR
    Sebastian Krieger
    email: solutions@nublia.com

REVISION
    1 (2012-08-11 11:22)

"""
from __future__ import division

__version__ = '$Revision: 1 $'
# $Source$

__all__ = ['flowfun']

import numpy

from common import simpson

def flowfun(u, v=None, flag=''):
    """Calculates the potential phi and the stream function psi of a
    two-dimensional flow defined by the velocity components u and v, so
    that
    
            d(phi)   d(psi)       d(phi)   d(psi)
        u = ------ - ------;  v = ------ + ------
              dx       dy           dy       dx

    PARAMETERS
        u, v (array like) :
            Zonal and meridional velocity field vectors. 'v' can be 
            ommited if the velocity vector field U is given in complex
            form, such that U = u + i*v.
        flag (string, optional) :
            If only the stream function is needed, the '-', 'psi' or 
            'streamfunction' flag should be used. For the velocity
            potential, use '+', 'phi' or 'potential'.
            
    RETURNS
    
    EXAMPLES
        phi, psi = flowfun(u, v)
        psi = flowfun(u + i*v, '-')
    
    REFERENCES
    
    Based upon http://www-pord.ucsd.edu/~matlab/stream.htm
    
    """
    # Checks input arguments
    u = numpy.asarray(u)
    if v == None:
        v = u.imag
        u = u.real
    if u.shape != v.shape:
        raise Exception, 'Error: matrices U and V must be of equal size'
    isphi, ispsi = True, True
    if flag in ['-', 'psi', 'streamfunction']:
        isphi = False
    if flag in ['+', 'phi', 'potential']:
        ispsi = False
    
    a, b = u.shape
    
    # Now, the main computations. Integrates the velocity fields to get the
    # velocity potential and stream function using Simpson rule summation
    
    # The velocity potential (phi), non-rotating part
    if isphi:
        cx = simpson(u[0, :])     # Computes the x-integration constant
        cy = simpson(v[:, 0])     # Computes the y-integration constant
        phi = simpson(v) + cx * numpy.ones((a, 1))
        phi = (phi + simpson(u.transpose()).transpose() + 
            (cy * numpy.ones((b, 1))).transpose()) / 2
    
    # Compute streamfunction (psi), solenoidal part
    if ispsi:
        cx = simpson(v[0, :])      # Computes the x-integration constant
        cy = simpson(u[:, 0])      # Computes the y-integration constant
        psi = -simpson(u) + cx * numpy.ones((a, 1))
        psi = (psi + simpson(v.transpose()).transpose() - 
            (cy * numpy.ones((b, 1))).transpose()) / 2
    
    if isphi & ispsi:
        return (phi, psi)
    elif isphi:
        return phi
    elif ispsi:
        return psi
    else:
        return None
