# -*- coding: utf-8 -*-
"""Dynamis module.

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

__all__ = ['f', 'flowfun', 'relation_dispersion']

import numpy
import gsw

from common import simpson


class constants:
    """Important geophysical constants in SI units.
    
    REFERENCES
        Moritz, H. Geodetic reference system 1980. Journal of Geodesy, 
        2000, 74, 128-162

    """
    # Earth's rotation rate, according to Moritz (2000), in s^{-1}
    omega = 7292115e-11
    #omega = 2 * pi / (3600 * 23 + 56 * 60)

    # Mean surface gravity [m / s**2], according to Moritz (2000)
    g = 9.797644656 # 9.81

    # Earth's mean radius [m], according to Moritz (2000)
    a = 6371008.7714
    b = 111177.5

    # Tidal frequencies [s**(-1)] in order of amplitude as in Stewart,
    # R. H. Introduction to physical oceanography; Texas A & M 
    # University, 2008, 345, available at http://oceanworld.tamu.edu/
    # resources/ocng_textbook/chapter17/chapter17_04.htm
    M2 = 2 * numpy.pi / (3600 * 12.4206);
    K1 = 2 * numpy.pi / (3600 * 23.9344);
    S2 = 2 * numpy.pi / (3600 * 12);
    O1 = 2 * numpy.pi / (3600 * 25.8194);


def f(lat, lat0=None, central=True, returns='full'):
    r"""Calculates the Coriolis parameter f using the beta plane 
    approximation.
    
    PARAMETERS
        lat (array like) :
            Latitude in degrees.
        lat0 (array like) :
            Central latitudes in degrees.
        returns (string, optional) :
            If set to 'full' returns f, f0 and \beta.
    
    RETURNS
        f (array like) :
            The Coriolis parameter f = f_0 + \beta y
        f0 (array like) :
            
        beta (array like) :
            The beta parameter.

        Note that the unit of f and f0 is s^{-1} and that the unit
        of \beta is (m s)^{-1}.

    """
    # Checks input arrays. If central latitudes are not set, creates an array
    # with central latitudes every 10 degrees. If central latitudes are 
    # equally spaced, determines spatial step in degrees.
    lat = numpy.asarray(lat)
    
    if central:
        if (lat0 == None) | (type(lat0) in [int, float]):
            if lat0 == None:
                dy = 10.
            else:
                dy = lat0
            ymin = numpy.floor(lat.min() / dy) * dy
            ymax = numpy.ceil(lat.max() / dy) * dy
            y0 = numpy.arange(ymin, ymax+dy, dy)
        else:
            dy = lat[1:] - lat[0:-1]
            if (dy == dy[0]).all():
                dy = dy[0]
            else:
                dy = None
        
        # Determine to which central latitude y0 every latitude y belongs
        # to. The first method (fast way) assumes regular spaced y0's and the
        # second method (slow way) is not implemented yet.
        if dy != None:
            Lat = numpy.round(lat / dy) * dy
        else:
            # TODO: Implement slow way!
            raise Warning, 'Slow way not implemented yet.'
        
        # Calculate the distance from the equator to the latitudes in meters.
        d = gsw.distance(0, lat).flatten()
        d0 = gsw.distance(0, [0, lat[0]])[0, 0]
        y = numpy.concatenate([[0], d.cumsum()]) - d0
        #
        d = gsw.distance(0, Lat).flatten()
        d0 = gsw.distance(0, [0, Lat[0]])[0, 0]
        Y = numpy.concatenate([[0], d.cumsum()]) - d0
    else:
        Lat = lat
        
    # The Coriolis parameter calculated at the central latitudes
    K = constants()
    f0 = 2. * K.omega * numpy.sin(numpy.deg2rad(Lat))
    b = 2. * K.omega / K.a * numpy.cos(numpy.deg2rad(Lat))
    if central:
        f = f0 + b * (y - Y)
    
    if returns == 'full':
        if central:
            return f, f0, b
        else:
            return f0, b
    else:
        return f


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


def relation_dispersion_Rossby(k, l, beta, Ro, U=0):
    """Relation dispersion of Rossby waves.

    PARAMETERS

    RETURNS
    
    """
    return - beta * k / (k**2 + l**2 + Ro**(-2)) + U * k
