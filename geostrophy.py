# -*- coding: iso-8859-15 -*-
"""Set of functions using the geostrophic balance in the atmosphere and
oceans.

DISCLAIMER
    Part of this module is based upon Matlab scripts provided by Robert
    Scott available at (TODO: include URL).
    
    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
w    email: sebastian.krieger@usp.br

REVISION
    1 (2012-12-15 20:55 -0300 DST)

"""
from __future__ import division
__version__ = '$Revision: 1 $'
# $Source$

__all__ = ['constants', 'metergrid', 'stencil_coefficients', 'curl',
    'gradient', 'ssh2vel']

from numpy import (arange, asarray, concatenate, ceil, cos, empty, exp, floor,
    isnan, linalg, meshgrid, ma, nan, ones, pi, round, sin, zeros)
from scipy import factorial
import gsw


class constants:
    """Important geophysical constants in SI units.
    
    REFERENCES
        Moritz, H. Geodetic reference system 1980. Journal of Geodesy, 
        2000, 74, 128-162

    """
    # Earth's rotation rate, according to Moritz (2000)
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
    M2 = 2 * pi / (3600 * 12.4206);
    K1 = 2 * pi / (3600 * 23.9344);
    S2 = 2 * pi / (3600 * 12);
    O1 = 2 * pi / (3600 * 25.8194);


class grid:
    """Common grid for sea surface height maps.
    
    This class was created to reduce redundant calculations and spare
    computational processing time. During class initialization, the
    spatial grid is determined and specific coefficients for this grid
    are calculated.
    
    """
    
    lon, lat = None, None
    x, y = None, None
    _lon, _y = None, None
    n = None
    nleft = None
    nright = None
    p = None
    units = None
    cyclic = None
    psi = None
    masked = None
    
    nx, ny = None, None
    
    coeffs_x = None
    coeffs_y = None
    
    G = None
    C = None
    
    
    def __init__(self, lon, lat, psi=None, n=3, p=1, cyclic=True, units='m'):
        """Initializes the class.
        
        PARAMETERS
            lon, lat (array like) :
                Longitude and latitude arrays in degrees.
            psi (array like) :
                Scalar or vector field maps as a function of longitude
                and latitude -- psi[lat, lon].
            n (integer, optional) :
                Length of the stencil used for centered differentials.
                The length has to be odd numbered. Default is n=3.
            p (integer, optional) :
                Order of the derivative to be calculated. Default is to
                calculate the first derivative (p=1). 2*n-p+1 gives the
                order of the approximation.
            cyclic (boolean, optional) :
                Sets whether `A` is to be considered zonally periodic.
                Default is true.
        
        RETURNS
            Nothing.
        
        """
        # Makes shure the length of the stencil is odd numbered.
        n += (n % 2) - 1
        self.n = n
        # Calculate left and right stencils.
        self.nleft = (self.n - 1) / 2
        self.nright = (self.n - 1) / 2 + 1

        self.cyclic = cyclic
        self.set_grid(lon, lat, p=p, units=units)
        if psi != None:
            self.set_psi(psi)
        
        return
    
    
    def __str__(self):
        if self.cyclic:
            print self.psi[:, self.nleft:-self.nright]
        else:
            print self.psi
        return
    
    
    def set_psi(self, psi, lon=None, lat=None):
        if (lon != None) | (lat != None):
            self.set_grid(lon, lat)
            
        # Sets masked values to NaN
        if type(psi) == ma.core.MaskedArray:
            psi[psi.mask] = nan
            self.masked = True
        else:
            self.masked = False
        
        # Makes `A` cyclic, if appropriate, by padding data on boundaries
        # according to the stencile width.
        if self.cyclic:
           psi = concatenate((psi[:, -self.nleft:], psi, 
            psi[:, :self.nright]), axis=1)
        
        # Verifies if psi matches the longitude and latitude shape
        b, a = self.lat.size, self.lon.size
        if (b, a) != psi.shape:
            raise Warning, ('Longitude and latitude grid dimensions do not'
                ' match sea surface height dimensions.')
        
        if self.masked:
            self.psi = ma.array(psi, mask=isnan(psi))
        else:
            self.psi = psi
        
        return
    
    
    def set_grid(self, lon=None, lat=None, p=None, units=None):
        if lon != None:
            self._lon = lon
            if self.cyclic:
                lon = concatenate((lon[-self.nleft:]-360., lon, 
                    lon[:self.nright]+360.))
                self.nx = lon.size - self.n
            else:
                self.nx = lon.size
            self.lon = lon
        
        if lat != None:
            self.lat = lat
            self.ny = lat.size
        
        if p != None:
            self.p = p
        elif self.p == None:
            self.p = 1
        
        if units != None:
            self.units = units
        elif self.units == None:
            self.units = 'm'
        
        # Convert grid from degrees latitude and longitude to SI units
        Lon, Lat = meshgrid(self.lon, self.lat)
        if self.units == 'm':
            self.x, self.y = self.metergrid(Lon, Lat)
        else:
            self.x, self.y = Lon, Lat
        if self.cyclic:
            self._y = self.y[:, self.nleft:-self.nright]
        else:
            self._y = self.y
        
        # Calculate the stencil coefficients (cn) for the n-sized stencil
        # on the current grid. As a simplification to reduce computation time,
        # it assumes that the zonal distance between grid points is the same
        # at each latitude. It also assumes that the meridional distances are 
        # the same at each longitude.for each latitudeIf possible, simplifies 
        # the problem to reduce. It starts calculating the zonal coefficents 
        # and then the meridional coefficients.
        b, a = Lon.shape
        self.coeffs_x = empty((b, a, self.n))
        for i in range(b):
            A = self.stencil_coefficients(self.x[i, :self.n], p=self.p)
            self.coeffs_x[i, :self.nleft, :] = A[None, :self.nleft, :]
            self.coeffs_x[i, self.nleft:-self.nright, :] = A[
                self.nleft:self.nleft+1, :].repeat(a-self.n, axis=0)
            self.coeffs_x[i, -self.nright:, :] = A[-self.nright:, :]
        
        self.coeffs_y = empty((b, a, self.n))
        B = self.stencil_coefficients(self.y[:, 0], p=self.p)
        self.coeffs_y[:, :, :] = B[:, None, :].repeat(a, axis=1)
        
        return


    def metergrid(self, lon, lat, unit='m'):
        """Converts zonal and meridional coordinates from degrees 
        latitude and longitude to another reference unit.
        
        PARAMETERS
            lon, lat (array like) :
                Longitude and latitude as bi-dimensional gridded 
                arrays.
            unit (string, optional) :
                Unit to which the coordinates will be converted.
        
        RETURNS
            x, y (array like) :
                New coordinates
        
        """
        if lon.shape != lat.shape:
            raise Warning, ('Longitude and latitude grid dimensions do not'
                ' match.')
        b, a = lon.shape
        x, y = lon * nan, lat * nan
        
        x = gsw.distance(lon, lat)
        x = concatenate([zeros((b, 1)), x.cumsum(axis=1)], axis=1)
        
        y = gsw.distance(lon.transpose(), lat.transpose()).transpose()
        y = (concatenate([zeros((1, a)), y.cumsum(axis=0)], axis=0) -
            0.5 * y.sum(axis=0))
        
        return x, y


    def stencil_coefficients(self, x, p=1):
        """Calculates the coefficients needed for the gradient.
        
        PARAMETERS
            x (array like) :
                Coordinate array such as longitude or latitude in 
                arbitrary units, e.g. degrees, meters, nautical miles.
            p (integer, optional) :
                Order of the derivative to be calculated. Default is to
                calculate the first derivative (p=1). 2*n-p+1 gives the
                order of the approximation.
        
        RETURNS
            Coefficients (cn) needed for the linear combination of `n` 
            points to get the first derivative according to Arbic et 
            al. (2012) equations (20) and (22). At the boundaries 
            forward and backward differences approximations are 
            calculated.
        
        REFERENCES
            Cushman-Roisin, B. & Beckers, J.-M. Introduction to 
            geophysical fluid dynamics: Physical and numerical aspects 
            Academic Press, 2011, 101, 828
            
            Arbic, Brian B. Scott, R. B.; Chelton, D. B.; Richman, J. 
            G. & Shriver, J. F. Effects of stencil width on surface 
            ocean geostrophic velocity and vorticity estimation from 
            gridded satellite altimeter data. Journal of Geophysical 
            Research, 2012, 117, C03029
        
        """
        N = x.size
        coeffs = zeros((N, self.n))
        JJ = arange(N)
        
        for jj in JJ:
            # Constructs matrices according to Cushman-Roisin & Beckers (2011)
            # equations (1.25) and adapted for variable grids as in Arbic et 
            # al. (2012), equations (20), (22). The linear system of equations
            # is solved afterwards.
            A = zeros((self.n, self.n))
            if jj < self.nleft:
                start = self.nleft - jj
            else:
                start = 0
            if jj > N - self.nright:
                stop = jj - (N - self.nright)
            else:
                stop = 0
            A[0, start:self.n+stop] = 1
            for i in range(1, self.n):
                A[i, start:self.n-stop] = (
                    x[jj-self.nleft+start:jj+self.nright-stop] - 
                    x[jj])**i
            B = zeros((self.n, 1))
            # This tells where the p-th derivative is calculated
            B[p] = factorial(p)
            C = linalg.solve(A[:self.n-start-stop, start:self.n-stop], 
                B[:self.n-(start+stop), :])
            coeffs[jj, start:self.n-stop] = C.flatten()
        
        return coeffs


    def gradient(self, p=None):
        """Returns the gradient of a bi-dimensional array according to
        a given stencil width.
        
        Calculates the first derivative of `A` with respect x and y using
        3, 5, or 7 point stencils. As default, it assumes that `A` is a 
        mapped field on the globe, so it is periodic in the x-direction. It
        exploits this peridicity so that there are not missing data points 
        at the boundaries.
        
        The gradient is computed using central differences in the interior
        and first differences at the boundaries. The returned gradient hence has
        the same shape as the sea surface height array.
        
        PARAMETERS
            p (integer, optional) :
                Order of the derivative to be calculated. Default is to
                calculate the first derivative (p=1). 2*n-p+1 gives the
                order of the approximation.
        
        RETURNS
            G (ndarray) :
                N arrays of the same shape as `A` giving the derivative of
                `A` with respect to each dimension.
        
        """
        # Changes the order of the derivative if necessary.
        if p != None:
            if self.p != p:
                self.set_grid(p=p, units=self.units)
        
        # Initializes some useful variables
        b, a = self.psi.shape
        Gx = zeros((b, a))
        Gy = zeros((b, a))
        
        # Calculate the derivatives!
        for i in arange(self.n) - self.nleft:
            if i < 0:
                u, v = -i, a
                s, t = -i, b
            else:
                u, v = 0, a - i
                s, t = 0, b - i

            Gx[:, u:v] += (self.coeffs_x[:, u:v, i+self.nleft] * 
                self.psi[:, u+i:v+i])
            Gy[s:t, :] += (self.coeffs_y[s:t, :, i+self.nleft] *
                self.psi[s+i:t+i, :])
        
        if self.cyclic:
            self.G =  (Gx[:, self.nleft:-self.nright] + 
                1j * Gy[:, self.nleft:-self.nright])
        else:
            self.G =  Gx + 1j * Gy

        return self.G


    def curl(self, p=None):
        """Returns the curl of a bi-dimensional vector array according 
        to a given stencil width.
        
        The input vector array should be in complex notation, such that
        A = u + j * v, where j = (-1)**0.5.
        
        Calculates the first derivative of `A` with respect x and y using
        p stencils. As default, it assumes that `A` is a mapped vector 
        field on the globe, so it is periodic in the x-direction. It
        exploits this peridicity so that there are no missing data points
        at the boundaries.
        
        The curl is computed using central differences in the interior
        and first differences at the boundaries. The returned gradient
        hence has the same shape as the input data array.
        
        PARAMETERS
            p (integer, optional) :
                Order of the derivative to be calculated. Default is to
                calculate the first derivative (p=1). 2*n-p+1 gives the
                order of the approximation.
        
        RETURNS
            C (ndarray) :
                Arrays of the same shape as `A` giving the curl of `A`.
        
        """
        # Changes the order of the derivative if necessary.
        if p != None:
            if self.p != p:
                self.set_grid(p=p, units=self.units)
        
        # Initializes some useful variables
        b, a = self.psi.shape
        Cx = zeros((b, a))
        Cy = zeros((b, a))
        
        # Calculate the derivatives!
        for i in arange(self.n) - self.nleft:
            if i < 0:
                u, v = -i, a
                s, t = -i, b
            else:
                u, v = 0, a - i
                s, t = 0, b - i

            # The zonal derivative of v
            Cx[:, u:v] += (self.coeffs_x[:, u:v, i+self.nleft] * 
                self.psi[:, u+i:v+i].imag)
            # The meridional derivative of u
            Cy[s:t, :] += (self.coeffs_y[s:t, :, i+self.nleft] *
                self.psi[s+i:t+i, :].real)
        
        if self.masked:
            Cx = ma.array(Cx, mask=self.psi.mask)
            Cy = ma.array(Cy, mask=self.psi.mask)
        
        if self.cyclic:
            self.C =  (Cx[:, self.nleft:-self.nright] - 
                Cy[:, self.nleft:-self.nright])
        else:
            self.C =  Cx - Cy

        return self.C


def f(lat, lat0=None, returns='full'):
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

    """
    # Checks input arrays. If central latitudes are not set, creates an array
    # with central latitudes every 10 degrees. If central latitudes are 
    # equally spaced, determines spatial step in degrees.
    lat = asarray(lat)
    if (lat0 == None) | (type(lat0) in [int, float]):
        if lat0 == None:
            dy = 10.
        else:
            dy = lat0
        ymin = floor(lat.min() / dy) * dy
        ymax = ceil(lat.max() / dy) * dy
        y0 = arange(ymin, ymax+dy, dy)
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
        Lat = round(lat / dy) * dy
    else:
        # TODO: Implement slow way!
        raise Warning, 'Slow way not implemented yet.'
    
    # Calculate the distance from the equator to the latitudes in meters.
    d = gsw.distance(0, lat).flatten()
    d0 = gsw.distance(0, [0, lat[0]])[0, 0]
    y = concatenate([[0], d.cumsum()]) - d0
    #
    d = gsw.distance(0, Lat).flatten()
    d0 = gsw.distance(0, [0, Lat[0]])[0, 0]
    Y = concatenate([[0], d.cumsum()]) - d0
    
    # The Coriolis parameter calculated at the central latitudes
    K = constants()
    f0 = 2. * K.omega * sin(Lat / 180. * pi)
    b = 2. * K.omega / K.a * cos(Lat / 180. * pi)
    f = f0 + b * (y - Y)
    
    if returns == 'full':
        return f, f0, b
    else:
        return f


def ssh2vel(psi, ys=2.2):
    """Calculates the geostrophic currents from sea surface height 
    maps.
    
    PARAMETERS
        psi (grid) :
            Sea surface height field grid.
        ys (float, optional) :
            Beta plane length scale in degrees. Default is 2.2. If 
            ys<0, then the equatorial region is not considered.
    
    RETURNS
        Ug (array like) :
            Geostrophic current in complex notation form, i.e.,
            Ug = ug + i*vg, where ug and vg are the zonal and 
            meridional componentsis of geostrophic velocity.
            
    REFERENCES
        Lagerloef, G. S. E. et al.; Tropical Pacific near-surface 
        currents estimated from altimeter, wind and drifter data; 
        Journal of Geophysical Research, 1999, 104, 23313-23326
    
        Arbic, Brian B. Scott, R. B.; Chelton, D. B.; Richman, J. G. &
        Shriver, J. F. Effects of stencil width on surface ocean
        geostrophic velocity and vorticity estimation from gridded
        satellite altimeter data Journal of Geophysical Research, 2012,
        117, C03029

    """
    # Checks lon, lat and psi and makes sure that the first derivative 
    # will be calculated
    lon, lat = psi._lon, psi.lat
    b, a = psi.psi.shape
    Lon, Lat = meshgrid(lon, lat)
    
    # Loads geophysical constants and calculates the Coriolis 
    # parameters f and beta according to the latitudes.
    K = constants()
    f = 2. * K.omega * sin(Lat / 180. * pi)
    b = 2. * K.omega / K.a * cos(Lat / 180. * pi)
    
    # Geostrophic velocity Ug. The calculation is performed using as in
    # Lagerloef et al. (1999). Let
    #
    #   \mathbf{Z} = \frac{\partial\Psi}{\partial x} + 
    #     i\frac{\partial\Psi}{\partial y}
    #
    # and let us decompose the geostrophic velocity in f-plane and beta-
    # plane components, such that
    #
    #   \mathbf{U_g} = W_b \mathbf{U_b} + W_f \mathbf{U_f}
    #
    # where
    #
    #   \mathbf{U_b} = 
    #       \frac{i g}{\beta y} \frac{\partial \mathbf{Z}}{\partial y} (eq. 6*)
    #   \mathbf{U_f} = \frac{i g}{f} \mathbf{Z}                        (eq. 7)
    #
    # are the geostrophic balance for each approximation. The weight
    # functions are
    #
    #   W_b = \exp \left[-\left(\theta / \theta_s \right)^2\right]
    #   W_f = 1-W_b
    #
    # as a function of latitude $\theta$ and length scale 
    # $\theta_s=2.2^{\circ}$
    Z2 = psi.gradient(p=2)
    Z1 = psi.gradient(p=1)
    
    psi2 = grid(psi._lon, psi.lat, psi=Z1.real, n=psi.n, cyclic=psi.cyclic)
    ZXY = psi2.gradient(p=1).imag
    
    #Zn = Z1 * 0.
    #Zn[1:-1, :] = (Z1[2:, :] - Z1[:-2, :]) / (psi._y[2:, :] - psi._y[:-2, :])
    
    Wb = exp(-(Lat / ys) ** 2.)
    #Wb[abs(Lat)<=5] = 1
    #Wb[abs(Lat)>5] = 0
    
    Wf = 1. - Wb
    
    if ys < 0:
        Ub = 0
    else:
        Ub = 1j * K.g / b * (ZXY + 1j * Z2.imag)
        #Ub = 1j * K.g / b * (Z2 - Z2.real)
        #Ub = 1j * K.g / b * Zn
        #Ub = 1j * K.g / (b * psi._y) * Z1
    Uf = 1j * K.g / f * Z1
    
    Ug = Wb * Ub + Wf * Uf
    if type(psi.psi) == ma.core.MaskedArray:
        Ug = ma.array(Ug, mask=isnan(Ug))
    
    return Ug


def ssh2vort(psi):
    """Calculates the geostrophic vorticity from sea surface height
    maps.
    
    PARAMETERS
        psi (grid) :
            Sea surface height field grid.
    
    RETURNS
        V (array like) :
            Geostrophic vorticity.
            
    REFERENCES
        Arbic, Brian B. Scott, R. B.; Chelton, D. B.; Richman, J. G. &
        Shriver, J. F. Effects of stencil width on surface ocean
        geostrophic velocity and vorticity estimation from gridded
        satellite altimeter data Journal of Geophysical Research, 2012,
        117, C03029

    """
    # Checks lon, lat and psi
    lon, lat = psi._lon, psi.lat
    b, a = psi.psi.shape
    Lon, Lat = meshgrid(lon, lat)
    
    # Loads geophysical constants and calculates the Coriolis 
    # parameters f and beta according to the latitudes.
    K = constants()
    f = 2. * K.omega * sin(Lat / 180. * pi)
    b = 2. * K.omega / K.a * cos(Lat / 180. * pi)
    
    # Geostrophic vorticity V. The calculation is performed as mentioned in
    # Arbic et al. (2012) using second and first derivatives of sea surface 
    # height.
    Z2 = psi.gradient(p=2)
    Z1 = psi.gradient(p=1)
    
    Vf = K.g / f * (Z2.real + Z2.imag)
    Vb = - K.g * b / f**2 * (Z2 - Z2.real)
    
    Vg = Vf + Vb
    if type(psi.psi) == ma.core.MaskedArray:
        Vg = ma.array(Vg, mask=isnan(Vg))
    
    return Vg
