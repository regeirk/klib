# -*- coding: iso-8859-1 -*-
"""
Sebastian's collection of scientific libraries.

DISCLAIMER
    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
    email: solutions@nublia.com

REVISION
    1 (2011-08-26 17:12 -0300)

REFERENCES
"""
from __future__ import division

__version__ = '$Revision: 1 $'
# $Source$

#try:
#    from matplotlib import use
#    use('Agg')
#except:
#    pass

from matplotlib import rcParams

fontsize = 'medium'
params = {'figure.figsize': [8, 11],
          'font.family': 'serif',
          'font.sans-serif': ['Helvetica'],
          'font.size': 18,
          'font.stretch': 'ultra-condensed',
          'text.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'text.usetex': True,
          'text.latex.unicode': True,
          'timezone': 'UTC'
         }
rcParams.update(params)


try:
    reload(cm)
    reload(common)
    reload(file)
    reload(stats)
    reload(ocean)
    reload(graphics)
    reload(interpolate)
except:
    import cm
    import common
    import file
    import file as filemngmnt # for backwards compatibility
    import stats
    import dynamics
    import graphics
    import interpolate

#__all__ = ['cwt', 'icwt', 'significance', 'Morlet', 'Paul', 'DOG',
#           'Mexican_hat']
