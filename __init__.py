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
params = {#'figure.figsize': [8, 11],
          'figure.dpi': 90,
          'font.cursive': 'cursive',
          'font.family': 'serif',
          'font.fantasy': ['fantasy'],
          'font.monospace': ['monospace'],
          'font.sans-serif': ['sans-serif'],
          'font.serif': ['Times'],
          'font.size': 18,
          'font.stretch': 'ultra-condensed',
          'text.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'xtick.major.size': 8,
          'xtick.minor.size': 4,
          #'xtick.direction': 'in',
          'ytick.labelsize': fontsize,
          'ytick.major.size': 8,
          'ytick.minor.size': 4,
          #'ytick.direction': 'in',
          'axes.titlesize': fontsize,
          'text.usetex': True,
          'text.latex.unicode': True,
          'text.latex.preamble': [r'\usepackage{times}'],
          'timezone': 'UTC',
          'axes.unicode_minus': True
         }

rcParams.update(params)


try:
    reload(cm)
    reload(gis)
    reload(file)
    reload(stats)
    reload(common)
    reload(dynamics)
    reload(graphics)
    reload(interpolate)
except:
    import cm
    import gis
    import gis as mapping
    import file
    import file as filemngmnt # for backwards compatibility
    import stats
    import common
    import dynamics
    import graphics
    import interpolate

#__all__ = ['cwt', 'icwt', 'significance', 'Morlet', 'Paul', 'DOG',
#           'Mexican_hat']
