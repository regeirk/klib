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

from os import environ

try:
    display = environ['DISPLAY'][-2:]
except:
    display = None
#
if display != ':0':
    try:
        from matplotlib import use
        use('Agg')
    except:
        print 'Warning: could not use Agg backend.'
        pass

from matplotlib import rcParams

fontsize = 'medium'
params = {
    'interactive'  : True,
    'toolbar': 'toolbar2',
    'timezone': 'UTC',
    'lines.linewidth': 1.0, # line width in points
    'font.family': 'serif',
    'font.size': 16.0,
    'font.cursive': 'cursive',
    'font.fantasy': ['fantasy'],
    'font.monospace': ['monospace'],
    'font.sans-serif': ['sans-serif'],
    'font.serif': ['Times'],
    'font.stretch': 'ultra-condensed',
    'text.usetex': True,
    'text.latex.unicode' : True,
    'text.latex.preamble': [r'\usepackage{times}'],
    'text.fontsize': fontsize,
    'text.color': '#555555',
    'axes.facecolor': '#EEEEEE', # axes background color
    'axes.edgecolor': '#BCBCBC', # axes edge color
    'axes.linewidth': 1, # edge linewidth
    'axes.grid': False, # display grid or not
    'axes.titlesize': 'large', # fontsize of the axes title
    'axes.labelsize': 'large', # fontsize of the x any y labels
    'axes.labelcolor': '#555555',
    'axes.axisbelow': True, # whether axis gridlines and ticks are below
    'axes.color_cycle': ['#348ABD', '#7A68A6', '#A60628', '#467821', '#CF4457',
        '#188487', '#E24A33'],  # blue, purple, red, green, pink, turquoise, orange
    'axes.unicode_minus': True,
    'xtick.major.size': 8, # major tick size in points
    'xtick.minor.size': 4, # minor tick size in points
    'xtick.major.pad': 6, # distance to major tick label in points
    'xtick.minor.pad': 6, # distance to the minor tick label in points
    'xtick.color': '#555555', # color of the tick labels
    'xtick.labelsize': fontsize,
    'xtick.direction': 'out', # direction: in or out
    'ytick.major.size': 8, # major tick size in points
    'ytick.minor.size': 4, # minor tick size in points
    'ytick.major.pad': 6, # distance to major tick label in points
    'ytick.minor.pad': 6, # distance to the minor tick label in points
    'ytick.color': '#555555', # color of the tick labels
    'ytick.labelsize': fontsize,
    'ytick.direction': 'out', # direction: in or out
    'grid.color': '#BCBCBC',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'legend.fancybox': True,
    'figure.figsize': [11, 8], # figure size in inches
    'figure.dpi': 90, # figure dots per inch
    'figure.facecolor' : '0.85', # figure facecolor; 0.75 is scalar gray
    'figure.edgecolor' : '0.50', # figure edgecolor
    'keymap.fullscreen': 'f', # toggling
    'keymap.home': ['h', 'r', 'home'], # home or reset mnemonic
    'keymap.back': ['left', 'c', 'backspace'], # forward / backward keys to enable
    'keymap.forward': ['right', 'v'], #   left handed quick navigation
    'keymap.pan': 'p', # pan mnemonic
    'keymap.zoom': 'o', # zoom mnemonic
    'keymap.save': 's', # saving current figure
    'keymap.grid': 'g', # switching on/off a grid in current axes
    'keymap.yscale': 'l', # toggle scaling of y-axes ('log'/'linear')
    'keymap.xscale': ['L', 'k'], # toggle scaling of x-axes ('log'/'linear')
    'keymap.all_axes': 'a', # enable all axes
    'contour.negative_linestyle': '--',
}
rcParams.update(params)


try:
    reload(cm)
    reload(common)
    reload(dynamics)
    reload(file)
    reload(gis)
    reload(graphics)
    reload(interpolate)
    reload(signal)
    reload(stats)

except:
    import cm
    import common
    import dynamics
    import file
    import file as filemngmnt # for backwards compatibility
    import gis
    import gis as mapping
    import graphics
    import interpolate
    import signal
    import stats
