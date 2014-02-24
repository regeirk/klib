# -*- coding: iso-8859-1 -*-
"""Extra color maps.

"""

from matplotlib import rcParams, colors
from os.path import dirname
from numpy import loadtxt

###############################################################################
# CONSTANTS AND PARAMETERS
#

# Color look up table (LUT) size
_LUTSIZE = rcParams['image.lut']
    
_custom_jet_data = {
    'red': (
        (0.0, 1, 1), 
        (0.3, 0, 0),
        (0.4, 0.49803921568627452, 0.49803921568627452),
        (0.6, 1, 1),
        (0.7, 1, 1),
        (0.8, 1, 1), 
        (1, 0.5, 0.5)
    ),
    'green': (
        (0.0, 1, 1),
        (0.3, 0.80784313725490198, 0.80784313725490198),
        (0.4, 1, 1),
        (0.6, 0.84313725490196079, 0.84313725490196079),
        (1, 0, 0)
    ),
    'blue': (
        (0.0, 1, 1), 
        (0.3, 0.81960784313725488, 0.81960784313725488),
        (0.4, 0, 0),
        (0.6, 0, 0),
        (1, 0, 0)
    )
}

_custom_no_green_data = {
    'red': (
        (0, 1., 1.),
        (0.1, 1., 1.),
        (0.20000000298000001, 0.0, 0.0), 
        (0.3, 0.16470588743699999, 0.16470588743699999),
        (0.33333334326699998, 0.33333334326699998, 0.33333334326699998),
        (0.40000000596000002, 0.49803921580299998, 0.49803921580299998),
        (0.46666666865299999, 0.66666668653500005, 0.66666668653500005),
        (0.53333336114899998, 1.0, 1.0), 
        (0.60000002384200002, 1.0, 1.0),
        (0.66666668653500005, 1.0, 1.0), 
        (0.73333334922799998, 1.0, 1.0),
        (0.80000001192100001, 1.0, 1.0), 
        (0.86666667461400004, 1.0, 1.0),
        (0.93333333730699997, 1.0, 1.0), 
        (1.0, 0.5, 0.5)
    ),
    'green': (
        (0, 1., 1.),
        (0.1, 1., 1.),
        (0.3, 1.0, 1.0), 
        (0.33333334326699998, 1.0, 1.0),
        (0.40000000596000002, 1.0, 1.0), 
        (0.46666666865299999, 1.0, 1.0),
        (0.53333336114899998, 1.0, 1.0), 
        (0.60000002384200002, 0.94117647409399996, 0.94117647409399996),
        (0.66666668653500005, 0.74901962280300005, 0.74901962280300005),
        (0.73333334922799998, 0.65882354974699997, 0.65882354974699997),
        (0.80000001192100001, 0.54117649793599998, 0.54117649793599998),
        (0.86666667461400004, 0.43921568989799997, 0.43921568989799997),
        (0.93333333730699997, 0.30196079611799997, 0.30196079611799997),
        (1.0, 0.0, 0.0)
    ),
    'blue': (
        (0, 1.0, 1.0), 
        (0.1, 1., 1.),
        (0.3, 1.0, 1.0), 
        (0.33333334326699998, 1.0, 1.0), 
        (0.40000000596000002, 1.0, 1.0), 
        (0.46666666865299999, 1.0, 1.0),
        (0.53333336114899998, 0.32941177487399997, 0.32941177487399997),
        (0.60000002384200002, 0.0, 0.0), 
        (0.66666668653500005, 0.0, 0.0), 
        (0.73333334922799998, 0.0, 0.0),
        (0.80000001192100001, 0.0, 0.0), 
        (0.86666667461400004, 0.0, 0.0), 
        (0.93333333730699997, 0.0, 0.0), 
        (1.0, 0.0, 0.0)
    )
}

_custom_YlOrRd_data = {
    'red': (
        (0.0, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 1.0), 
        (1.0, 0.8, 0.8)
    ),
    'green': (
        (0.0, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.25, 1.0, 1.0),
        (0.5, 0.62352941176470589, 0.62352941176470589), 
        (1.0, 0.0, 0.0)
    ), 
    'blue': (
        (0.0, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0)
    )
}

# Custom colormap for chlorophyll-a maps. The long data defitions are based on
# http://oceancolor.gsfc.nasa.gov/DOCS/standard_chlorophyll_colorscale.txt
_custom_chla_data = loadtxt('%s/aux/chla_cmap.dat' % 
    (dirname(__file__)))[:, 2:] / 255.

###############################################################################
# GENERATES THE COLORMAPS AND REVERSE THEM TOO
#
custom_jet = colors.LinearSegmentedColormap('custom_jet', _custom_jet_data,
    _LUTSIZE)
custom_no_green = colors.LinearSegmentedColormap('custom_no_green',
    _custom_no_green_data, _LUTSIZE)
custom_YlOrRd = colors.LinearSegmentedColormap('custom_YlOrRd',
    _custom_YlOrRd_data, _LUTSIZE)
custom_chla = colors.ListedColormap(_custom_chla_data, name='custom_chla')

# Reverse the colormaps in 'datad' list. Reversed colormaps have '_r' appended 
# to their name.
def _revcmap(data):
    """Reverses the color map."""
    data_r = {}
    for key, val in data.iteritems():
        val = list(val)
        valrev = val[::-1]
        valnew = []
        for a, b, c in valrev:
            valnew.append((1. - a, b, c))
        data_r[key] = valnew
    return data_r

datad = dict(
    custom_jet = _custom_jet_data
)
_cmapnames = datad.keys()
for _cmapname in _cmapnames:
    _cmapname_r = '_cmaname' + '_r'
    _cmapdat_r = _revcmap(datad[_cmapname])
    datad[_cmapname_r] = _cmapdat_r
    locals()[_cmapname_r] = colors.LinearSegmentedColormap(_cmapname_r,
        _cmapdat_r, _LUTSIZE)
