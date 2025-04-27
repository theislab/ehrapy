from enum import Enum

from matplotlib.colors import LinearSegmentedColormap


class Colormaps(Enum):
    """Available colormaps."""

    grey_red = LinearSegmentedColormap.from_list("grouping", ["lightgray", "red", "darkred"], N=128)
    grey_green = LinearSegmentedColormap.from_list("grouping", ["lightgray", "limegreen", "forestgreen"], N=128)
    grey_yellow = LinearSegmentedColormap.from_list("grouping", ["lightgray", "yellow", "gold"], N=128)
    grey_violet = LinearSegmentedColormap.from_list("grouping", ["lightgray", "mediumvioletred", "indigo"], N=128)
    grey_blue = LinearSegmentedColormap.from_list("grouping", ["lightgray", "cornflowerblue", "darkblue"], N=128)
