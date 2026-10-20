"""
journal_style.py -- single figure style for the whole numerical section,
following top-journal (Operations Research / Management Science)
conventions:

  - strictly grayscale: curves are distinguished by gray level AND dash
    pattern AND open marker (triple encoding), never by colour
  - no coloured fills; shaded regions use light gray or sparse hatching
  - no coloured text or annotation arrows; annotations are plain black
  - serif (Times/STIX-like) mathematics via mathtext 'stix'
  - thin spines (0.6 pt), outward major+minor ticks, frameless legends
  - plain, non-bold panel labels "(a) ..." at small size

Call apply_style() once before importing pyplot-heavy code, or use
`with journal_style():` around a single figure.
"""

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Grayscale encoding
# ---------------------------------------------------------------------------
K_BLACK = 'black'
K_DARK = '0.25'
K_MED = '0.45'
K_LIGHT = '0.60'
GSCALE = [K_BLACK, K_DARK, K_MED, K_LIGHT]          # up to 4 curves
FILL_HI = '0.93'                                    # "positive" region
FILL_LO = '0.86'                                    # "negative" region
BAR_FACE = '0.88'
BAR_EDGE = '0.30'

DASH = dict(solid='-',
            dashed=(0, (4, 2)),
            dashdot=(0, (5, 2, 1, 2)),
            dotted=(0, (1.2, 1.6)),
            longdash=(0, (6, 3)))
DASHES = [DASH['solid'], DASH['dashed'], DASH['dashdot'], DASH['dotted']]
MARKERS = ['o', 's', '^', 'D']

_RCPARAMS = {
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'text.usetex': False,
    'axes.labelsize': 9.5,
    'axes.titlesize': 9.5,
    'axes.titleweight': 'normal',
    'axes.linewidth': 0.6,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'lines.linewidth': 1.1,
    'lines.markersize': 4,
}


def apply_style():
    """Apply the journal style globally."""
    mpl.rcParams.update(_RCPARAMS)


def journal_style():
    """Context manager: `with journal_style(): ...`."""
    return mpl.rc_context(_RCPARAMS)


def curve(i, marker=False, markevery=8):
    """Style dict for the i-th curve (grayscale + dash [+ open marker])."""
    d = dict(color=GSCALE[i % len(GSCALE)],
             linestyle=DASHES[i % len(DASHES)],
             linewidth=1.1)
    if marker:
        d.update(marker=MARKERS[i % len(MARKERS)], markersize=3.5,
                 markevery=markevery, markerfacecolor='white',
                 markeredgewidth=0.9)
    return d


def panel_label(ax, label):
    """Plain non-bold panel label, e.g. panel_label(ax, '(a) ...')."""
    ax.set_title(label, fontsize=9.5, fontweight='normal', pad=6)


def open_marker(ax, x, y, i=0, size=4.5):
    """Open (white-faced) marker used to flag crossing points etc."""
    ax.plot(x, y, marker=MARKERS[i % len(MARKERS)], color=GSCALE[i],
            markersize=size, markerfacecolor='white',
            markeredgewidth=1.1, zorder=5, linestyle='none')
