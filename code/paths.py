"""Locations of the data products. Every script imports this module instead of hard-coding paths.

  DFW_DATA    the unpacked Zenodo data record (default <repo>/data, where fetch_data.py puts it)
  DFW_ATLAS   the NSA atlas images (default <repo>/atlas, where download_atlas.sh puts them)
  DFW_MASKED  set to 1 to run the inference on the point-source-masked warps (the masking test of Sec. IV D)

Re-running a pipeline step overwrites its products under DFW_DATA; point DFW_DATA at a copy to keep the originals.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('DFW_DATA', os.path.join(ROOT, 'data'))
ATLAS = os.environ.get('DFW_ATLAS', os.path.join(ROOT, 'atlas'))
MASKED = os.environ.get('DFW_MASKED', '0') == '1'
SIGNALS = os.path.join(DATA, 'signals')                     # forward-model predicted-signal tables


def cat(*p):                                                # catalogue inputs: NSA match, halos, seeing
    return os.path.join(DATA, 'catalogue', *p)


def meas(*p):                                               # warp measurements
    return os.path.join(DATA, 'measurement', *p)


def sg(*p):                                                 # self-gravity profiles and the S_i calibration
    return os.path.join(DATA, 'selfgravity', *p)


def mask(*p):                                               # point-source masking
    return os.path.join(DATA, 'masking', *p)


def lap(*p):                                                # likelihood runs (Laplace grid)
    return os.path.join(DATA, 'inference', *p)


def results(*p):                                            # summary products behind the paper's numbers
    return os.path.join(DATA, 'results', *p)


NEWF = mask('new_w1_masked_gaia5.npz') if MASKED else meas('new_w1.npz')
LAPDIR = lap('masked') if MASKED else lap()
