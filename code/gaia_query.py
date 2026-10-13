"""Gaia DR3 sources within 3.2 R + 25" (R = SERSIC_TH50) of each warp_match.fits row (TAP sync, one query per galaxy).
python3 gaia_query.py <start> <stop>  ->  gaia/gaia_<start>_<stop>.json
{row: [[ra, dec, ref_epoch, G, bp_rp, parallax_over_error, pmra, pmdec, pmra_error, pmdec_error, astrometric_params_solved,
        astrometric_excess_noise, ruwe], ...]}  (null -> NaN)"""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
import paths as P
CAT = fits.open(P.cat('warp_match.fits'))[1].data
URL = 'https://gea.esac.esa.int/tap-server/tap/sync'
COLS = ('ra, dec, ref_epoch, phot_g_mean_mag, bp_rp, parallax_over_error, pmra, pmdec, pmra_error, pmdec_error, '
        'astrometric_params_solved, astrometric_excess_noise, ruwe')


def query(row):
    ra, dec, R = float(CAT['RA'][row]), float(CAT['DEC'][row]), float(CAT['SERSIC_TH50'][row])
    rad = max((3.2 * R + 25.) / 3600., 0.5 / 60.)
    q = ("SELECT %s FROM gaiadr3.gaia_source WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.6f, %.6f, %.6f)) = 1"
         % (COLS, ra, dec, rad))
    data = urllib.parse.urlencode(dict(REQUEST='doQuery', LANG='ADQL', FORMAT='csv', QUERY=q)).encode()
    for attempt in range(8):
        try:
            txt = urllib.request.urlopen(URL, data, timeout=90).read().decode()
            lines = txt.strip().split('\n')[1:]
            return row, [[float(x) if x not in ('', 'null') else float('nan') for x in l.split(',')] for l in lines if l]
        except Exception as e:
            err = e
            time.sleep(2 + 3 * attempt)
    return row, 'error: %s' % err


if __name__ == '__main__':
    a, b = int(sys.argv[1]), min(int(sys.argv[2]), len(CAT))
    os.makedirs(P.mask('gaia'), exist_ok=True)
    t = time.time()
    fn = P.mask('gaia', 'gaia_%05d_%05d.json' % (a, b))
    res = {int(k): v for k, v in json.load(open(fn)).items()} if os.path.exists(fn) else {}
    todo = [r for r in range(a, b) if not isinstance(res.get(r), list)]    # missing or failed rows only
    with ThreadPoolExecutor(int(os.environ.get('NTHREAD', 6))) as ex:
        res.update(dict(ex.map(query, todo)))
    bad = [r for r, v in res.items() if isinstance(v, str)]
    json.dump(res, open(fn, 'w'))
    print('rows %d-%d: %d failed, %d sources, %.0f s' % (a, b, len(bad), sum(len(v) for v in res.values() if not isinstance(v, str)), time.time() - t))
