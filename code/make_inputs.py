"""Inputs derived from the catalogue match catalogue/warp_match.fits (the NSA v1_0_1 entries of the DF20 warp sample):

  catalogue/atlas_urls.txt   public SDSS DR17 URLs of the NSA atlas child images (download_atlas.sh)
  catalogue/rcf_needed.txt   the SDSS (run, camcol, field) of every galaxy, for the seeing query (get_seeing.py)
  catalogue/gal_seeing.npy   per-galaxy r- and i-band seeing FWHM (arcsec) from catalogue/field_seeing.csv
                             (SDSS DR17 Field.psfWidth, written by get_seeing.py); run again after get_seeing.py.
python3 make_inputs.py
"""
import os
import numpy as np
from astropy.io import fits
import paths as P

c = fits.open(P.cat('warp_match.fits'))[1].data
SAS = 'https://data.sdss.org/sas/dr17/sdss/atlas/v1/detect/v1_0'
urls = ['%s/%s/atlases/%d/%s-%d-atlas-%d.fits.gz' % (SAS, c['SUBDIR'][i].strip(), c['PID'][i], c['IAUNAME'][i].strip(),
                                                    c['PID'][i], c['AID'][i]) for i in range(len(c))]
open(P.cat('atlas_urls.txt'), 'w').write('\n'.join(urls) + '\n')
rcf = np.unique(np.column_stack([c['RUN'], c['CAMCOL'], c['FIELD']]).astype(int), axis=0)
np.savetxt(P.cat('rcf_needed.txt'), rcf, fmt='%d')
print('atlas_urls.txt: %d images; rcf_needed.txt: %d fields' % (len(urls), len(rcf)))
if os.path.exists(P.cat('field_seeing.csv')):
    f = np.genfromtxt(P.cat('field_seeing.csv'), delimiter=',', names=True)
    see = {(int(a), int(b), int(d)): (r, i) for a, b, d, r, i in zip(f['run'], f['camcol'], f['field'], f['psfWidth_r'], f['psfWidth_i'])}
    s = np.array([see.get((int(a), int(b), int(d)), (np.nan, np.nan)) for a, b, d in zip(c['RUN'], c['CAMCOL'], c['FIELD'])])
    np.save(P.cat('gal_seeing.npy'), s)
    print('gal_seeing.npy: %d galaxies, %d without a seeing value; median r %.2f" i %.2f"'
          % (len(s), np.isnan(s[:, 0]).sum(), np.nanmedian(s[:, 0]), np.nanmedian(s[:, 1])))
