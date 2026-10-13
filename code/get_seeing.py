"""Fetch SDSS DR17 Field.psfWidth_{r,i} (arcsec) for every (run, camcol, field) of the warp sample.
One SkyServer SqlSearch query per run. Output: field_seeing.csv (run,camcol,field,psfWidth_r,psfWidth_i)."""
import time, urllib.request, urllib.parse, numpy as np, os
import paths as P
rcf = np.loadtxt(P.cat('rcf_needed.txt'), dtype=int)
done = set()
if os.path.exists(P.cat('field_seeing.csv')):
    for l in open(P.cat('field_seeing.csv')):
        if l[0].isdigit():
            done.add(int(l.split(',')[0]))
out = open(P.cat('field_seeing.csv'), 'a')
if not done:
    out.write('run,camcol,field,psfWidth_r,psfWidth_i\n')
for run in np.unique(rcf[:, 0]):
    if run in done:
        continue
    m = rcf[rcf[:, 0] == run]
    cond = ' OR '.join('(camcol=%d AND field IN (%s))' % (cc, ','.join(str(f) for f in m[m[:, 1] == cc, 2]))
                       for cc in np.unique(m[:, 1]))
    sql = 'SELECT run, camcol, field, psfWidth_r, psfWidth_i FROM Field WHERE run=%d AND (%s)' % (run, cond)
    u = 'https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?' + urllib.parse.urlencode({'cmd': sql, 'format': 'csv'})
    for attempt in range(4):
        try:
            txt = urllib.request.urlopen(u, timeout=60).read().decode()
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1)); txt = ''
    rows = [l for l in txt.splitlines() if l and l[0].isdigit()]
    out.write('\n'.join(rows) + ('\n' if rows else '')); out.flush()
    time.sleep(0.2)
out.close()
