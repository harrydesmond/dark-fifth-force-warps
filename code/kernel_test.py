import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from selfgrav import kernel
# (1) elliptic closed form vs direct phi integral
a, b = 2.3, 1.1
print('phi integral', quad(lambda p: (a - b*np.cos(p))**-1.5, 0, 2*np.pi)[0],
      'closed', (lambda m: 4*__import__('scipy.special').special.ellipe(m)/((a-b)*np.sqrt(a+b)))(2*b/(a+b)))
# (2) uniform sheet Sigma=1 out to Rmax, bend z = J0(kR): self-force at R ~ 0 should be 2 pi G Sigma k J0(kR)
for k in (2., 4.):
    for n, Rmax, h in ((6000, 60., 0.01), (12000, 60., 0.005)):
        R = np.linspace(0.005, Rmax, n); dR = R[1]-R[0]
        w = 1. * 2*np.pi*R*dR * np.exp(-(R/(0.8*Rmax))**8)      # smooth outer taper
        K = kernel(R, w, h)
        z = j0(k*R)
        f = (K * (z[None, :] - z[:, None])).sum(1)
        i = np.searchsorted(R, [1., 3., 5.])
        print('k=%g n=%d h=%g  -f/(2pi k z) at R=1,3,5:' % (k, n, h), np.round(-f[i]/(2*np.pi*k*z[i]), 3))
