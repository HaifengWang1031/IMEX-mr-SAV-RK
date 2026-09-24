"""Explicit initial conditions and forcing used by the migrated NS experiments."""
import numpy as np
from solver import FourierNS


def build(c):
    nu = c['nu']
    force = c['forcing']
    if force['kind'] not in ('none', 'cos_x', 'cos_y'):
        raise ValueError('Unknown vorticity forcing')
    def forcing(x, y, t):
        if force['kind'] == 'none':
            return np.zeros_like(x)
        z = x if force['kind'] == 'cos_x' else y
        return force['amplitude'] * np.cos(force['wavenumber'] * z)
    m = FourierNS(nu, (c['N'], c['N']), tuple(c['domain']), forcing, c['threads'])
    x, y = m.X[:-1, :-1], m.Y[:-1, :-1]
    ic = c['initial']; kind = ic['kind']
    if kind in ('trig', 'random'):
        w = np.zeros_like(x); rng = np.random.default_rng(ic['seed'])
        for k in range(1, ic['modes'] + 1):
            for j in range(1, ic['modes'] + 1):
                a, b = rng.uniform(0, 2*np.pi, 2) if kind == 'random' else (0, 0)
                w += (k*k+j*j)**(-1.5)*np.cos(k*x+a)*np.cos(j*y+b)
        w -= w.mean(); w *= ic['amplitude']/np.sqrt(np.mean(w*w))
    elif kind == 'bursting':
        psi = np.zeros_like(x)
        for k in range(-10, 11):
            for j in range(-10, 11):
                r = np.hypot(k, j)
                if 0 < r <= 10:
                    psi += r**-3*(np.cos(k*x)+np.sin(k*x))*(np.cos(j*y)+np.sin(j*y))
        u, v = m.stream2velocity(ic['eps']*psi)
        w = m.velocity2vorticity(u, v)
    elif kind == 'shear':
        u = np.where(y <= .5, np.tanh(ic['rho']*(y-.25)), np.tanh(ic['rho']*(.75-y)))
        v = ic['delta']*np.sin(2*np.pi*x)
        w = m.velocity2vorticity(u, v)
    elif kind == 'isotropic':
        if ic['modes'] >= c['N']/3:
            raise ValueError('Initial Fourier support must lie within the dealiased band')
        rng = np.random.default_rng(ic['seed']); coeff = {}
        for ky in range(-ic['modes'], ic['modes']+1):
            for kx in range(-ic['modes'], ic['modes']+1):
                r = np.hypot(kx, ky)
                if r == 0 or r > ic['modes'] or ky < 0 or (ky == 0 and kx < 0):
                    continue
                z = np.exp(-.5*((r-ic['peak'])/ic['width'])**2)*np.exp(1j*rng.uniform(0, 2*np.pi))
                coeff[kx,ky] = z; coeff[-kx,-ky] = z.conjugate()
        scale = ic['amplitude']/np.sqrt(sum(abs(z)**2 for z in coeff.values()))
        hat = np.zeros_like(x, dtype=complex)
        for (kx,ky), z in coeff.items():
            hat[ky%c['N'], kx%c['N']] = c['N']**2*scale*z
        w = np.fft.ifft2(hat).real
    else:
        raise ValueError(f'Unsupported initial condition: {kind}')
    return m, m.initial_state(w)
