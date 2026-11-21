"""
Donoho--Johnstone benchmark for the Cauchy/Cauchy posterior mean.

Python port of COMPARE_<SIGNAL>_CACA_EXPORT_ESTSIG.m (author's MATLAB code),
with additional comparison rules (horseshoe posterior mean with levelwise
MMLE of tau, EBayesThresh posterior mean, BayesThresh, BlockJS, VisuShrink).

Design
  N = 1024, coarsest retained level L = 3 (8 scaling coefficients),
  noise sd = 1 (used only to generate data), sigma estimated by finest-level
  MAD in every replication; root SNR = sd(f)/sigma (population sd).
  Blocks: Haar; Bumps: Daubechies-3 (6 taps); Doppler, HeaviSine: Symmlet-4.
  All methods see the same noisy realization in each replication.
"""
import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr, erf
from scipy.optimize import minimize_scalar, minimize

SQ2 = np.sqrt(2.0)
Z075 = 0.6744897501960817
PI = np.pi

# --------------------------------------------------------------------------
# Signals and filters (identical to the MATLAB scripts)
# --------------------------------------------------------------------------

def dj_signal(name, N):
    t = np.arange(1, N + 1) / (N + 1)
    pos = np.array([.10, .13, .15, .23, .25, .40, .44, .65, .76, .78, .81])
    if name == 'Blocks':
        ht = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
        return np.sum(ht[:, None] * (t[None, :] >= pos[:, None]), axis=0)
    if name == 'Bumps':
        ht = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
        w = np.array([.005, .005, .006, .010, .010, .030, .010, .010, .005, .008, .005])
        return np.sum(ht[:, None] * (1 + np.abs((t[None, :] - pos[:, None]) / w[:, None])) ** (-4), axis=0)
    if name == 'Doppler':
        return np.sqrt(t * (1 - t)) * np.sin(2.1 * PI / (t + 0.05))
    if name == 'HeaviSine':
        return 4 * np.sin(4 * PI * t) - np.sign(t - 0.30) - np.sign(0.72 - t)
    raise ValueError(name)


FILTERS = {
    'Haar': (np.array([1, 1]) / SQ2, np.array([1, -1]) / SQ2),
    'Daub3': (np.array([0.0352262918857095, -0.0854412738820267, -0.135011020010255,
                        0.459877502118492, 0.806891509311093, 0.332670552950083]),
              np.array([-0.332670552950083, 0.806891509311093, -0.459877502118492,
                        -0.135011020010255, 0.0854412738820267, 0.0352262918857095])),
    'Symm4': (np.array([-0.0757657147892733, -0.0296355276459985, 0.497618667632015,
                        0.803738751805916, 0.297857795605277, -0.0992195435768472,
                        -0.0126039672620378, 0.0322231006040427]),
              np.array([-0.0322231006040427, -0.0126039672620378, 0.0992195435768472,
                        0.297857795605277, -0.803738751805916, 0.497618667632015,
                        0.0296355276459985, -0.0757657147892733])),
}
SIGNAL_FILTER = {'Blocks': 'Haar', 'Bumps': 'Daub3', 'Doppler': 'Symm4', 'HeaviSine': 'Symm4'}

# --------------------------------------------------------------------------
# Periodic orthonormal DWT (port of my_periodic_dwt), as an explicit matrix
# Ordering: [a_L (2^L), d_L (2^L), d_{L+1}, ..., d_{J-1}]; level j occupies
# indices 2^j ... 2^{j+1}-1.
# --------------------------------------------------------------------------

def _analysis(x, h, g):
    n = x.shape[0]
    idx = (2 * np.arange(n // 2)[:, None] + np.arange(len(h))[None, :]) % n
    xx = x[idx]                       # (n/2, nh, ...)
    a = np.tensordot(xx, h, axes=([1], [0])) if x.ndim == 1 else np.einsum('kij,i->kj', xx, h)
    d = np.tensordot(xx, g, axes=([1], [0])) if x.ndim == 1 else np.einsum('kij,i->kj', xx, g)
    return a, d


def dwt_matrix(h, g, N, L):
    X = np.eye(N)
    J = int(np.log2(N))
    out = []
    a = X
    for j in range(J - 1, L - 1, -1):
        a, d = _analysis(a, h, g)
        out.append(d)
    blocks = [a] + out[::-1]
    W = np.vstack(blocks)
    assert np.allclose(W @ W.T, np.eye(N), atol=1e-8), 'DWT matrix not orthogonal'
    return W


def lev(j):
    return slice(2 ** j, 2 ** (j + 1))


def mad_sigma(x):
    return max(np.median(np.abs(x - np.median(x))) / Z075, 1e-8)


def soft(x, T):
    return np.sign(x) * np.maximum(np.abs(x) - T, 0.0)


# --------------------------------------------------------------------------
# 1. Cauchy/Cauchy posterior mean (port of caca_mean_vectorized)
# --------------------------------------------------------------------------
TGRID = np.linspace(-24, 12, 6001)
with np.errstate(invalid='ignore', divide='ignore'):
    QCACA = TGRID / (PI ** 2 * np.sinh(TGRID))
QCACA[np.abs(TGRID) < 1e-10] = 1 / PI ** 2


def caca_mean_std(z, tauStd, tgrid=TGRID, q=QCACA):
    """Posterior mean of theta/sigma given z=|d|/sigma>=0, prior Ca/Ca(tauStd)."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    a = tauStd * np.exp(tgrid)[:, None]
    for s in range(0, z.size, 400):
        zz = z[s:s + 400][None, :]
        Wp = q[:, None] * np.exp(-0.5 * (a - zz) ** 2)
        Wm = q[:, None] * np.exp(-0.5 * (a + zz) ** 2)
        den = np.trapezoid(Wp + Wm, tgrid, axis=0)
        num = np.trapezoid(a * (Wp - Wm), tgrid, axis=0)
        r = np.where(den > 1e-300, num / np.maximum(den, 1e-300), 0.0)
        out[s:s + 400] = np.maximum(r, 0.0)
    return out


class InterpRule:
    """Tabulated odd rule rho(z) on z in [0, zmax]; exact evaluation beyond."""
    def __init__(self, fn, zmax=40.0, dz=0.002):
        self.zg = np.arange(0, zmax + dz / 2, dz)
        self.vals = fn(self.zg)
        self.fn = fn
        self.zmax = zmax

    def __call__(self, d, sigma):
        z = np.abs(d) / sigma
        out = np.interp(np.minimum(z, self.zmax), self.zg, self.vals)
        big = z > self.zmax
        if np.any(big):
            out[big] = self.fn(z[big])
        return sigma * np.sign(d) * out


# --------------------------------------------------------------------------
# 2. Horseshoe posterior mean, levelwise MMLE of tau on [1/n_j, 1]
#    theta/sigma | lambda ~ N(0, lambda^2 tau^2), lambda ~ C+(0,1).
# --------------------------------------------------------------------------
HS_T = np.linspace(-20, 20, 2001)
HS_W = 1.0 / (PI * np.cosh(HS_T))
HS_TAUS = np.exp(np.linspace(np.log(1 / 512), 0.0, 49))


def hs_logm_and_mean(z, tau):
    z = np.asarray(z, float)
    s2 = (tau * np.exp(HS_T)) ** 2
    v = 1 + s2
    logm = np.empty_like(z)
    fac = np.empty_like(z)
    for s in range(0, z.size, 500):
        zz = z[s:s + 500][None, :]
        lg = -0.5 * zz ** 2 / v[:, None] - 0.5 * np.log(2 * PI * v)[:, None] + np.log(HS_W)[:, None]
        mx = lg.max(axis=0)
        E = np.exp(lg - mx)
        den = np.trapezoid(E, HS_T, axis=0)
        num = np.trapezoid(E * (s2 / v)[:, None], HS_T, axis=0)
        logm[s:s + 500] = mx + np.log(den)
        fac[s:s + 500] = num / den
    return logm, fac


class HorseshoeTables:
    def __init__(self, zmax=40.0, dz=0.005):
        self.zg = np.arange(0, zmax + dz / 2, dz)
        self.zmax = zmax
        self.logm = []
        self.fac = []
        for tau in HS_TAUS:
            lm, fc = hs_logm_and_mean(self.zg, tau)
            self.logm.append(lm)
            self.fac.append(fc)

    def _eval(self, z, k):
        lm = np.interp(np.minimum(z, self.zmax), self.zg, self.logm[k])
        fc = np.interp(np.minimum(z, self.zmax), self.zg, self.fac[k])
        big = z > self.zmax
        if np.any(big):
            lm[big], fc[big] = hs_logm_and_mean(z[big], HS_TAUS[k])
        return lm, fc

    def level(self, d, sigma):
        z = np.abs(d) / sigma
        n = z.size
        ks = [k for k, t in enumerate(HS_TAUS) if t >= 1.0 / n - 1e-12]
        best, bk = -np.inf, ks[0]
        for k in ks:
            ll = self._eval(z, k)[0].sum()
            if ll > best:
                best, bk = ll, k
        fc = self._eval(z, bk)[1]
        return fc * d, HS_TAUS[bk]


# --------------------------------------------------------------------------
# 3. BAMS (port of bams_embedded / bams_Wavmat / bams_bayesrule)
# --------------------------------------------------------------------------

def bams_wavmat(h, N, k0, shift=2):
    J = int(np.log2(N))
    h = np.asarray(h, float)
    g = np.flip(h * (-1.0) ** np.arange(1, len(h) + 1))
    hp = np.concatenate([h, np.zeros(N)])
    gp = np.concatenate([g, np.zeros(N)])
    old = np.eye(2 ** (J - k0))
    for k in range(k0, 0, -1):
        ubJk, ubJk1 = 2 ** (J - k), 2 ** (J - k + 1)
        ii = np.arange(1, ubJk1 + 1)[:, None]
        jj = np.arange(1, ubJk + 1)[None, :]
        mod = np.mod(N + ii - 2 * jj + shift, ubJk1)
        mod = mod + (mod == 0) * ubJk1
        hmat = hp[mod - 1]
        gmat = gp[mod - 1]
        old = np.vstack([old @ hmat.T, gmat.T])
    return old


def bams_rule(d, mu, tau, eps):
    x = np.abs(d)
    r = np.sqrt(2 * mu)
    with np.errstate(all='ignore'):
        de = 0.5 * r * np.exp(-x * r)
        m = (tau * np.exp(-x / tau) - np.exp(-r * x) / r) / (2 * tau ** 2 - 1 / mu)
        br = (tau * (tau ** 2 - 1 / (2 * mu)) * x * np.exp(-x / tau)
              + tau ** 2 / mu * (np.exp(-r * x) - np.exp(-x / tau))) / \
             ((tau ** 2 - 1 / (2 * mu)) * (tau * np.exp(-x / tau) - np.exp(-r * x) / r))
        bre = (1 - eps) * m * br / ((1 - eps) * m + eps * de)
    bad = ~np.isfinite(bre)
    bre[bad] = x[bad]   # numerical guard: underflow at very large |d| -> identity
    return np.sign(d) * bre, int(bad.sum())


def bams(y, Wb, J, depth, Ctukey=2, gam=2):
    sw = Wb @ y
    coarsest = J - depth
    swt = sw.copy()
    niz = np.sort(sw[2 ** (J - 1):])
    mm = niz.size
    lo, hi = mm // 4, (3 * mm) // 4
    pseudos = abs(niz[hi - 1] - niz[lo - 1]) / Ctukey
    mu = 1 / pseudos ** 2
    tauu = np.sqrt(max(np.std(sw, ddof=1) ** 2 - 1 / mu, 1e-4))
    nbad = 0
    for i in range(J - 1, coarsest - 1, -1):
        eps_i = 1 - 1 / (i - coarsest + 1) ** gam
        tau_i = tauu * np.sqrt(i)
        swt[lev(i)], b = bams_rule(sw[lev(i)], mu, tau_i, eps_i)
        nbad += b
    return Wb.T @ swt, nbad


# --------------------------------------------------------------------------
# 4/5. Johnstone--Silverman EBayes, Laplace slab a=0.5: posterior median/mean
# --------------------------------------------------------------------------

def phi0(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * PI)


def lap_parts(x, a):
    x = np.abs(x)
    ph = phi0(x)
    Ap = 0.5 * a * np.exp(0.5 * a ** 2 - a * x + log_ndtr(x - a))
    Am = 0.5 * a * np.exp(0.5 * a ** 2 + a * x + log_ndtr(-x - a))
    g = np.maximum(Ap + Am, 1e-300)
    return ph, g, Ap, Am


def eb_fit_w(z, a):
    n = z.size
    ph, g, _, _ = lap_parts(z, a)
    tU = np.sqrt(2 * np.log(max(n, 2)))
    phT, _, ApT, AmT = lap_parts(np.array([tU]), a)
    denT = phT[0] + (ApT[0] - AmT[0])
    wMin = 1e-8 if (denT <= 0 or not np.isfinite(denT)) else phT[0] / denT
    wMin = min(max(wMin, 1e-8), 1 - 1e-8)
    score = lambda w: np.sum((g - ph) / np.maximum((1 - w) * ph + w * g, 1e-300))
    if score(wMin) <= 0:
        return wMin
    if score(1 - 1e-10) >= 0:
        return 1 - 1e-10
    lo, hi = wMin, 1 - 1e-10
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eb_level(d, sigma, a=0.5, kind='median'):
    z = d / sigma
    w = eb_fit_w(z, a)
    x = np.abs(z)
    ph, g, Ap, _ = lap_parts(x, a)
    wpost = w * g / np.maximum((1 - w) * ph + w * g, 1e-300)
    if kind == 'mean':
        # slab posterior mean: x - a*(e^{-ax}Phi(x-a) - e^{ax}Phi(-x-a))/(...)
        l1 = -a * x + log_ndtr(x - a)
        l2 = a * x + log_ndtr(-x - a)
        mx = np.maximum(l1, l2)
        e1, e2 = np.exp(l1 - mx), np.exp(l2 - mx)
        mu_slab = x - a * (e1 - e2) / (e1 + e2)
        return sigma * np.sign(z) * wpost * mu_slab
    Ftail0 = Ap / np.maximum(g, 1e-300)
    keep = (wpost * Ftail0 > 0.5) & (x > 0)
    mabs = np.zeros_like(x)
    if np.any(keep):
        xx, wp = x[keep], wpost[keep]
        # u = D e^{ax}/(2 wp) = [Phi(x-a) + e^{2ax}Phi(-x-a)]/(2 wp)
        u = (ndtr(xx - a) + np.exp(2 * a * xx + log_ndtr(-xx - a))) / (2 * wp)
        u = np.clip(u, 1e-14, 1 - 1e-14)
        mabs[keep] = np.maximum(xx - a - ndtri(u), 0.0)
    return sigma * np.sign(z) * mabs


# --------------------------------------------------------------------------
# 6. BayesThresh (Abramovich, Sapatinas & Silverman 1998), posterior median
#    theta_jk ~ pi_j N(0, tau_j^2) + (1-pi_j) delta_0,
#    tau_j^2 = C1 2^{-alpha j}, pi_j = min(1, C2 2^{-beta j}), alpha=.5, beta=1;
#    (C1, C2) by marginal maximum likelihood over the processed levels.
# --------------------------------------------------------------------------

def bayesthresh(D, sigma, L, J, alpha=0.5, beta=1.0):
    def nll(par):
        C1, C2 = np.exp(par)
        tot = 0.0
        for j in range(L, J):
            d = D[j]
            t2 = C1 * 2.0 ** (-alpha * j)
            p = min(1.0, C2 * 2.0 ** (-beta * j))
            v1 = sigma ** 2 + t2
            l0 = -0.5 * d ** 2 / sigma ** 2 - 0.5 * np.log(2 * PI * sigma ** 2)
            l1 = -0.5 * d ** 2 / v1 - 0.5 * np.log(2 * PI * v1)
            a = np.log(max(1 - p, 1e-300)) + l0
            b = np.log(max(p, 1e-300)) + l1
            mx = np.maximum(a, b)
            tot += np.sum(mx + np.log(np.exp(a - mx) + np.exp(b - mx)))
        return -tot
    best = None
    for c1 in np.log([1, 10, 100, 1000]):
        for c2 in np.log([1, 10, 100]):
            r = minimize(nll, [c1, c2], method='Nelder-Mead',
                         options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 400})
            if best is None or r.fun < best.fun:
                best = r
    C1, C2 = np.exp(best.x)
    out = {}
    for j in range(L, J):
        d = D[j]
        t2 = C1 * 2.0 ** (-alpha * j)
        p = min(1.0, C2 * 2.0 ** (-beta * j))
        v1 = sigma ** 2 + t2
        x = np.abs(d)
        l0 = np.log(max(1 - p, 1e-300)) - 0.5 * x ** 2 / sigma ** 2 - 0.5 * np.log(sigma ** 2)
        l1 = np.log(max(p, 1e-300)) - 0.5 * x ** 2 / v1 - 0.5 * np.log(v1)
        p0 = 1 / (1 + np.exp(np.clip(l1 - l0, -700, 700)))      # P(theta=0 | d)
        m = t2 / v1 * x
        s = np.sqrt(sigma ** 2 * t2 / v1)
        # median > 0 iff (1-p0) Phi(-m/s) + p0 < 1/2
        pos = (1 - p0) * ndtr(-m / s) + p0 < 0.5
        med = np.zeros_like(x)
        q = (0.5 - p0[pos]) / (1 - p0[pos])
        med[pos] = np.maximum(m[pos] + s * ndtri(np.clip(q, 1e-15, 1 - 1e-15)), 0.0)
        out[j] = np.sign(d) * med
    return out


# --------------------------------------------------------------------------
# 7/8. Raised-cosine and Epanechnikov Bayes (ports)
# --------------------------------------------------------------------------
XGL, WGL = np.polynomial.legendre.leggauss(64)


def raised_cosine_level(d, sigma, alpha, tau):
    th = tau * XGL[:, None]
    qw = tau * WGL[:, None]
    gp = (1 + np.cos(PI * th / tau)) / (2 * tau)
    E = np.exp(-0.5 * ((th - d[None, :]) / sigma) ** 2)
    sm = np.sum(qw * gp * E, axis=0)
    sn = np.sum(qw * gp * th * E, axis=0)
    zl = np.exp(-0.5 * (d / sigma) ** 2)
    return (1 - alpha) * sn / np.maximum(alpha * zl + (1 - alpha) * sm, 1e-300)


def epanechnikov_level(d, alpha, beta, lam):
    th = beta * XGL[:, None]
    qw = beta * WGL[:, None]
    gp = (3 / (4 * beta)) * (1 - (th / beta) ** 2)
    rate = np.sqrt(2 * lam)
    E = np.exp(-rate * np.abs(th - d[None, :]))
    sm = np.sum(qw * gp * E, axis=0)
    sn = np.sum(qw * gp * th * E, axis=0)
    zl = np.exp(-rate * np.abs(d))
    return (1 - alpha) * sn / np.maximum(alpha * zl + (1 - alpha) * sm, 1e-300)


# --------------------------------------------------------------------------
# 9. Smooth SCAD-SURE (port)
# --------------------------------------------------------------------------

def scad_apply(d, lam, a):
    if lam <= 0:
        return d.copy()
    r = np.abs(d)
    out = np.zeros_like(d)
    mid = (r > lam) & (r < a * lam)
    hi = r >= a * lam
    s = (r[mid] - lam) / ((a - 1) * lam)
    out[mid] = d[mid] - np.sign(d[mid]) * lam * np.cos(0.5 * PI * s) ** 2
    out[hi] = d[hi]
    return out


def scad_sure_level(d, sigma, a=2.0, nl=121):
    n = d.size
    lu = sigma * np.sqrt(2 * np.log(max(n, 2)))
    grid = np.concatenate([[0.0], np.linspace(lu / nl, lu, nl)])
    r = np.abs(d)
    sure = np.empty(grid.size)
    sure[0] = n * sigma ** 2
    for i, lam in enumerate(grid[1:], 1):
        low = r <= lam
        mid = (r > lam) & (r < a * lam)
        F = np.full(n, sigma ** 2)
        F[low] = d[low] ** 2 - sigma ** 2
        s = (r[mid] - lam) / ((a - 1) * lam)
        hmag = lam * np.cos(0.5 * PI * s) ** 2
        hp = -PI / (2 * (a - 1)) * np.sin(PI * s)
        F[mid] = hmag ** 2 + sigma ** 2 - 2 * sigma ** 2 * hp
        sure[i] = F.sum()
    return scad_apply(d, grid[np.argmin(sure)], a)


# --------------------------------------------------------------------------
# 10. NCPmean-2 (Wang & Wood), port
# --------------------------------------------------------------------------
U01 = 0.5 * (XGL + 1)
WU01 = 0.5 * WGL


def ncp_quant(z, s2, lam, m=2):
    r = z / (2 * s2)
    eta = m / 2 + lam
    lb = np.log(WU01)[:, None] + eta * np.log(U01)[:, None] - U01[:, None] * r[None, :]
    mx = lb.max(axis=0)
    W = np.exp(lb - mx)
    den = np.maximum(W.sum(axis=0), 1e-300)
    A1 = (W * U01[:, None]).sum(axis=0) / den
    A2 = (W * U01[:, None] ** 2).sum(axis=0) / den
    logR = np.log(lam + 1) + r + mx + np.log(den)
    return logR, A1, A2


def ncp_nll(p, logR):
    a = np.log(p)
    b = np.log(1 - p) + logR
    mx = np.maximum(a, b)
    v = -np.sum(mx + np.log(np.exp(a - mx) + np.exp(b - mx)))
    return v if np.isfinite(v) else 1e200


def ncp_fit_p(logR):
    r = minimize_scalar(lambda p: ncp_nll(p, logR), bounds=(1e-9, 1 - 1e-9),
                        method='bounded', options={'xatol': 1e-9})
    cand = [1e-10, r.x, 1 - 1e-10]
    vals = [ncp_nll(c, logR) for c in cand]
    return min(max(cand[int(np.argmin(vals))], 1e-10), 1 - 1e-10)


def ncpmean2(D, L, J):
    m = 2
    Z = {j: (D[j].reshape(-1, m) ** 2).sum(axis=1) for j in range(L, J)}
    allz = np.concatenate([Z[j] for j in range(L, J)])
    zf = (D[J - 1].reshape(-1, m) ** 2).sum(axis=1)
    s2 = max(np.median(zf) / (2 * np.log(2)), 1e-12)
    nb = allz.size
    p0 = np.mean(allz < 2 * m * s2 * np.log(max(nb, 2)))
    sumM = m * nb
    exc = allz.sum() - s2 * sumM
    if exc <= 1e-12 * s2 * sumM:
        bb = 1e8 / max(s2, 1e-12)
    else:
        bb = max((1 - p0) * sumM / exc, 1e-12 / max(s2, 1e-12))
    ub = np.clip(bb * s2 / (1 + bb * s2), 1e-10, 1 - 1e-10)
    lam = max(-np.log(2) / np.log(ub) - 1, -1 + 1e-8)
    out = {}
    for j in range(L, J):
        z = Z[j]
        logR, A1, A2 = ncp_quant(z, s2, lam, m)
        pJ = ncp_fit_p(logR)
        x = np.log(pJ) - np.log(1 - pJ) - logR
        pi0 = 1 / (1 + np.exp(-np.clip(x, -700, 700)))
        Pm = np.maximum((1 - pi0) * (m * s2 + z - (m * s2 + 2 * z) * A1 + z * A2), 0)
        fac = np.where(z > 1e-14, np.sqrt(Pm / np.maximum(z, 1e-300)), 0.0)
        out[j] = (D[j].reshape(-1, m) * fac[:, None]).ravel()
    return out


# --------------------------------------------------------------------------
# 11. LNWS posterior mean (Remenyi & Vidakovic), port
# --------------------------------------------------------------------------

def lnws_logm(x, b):
    x = np.maximum(x, 1e-300)
    q = 1 + 2 * b
    lm0 = 0.5 * np.log(x) - 0.5 * np.log(2 * PI) - 0.5 * x
    E = np.maximum(erf(np.sqrt(x / (2 * q))), 1e-300)
    lm1 = np.log(b) - 0.5 * np.log(q) - (b / q) * x + np.log(E)
    return lm0, lm1


def lnws_level_nll(e, lm0, lm1):
    A = np.log(max(e, 1e-300)) + lm0
    B = np.log(max(1 - e, 1e-300)) + lm1
    mx = np.maximum(A, B)
    v = -np.sum(mx + np.log(np.exp(A - mx) + np.exp(B - mx)))
    return v if np.isfinite(v) else 1e200


def lnws_fit_eps(x, b):
    lm0, lm1 = lnws_logm(x, b)
    f = lambda e: lnws_level_nll(e, lm0, lm1)
    r = minimize_scalar(f, bounds=(1e-8, 1 - 1e-8), method='bounded', options={'xatol': 1e-8})
    e = r.x
    vals = [f(1e-10), f(e), f(1 - 1e-10)]
    i = int(np.argmin(vals))
    return [1e-10, e, 1 - 1e-10][i], vals[i]


def lnws(D, L, J, sigma, bmin=1e-6, bmax=10, nb=61):
    s2 = sigma ** 2
    X = {}
    for j in range(L, J):
        c = D[j]
        p = D[j - 1]
        X[j] = np.maximum((c[0::2] ** 2 + c[1::2] ** 2 + p ** 2) / s2, 1e-14)
    allx = np.concatenate(list(X.values()))
    b0 = 1 / max(allx.mean(), 1e-12)
    blo = max(min(bmin, b0 / 100), 1e-10)
    bhi = min(max(bmax, b0 * 100), 1e4)
    lg = np.linspace(np.log(blo), np.log(bhi), nb)
    prof = lambda lb: sum(lnws_fit_eps(X[j], np.exp(lb))[1] for j in range(L, J))
    pv = np.array([prof(v) for v in lg])
    im = int(np.argmin(pv))
    ilo, ihi = max(0, im - 1), min(nb - 1, im + 1)
    if ilo == ihi:
        lb = lg[im]
    else:
        lb = minimize_scalar(prof, bounds=(lg[ilo], lg[ihi]), method='bounded',
                             options={'xatol': 1e-7}).x
    b = np.exp(lb)
    q = 1 + 2 * b
    out = {}
    for j in range(L, J):
        x = X[j]
        e, _ = lnws_fit_eps(x, b)
        lm0, lm1 = lnws_logm(x, b)
        A = np.log(max(e, 1e-300)) + lm0
        B = np.log(max(1 - e, 1e-300)) + lm1
        mx = np.maximum(A, B)
        p0 = np.exp(A - mx) / (np.exp(A - mx) + np.exp(B - mx))
        E = np.maximum(erf(np.sqrt(x / (2 * q))), 1e-300)
        dE = (q + x + np.sqrt(2 * x * q / PI) * np.exp(-x / (2 * q)) / E) / q ** 2
        fac = np.sqrt(np.maximum((1 - p0) * dE, 0) / np.maximum(x, 1e-14))
        c = D[j].copy()
        c[0::2] *= fac
        c[1::2] *= fac
        out[j] = c
    return out


# --------------------------------------------------------------------------
# 12-16. SureShrink, BayesShrink, BiShrink, BlockJS, VisuShrink
# --------------------------------------------------------------------------

def sureshrink_level(d, sigma):
    n = d.size
    zs = np.sort(np.abs(d) / sigma)
    z2 = zs ** 2
    k = np.arange(1, n + 1)
    sure = n - 2 * k + np.cumsum(z2) + (n - k) * z2
    t = min(zs[np.argmin(sure)], np.sqrt(2 * np.log(max(n, 2))))
    return soft(d, sigma * t)


def bayesshrink_level(d, sigma):
    sx = np.sqrt(max(np.mean(d ** 2) - sigma ** 2, 0))
    if sx <= np.sqrt(np.finfo(float).eps) * max(1, sigma):
        return np.zeros_like(d)
    return soft(d, sigma ** 2 / sx)


def periodic_local_mean(x, w=7):
    r = (w - 1) // 2
    return sum(np.roll(x, k) for k in range(-r, r + 1)) / w


def bishrink(D, L, J, sigma, w=7):
    out = {}
    for j in range(L, J):
        c = D[j]
        p = np.repeat(D[j - 1], 2)
        S = np.sqrt(np.maximum(periodic_local_mean(c ** 2, w) - sigma ** 2, np.finfo(float).eps))
        T = np.sqrt(3) * sigma ** 2 / S
        R = np.sqrt(c ** 2 + p ** 2)
        fac = np.where(R > 1e-300, np.maximum(R - T, 0) / np.maximum(R, 1e-300), 0.0)
        out[j] = c * fac
    return out


def blockjs_level(d, sigma, N, lam=4.50524):
    Lb = max(int(np.floor(np.log(N))), 1)
    out = np.empty_like(d)
    for s in range(0, d.size, Lb):
        b = d[s:s + Lb]
        S2 = np.sum(b ** 2)
        out[s:s + Lb] = max(0.0, 1 - lam * b.size * sigma ** 2 / S2) * b if S2 > 0 else 0 * b
    return out


# --------------------------------------------------------------------------
# Method registry
# --------------------------------------------------------------------------
METHODS = ['CaCaMean', 'Horseshoe', 'BAMS', 'EBayesMed', 'EBayesMean', 'BayesThresh',
           'RaisedCosine', 'Epanechnikov', 'NCPmean2', 'LNWS', 'BayesShrink', 'BiShrink',
           'SureShrink', 'SmoothSCAD', 'BlockJS', 'VisuShrink']


class Bench:
    def __init__(self, signal, N=1024, L=3, snr=5.0, snr_type='sd', C_mean=45.0, alpha_mean=1.45):
        self.signal, self.N, self.L = signal, N, L
        self.J = J = int(np.log2(N))
        f0 = dj_signal(signal, N)
        if snr_type == 'sd':            # root SNR = sd(f)/sigma, population sd
            self.f = f0 * snr / np.std(f0)
        else:                           # MATLAB DJTests convention mean(f^2)=SNR
            self.f = f0 * np.sqrt(snr / np.mean(f0 ** 2))
        h, g = FILTERS[SIGNAL_FILTER[signal]]
        self.h = h
        self.W = dwt_matrix(h, g, N, L)
        self.W2 = dwt_matrix(h, g, N, L - 1)
        self.Wb = bams_wavmat(h, N, J - L)
        self.cmean = {j: InterpRule(lambda z, t=C_mean * 2.0 ** (-alpha_mean * j): caca_mean_std(z, t))
                      for j in range(L, J)}
        self.hs = HorseshoeTables()
        self.bams_bad = 0

    def split(self, w, L):
        return {j: w[lev(j)].copy() for j in range(L - 1 if L > 0 else 0, self.J)}

    def recon(self, w, Dhat, W):
        w = w.copy()
        for j, v in Dhat.items():
            w[lev(j)] = v
        return W.T @ w

    def run(self, y):
        L, J, N = self.L, self.J, self.N
        w = self.W @ y
        D = {j: w[lev(j)] for j in range(L, J)}
        sig = mad_sigma(D[J - 1])
        est = {}

        def levelwise(fn):
            return self.recon(w, {j: fn(D[j], j) for j in range(L, J)}, self.W)

        est['CaCaMean'] = levelwise(lambda d, j: self.cmean[j](d, sig))
        est['Horseshoe'] = levelwise(lambda d, j: self.hs.level(d, sig)[0])
        fb, nb = bams(y, self.Wb, J, J - L)
        self.bams_bad += nb
        est['BAMS'] = fb
        est['EBayesMed'] = levelwise(lambda d, j: eb_level(d, sig, 0.5, 'median'))
        est['EBayesMean'] = levelwise(lambda d, j: eb_level(d, sig, 0.5, 'mean'))
        est['BayesThresh'] = self.recon(w, bayesthresh(D, sig, L, J), self.W)
        tauRC = max(max(np.max(np.abs(D[j])) for j in range(L, J)), 1e-8)
        est['RaisedCosine'] = levelwise(lambda d, j: raised_cosine_level(
            d, sig, min(max(1 - 1 / (j - L + 1) ** 2, 0), 1 - 1e-12), tauRC))
        sF = max(np.std(D[J - 1], ddof=1), 1e-8)
        lamEP = 1 / sF ** 2 + (1 / 2) * np.exp(-sF / 2)
        est['Epanechnikov'] = levelwise(lambda d, j: epanechnikov_level(
            d, min(max(1 - 1 / (j - L + 1) ** 2, 0), 1 - 1e-12), max(np.max(np.abs(d)), 1e-8), lamEP))
        est['NCPmean2'] = self.recon(w, ncpmean2(D, L, J), self.W)
        w2 = self.W2 @ y
        D2 = {j: w2[lev(j)] for j in range(L - 1, J)}
        est['LNWS'] = self.recon(w2, lnws(D2, L, J, mad_sigma(D2[J - 1])), self.W2)
        est['BayesShrink'] = levelwise(lambda d, j: bayesshrink_level(d, sig))
        est['BiShrink'] = self.recon(w2, bishrink(D2, L, J, sig), self.W2)
        est['SureShrink'] = levelwise(lambda d, j: sureshrink_level(d, sig))
        est['SmoothSCAD'] = levelwise(lambda d, j: scad_sure_level(d, sig))
        est['BlockJS'] = levelwise(lambda d, j: blockjs_level(d, sig, N))
        tU = sig * np.sqrt(2 * np.log(N))
        est['VisuShrink'] = levelwise(lambda d, j: soft(d, tU))
        return est, sig
