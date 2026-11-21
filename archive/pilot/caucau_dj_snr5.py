#!/usr/bin/env python3
"""Reproducible Cauchy/Cauchy wavelet shrinkage illustration.
Four Donoho--Johnstone signals, N=1024, SD-ratio SNR=5, noise SD 1,
finest-detail centered MAD plug-in SD. 15 paired comparator rules.

Prerequisites: numpy, scipy, numba. Run: python caucau_dj_snr5.py --M 1000
All method choices are made without reference to true signal coefficients.
Some classical 2D methods (BiShrink) are explicitly adapted to 1D.
"""
import argparse, csv, os
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr, ndtri, log_ndtr
from numba import njit

N=1024
J=10
J0=3
C_M, A_M=45.0,1.45
C_MED,A_MED=35.0,1.25
ZMAX=35.0
ZSTEP=0.035
ZGRID=np.arange(0,ZMAX+ZSTEP/2,ZSTEP)
TGRID=np.linspace(-24,12,2401)
# Prefix columns exactly as stored in CSV / LaTeX
METHODS=['CauCau mean','CauCau median','CauCau SPoM','CauCau Bayes test',
'BAMS','EB Laplace median','EB Laplace mean','EB Gaussian mean',
'Horseshoe EB mean','Raised-cosine Bayes','Epanechnikov Bayes',
'BayesShrink','SureShrink','BiShrink (1D)','BlockJS','Universal soft']
assert len(METHODS)==16


def wavelet_matrix(N,h,shift=2):
    """Exactly reproduces the sparse, periodic Wavmat convention in SCOPE_Exp123."""
    J=int(np.log2(N)); old=sparse.eye(2**J0,format='csr')
    h=np.asarray(h,dtype=float); h=h/np.linalg.norm(h)
    # MATLAB: g=fliplr(h.*(-1).^(1:numel(h)))
    g=(h * (-1.)**np.arange(1,h.size+1))[::-1]
    for k in range(J-J0,0,-1):
        m=2**(J-k); nin=2*m
        # Matrix sample at MATLAB one-based col = mod(tap+2*row-2-1,nin)+1
        rows=np.repeat(np.arange(m),len(h))
        taps=np.tile(np.arange(len(h)),m)
        cols=(2*rows+taps)%nin
        H=sparse.coo_matrix((h[taps],(rows,cols)),shape=(m,nin)).tocsr()
        G=sparse.coo_matrix((g[taps],(rows,cols)),shape=(m,nin)).tocsr()
        old=sparse.vstack((old@H,G),format='csr')
    rng=np.random.default_rng(39)
    v=rng.standard_normal(N)
    resid=np.linalg.norm(old.T@(old@v)-v)/np.linalg.norm(v)
    assert resid<1e-10, ('Bad wavelet orthogonality',resid)
    return old

FILTERS={
 'Blocks':[1,1],
 'Bumps':[.3326705529500826,.8068915093110926,.4598775021184916,-.1350110200102546,-.08544127388202666,.03522629188570954],
 'Doppler':[-.07576571478950221,-.029635527646002493,.497618667632775,.8037387518051321,.29785779560530606,-.09921954357663353,-.012603967262031304,.032223100604051466],
 'HeaviSine':[-.07576571478950221,-.029635527646002493,.497618667632775,.8037387518051321,.29785779560530606,-.09921954357663353,-.012603967262031304,.032223100604051466]
}

def dj_signal(name):
    t=np.arange(1,N+1)/N
    pos=np.array([.10,.13,.15,.23,.25,.40,.44,.65,.76,.78,.81])
    ht_b=np.array([4.,5.,3.,4.,5.,4.2,2.1,4.3,3.1,5.1,4.2])
    ht_k=np.array([4.,-5.,3.,-4.,5.,-4.2,2.1,4.3,-3.1,2.1,-4.2])
    widths=np.array([.005,.005,.006,.010,.010,.030,.010,.010,.005,.008,.005])
    if name=='Doppler': f0=np.sqrt(t*(1-t))*np.sin(2*np.pi*1.05/(t+.05))
    elif name=='HeaviSine': f0=4*np.sin(4*np.pi*t)-np.sign(t-.3)-np.sign(.72-t)
    elif name=='Bumps': f0=np.sum(ht_b[:,None]/(1+np.abs((t[None,:]-pos[:,None])/widths[:,None]))**4,axis=0)
    elif name=='Blocks': f0=np.sum(ht_k[:,None]*(t[None,:]>=pos[:,None]),axis=0)
    else: raise ValueError(name)
    f=5*f0/f0.std(ddof=0)
    assert abs(f.std(ddof=0)-5)<1e-11
    return f


def prior_q(t):
    with np.errstate(invalid='ignore',divide='ignore'):
        out=t/(np.pi**2*np.sinh(t))
    out[np.abs(t)<1e-9]=1/np.pi**2
    return out


def caca_lookup(tau,what):
    """Cau/Cau exact posterior integrals on a dense, standardized grid.
    Implements the t=log(theta/tau) quadrature in the supplied MATLAB.
    """
    aa=tau*np.exp(TGRID)
    q=prior_q(TGRID)
    dt=TGRID[1]-TGRID[0]
    out=np.empty_like(ZGRID)
    for k in range(0,len(ZGRID),64):
        z=ZGRID[k:k+64]
        p=q[:,None]*np.exp(-.5*(aa[:,None]-z[None,:])**2)
        m=q[:,None]*np.exp(-.5*(aa[:,None]+z[None,:])**2)
        den=np.trapezoid(p+m,dx=dt,axis=0)
        if what=='mean':
            num=np.trapezoid(aa[:,None]*(p-m),dx=dt,axis=0)
            out[k:k+len(z)]=np.maximum(num/den,0)
        else:
            pp=np.trapezoid(p,dx=dt,axis=0)
            mm=np.trapezoid(m,dx=dt,axis=0)
            target=.5*(pp-mm)
            cum=np.vstack((np.zeros((1,len(z))),np.cumsum(.5*dt*(p[:-1]+p[1:]),axis=0)))
            ix=np.argmax(cum>=target[None,:],axis=0)
            ix=np.maximum(ix,1)
            ll=cum[ix-1,np.arange(len(z))];rr=cum[ix,np.arange(len(z))]
            frac=np.maximum(0,np.minimum(1,(target-ll)/np.maximum(rr-ll,1e-300)))
            tmid=TGRID[ix-1]+frac*dt
            out[k:k+len(z)]=tau*np.exp(tmid)
    out[0]=0.0
    return out


def lookup_shrink(d,sigma,tab):
    z=np.abs(d)/sigma
    v=np.interp(np.minimum(z,ZMAX),ZGRID,tab)
    # Cauchy/Cauchy posterior correction ~2/z for large signals.
    v=np.where(z>ZMAX,np.maximum(z-2/np.maximum(z,1),0),v)
    return np.sign(d)*sigma*v


def score_H(t):
    if abs(t)<1e-4:
        return -1+2*t/3-t*t/6+t**3/45-t**4/120+13*t**5/3780-t**6/5040
    return -1/np.sinh(t)+np.exp(-t)/t


def spom_parameters(j):
    tau=100*2**(-1.1*j)
    fn=lambda t:tau*np.exp(t)-score_H(t)/tau
    opt=minimize_scalar(fn,bounds=(-20,20),method='bounded',options={'xatol':1e-11})
    t0=opt.x;dcrit=opt.fun
    top=max(20.,np.log(100/tau)+1)
    tg=np.linspace(t0,top,9000)
    gg=np.array([fn(v) for v in tg])
    # birth is increasing for t>=t0
    assert np.min(np.diff(gg))>-1e-6
    return tau,dcrit,gg,tau*np.exp(tg)


def caca_bt_threshold(j):
    tau=20*2.**(-j)
    eps=1.25*(j/J0)**(2/3)
    inside=TGRID<=np.log(eps/tau)
    aa=tau*np.exp(TGRID)
    qq=prior_q(TGRID)
    def p0(z):
        w=qq*(np.exp(-.5*(aa-z)**2)+np.exp(-.5*(aa+z)**2))
        return np.trapezoid(w[inside],TGRID[inside])/np.trapezoid(w,TGRID)
    if p0(0)<=.5: return 0.
    return brentq(lambda z:p0(z)-.5,0,50,xtol=1e-8)


def hs_lookup(tau,uq,uw):
    """Horseshoe marginal likelihood and 1-E[kappa|z] at sigma=1.
    tau is local prior SD * half-Cauchy parameter in noise SD units.
    """
    r=1/tau
    kappa=(1-uq*uq)
    wk=uw/(1+(r*r-1)*uq*uq)
    m=np.empty(len(ZGRID)); coef=np.empty(len(ZGRID))
    for i in range(0,len(ZGRID),100):
        z=ZGRID[i:i+100]
        wt=wk[:,None]*np.exp(-.5*kappa[:,None]*z[None,:]**2)
        integ=wt.sum(axis=0)
        ekappa=(kappa[:,None]*wt).sum(axis=0)/integ
        # m(z|tau)=2*r/(pi*sqrt(2pi))*integ
        m[i:i+len(z)]=np.log(2*r/(np.pi*np.sqrt(2*np.pi)))+np.log(integ)
        coef[i:i+len(z)]=1-ekappa
    return m,coef


def hs_select_and_estimate(d,sigma,hs_lik,hs_coef):
    """Marginal maximum likelihood tau on a frozen grid, per detail level."""
    z=np.minimum(np.abs(d)/sigma[:,None],ZMAX)
    nrep=d.shape[0]
    scores=np.empty((nrep,hs_lik.shape[0]))
    for k in range(len(scores[0])):
        scores[:,k]=np.interp(z,ZGRID,hs_lik[k]).sum(axis=1)
    best=np.argmax(scores,axis=1)
    coeff=np.empty_like(d)
    for k in np.unique(best):
        rr=np.flatnonzero(best==k)
        coeff[rr]=np.interp(z[rr],ZGRID,hs_coef[k])
    return d*coeff,best


def eb_lap_parts(x,a=.5):
    phi=np.exp(-.5*x*x)/np.sqrt(2*np.pi)
    # log stable for tails and large x
    Ap=.5*a*np.exp(.5*a*a-a*x+log_ndtr(x-a))
    Am=.5*a*np.exp(.5*a*a+a*x+log_ndtr(-x-a))
    return phi,Ap+Am,Ap,Am


def eb_laplace(d,sigma,a=.5):
    """MML spike mixing weight then Laplace-slab posterior mean and median.
    Exact same posterior-median convention as author's archived MATLAB.
    """
    nrep,n=d.shape
    z=d/sigma[:,None]; x=np.abs(z)
    phi,g,Ap,Am=eb_lap_parts(x,a)
    Tu=np.sqrt(2*np.log(max(n,2)))
    phiT,gT,ApT,AmT=eb_lap_parts(np.asarray(Tu),a)
    w_min=float(np.clip(phiT/(phiT+ApT-AmT),1e-8,1-1e-8))
    # w MML fixed a and universal posterior median cap
    def score(w):
        return np.sum((g-phi)/np.maximum((1-w)*phi+w*g,1e-300),axis=1)
    low=np.full(nrep,w_min);high=np.full(nrep,1-1e-10)
    sLo=score(low[:,None]);sHi=score(high[:,None])
    for _ in range(46):
        mid=(low+high)/2
        sm=score(mid[:,None]);gt=sm>0
        low=np.where(gt,mid,low);high=np.where(gt,high,mid)
    w=np.where(sLo<=0,w_min,np.where(sHi>=0,1-1e-10,(low+high)/2))
    denom=(1-w[:,None])*phi+w[:,None]*g
    post=w[:,None]*g/np.maximum(denom,1e-300)
    mu_slab=x+a*(Am-Ap)/np.maximum(g,1e-300)
    mu=np.sign(z)*np.maximum(post*mu_slab,0)
    # Posterior median: wpost*P_slab(theta>0|x)<=1/2 gives zero.
    Fplus=Ap/np.maximum(g,1e-300)
    use= (post*Fplus>.5) & (x>0)
    med=np.zeros_like(x)
    if np.any(use):
        xx=x[use];pp=post[use]
        DD=np.exp(-a*xx)*ndtr(xx-a)+np.exp(a*xx)*ndtr(-xx-a)
        target=np.clip(DD*np.exp(a*xx)/(2*pp),1e-13,1-1e-13)
        med[use]=np.maximum(xx-a-ndtri(target),0)
    return sigma[:,None]*mu, sigma[:,None]*np.sign(z)*med


def eb_gaussian(d,sigma):
    """Gaussian spike + N(0,v) slab; ML via levelwise 2-parameter EM."""
    z=d/sigma[:,None];n=z.shape[1]
    vs=np.maximum(np.mean(z*z,axis=1)-1,.1)
    pi=np.clip(np.mean(np.abs(z)>1.75,axis=1)*1.5,.03,.95)
    for it in range(45):
        # ratio slab / null N(0,1+v)/N(0,1)
        lr=-.5*np.log1p(vs[:,None])+.5*z*z*vs[:,None]/(1+vs[:,None])
        q=1/(1+(1-pi[:,None])/pi[:,None]*np.exp(-np.clip(lr,-400,400)))
        newpi=np.clip(np.mean(q,axis=1),1/n,.9999)
        newv=np.clip(np.sum(q*z*z,axis=1)/np.maximum(q.sum(axis=1),1e-15)-1,.001,1e4)
        if max(np.max(np.abs(newv-vs)/(1+vs)),np.max(np.abs(newpi-pi)))<1e-5:
            vs=newv;pi=newpi;break
        vs=newv;pi=newpi
    lr=-.5*np.log1p(vs[:,None])+.5*z*z*vs[:,None]/(1+vs[:,None])
    q=1/(1+(1-pi[:,None])/pi[:,None]*np.exp(-np.clip(lr,-400,400)))
    return d*q*vs[:,None]/(1+vs[:,None])


@njit(cache=True)
def gauss_quad_bayes(d,sigma,alpha,slabscale,rate,xq,wq,kind):
    """Adaptively localized 64-point Gauss-Legendre posterior mean.

    Normal likelihood: integrate raised-cosine slab over +/-9 sigma
    of the observed coefficient, truncated to its compact support.
    Laplace likelihood: split the integral at theta=d and truncate at
    +/-20/rate, which handles the likelihood cusp and wide supports.
    The Gaussian and double-exponential likelihoods are those used by
    the original author's RC and Epanechnikov implementations.
    """
    nr,n=d.shape
    out=np.zeros((nr,n))
    for i in range(nr):
        ss=sigma[i];bb=slabscale[i]
        if bb<1e-12: continue
        for j in range(n):
            x=d[i,j];den=0.;num=0.
            if kind==0:
                lo=max(-bb,x-9*ss); hi=min(bb,x+9*ss)
                if hi>lo:
                    mid=(hi+lo)*.5;half=(hi-lo)*.5
                    for q in range(len(xq)):
                        th=mid+half*xq[q]
                        prior=.5*(1+np.cos(np.pi*th/bb))/bb
                        lik=np.exp(-.5*((th-x)/ss)**2)
                        w=half*wq[q]*prior*lik
                        den+=w;num+=w*th
                atom=np.exp(-.5*(x/ss)**2)
            else:
                rr=rate[i]; cutoff=20/rr
                for side in range(2):
                    if side==0:
                        lo=max(-bb,x-cutoff);hi=min(bb,x)
                    else:
                        lo=max(-bb,x);hi=min(bb,x+cutoff)
                    if hi>lo:
                        mid=(hi+lo)*.5;half=(hi-lo)*.5
                        for q in range(len(xq)):
                            th=mid+half*xq[q];u=th/bb
                            prior=(3/(4*bb))*(1-u*u)
                            lik=np.exp(-rr*abs(th-x))
                            w=half*wq[q]*prior*lik
                            den+=w;num+=w*th
                atom=np.exp(-rr*abs(x))
            out[i,j]=(1-alpha)*num/((1-alpha)*den+alpha*atom+1e-300)
    return out


def bams_all(d,sigma):
    """BAMS author coefficient rule in wavelet domain, vectorized MxN.
    Uses the same MAD noise estimate, but BAMS total-energy slab scale.
    """
    nr,n=d.shape
    # Force the same centered finest-detail MAD plug-in noise variance
    # as every other estimator (BAMS variance-matched adaptation).
    sigmaB2=sigma*sigma
    mu=1/np.maximum(sigmaB2,1e-15)
    std_all=d.std(axis=1,ddof=1)
    tauu=np.sqrt(np.maximum(std_all**2-sigmaB2,.0001))
    z=d.copy()
    for j in range(J0,J):
        dj=d[:,2**j:2**(j+1)]
        eps=1-1/(j-J0+1)**2
        tau=tauu*np.sqrt(j)
        # source: de = .5 sqrt(2mu)exp(-abs(d)sqrt(2mu)); m convolution Laplace
        invs=np.sqrt(sigmaB2/2)
        aa=np.sqrt(2*mu)
        e1=np.exp(-np.abs(dj)/tau[:,None]);e2=np.exp(-np.abs(dj)*aa[:,None])
        dv=2*tau**2-sigmaB2
        marginal=(tau[:,None]*e1-invs[:,None]*e2)/dv[:,None]
        de=.5*aa[:,None]*e2
        br_num=(tau*(tau*tau-sigmaB2/2))[:,None]*np.abs(dj)*e1 +\
               (tau*tau*sigmaB2)[:,None]*(e2-e1)
        br_den=(tau*tau-sigmaB2/2)[:,None]*(tau[:,None]*e1-invs[:,None]*e2)
        br=br_num/np.where(np.abs(br_den)<1e-300,1e-300,br_den)
        post=(1-eps)*marginal/np.maximum((1-eps)*marginal+eps*de,1e-300)
        est=np.sign(dj)*np.maximum(br*post,0)
        # rare 0/0 underflow: large coefficients should escape instead
        est=np.where(np.isfinite(est),est,dj)
        z[:,2**j:2**(j+1)]=est
    return z


def sureshrink(d,sigma):
    n=d.shape[1]
    z=np.abs(d)/sigma[:,None]
    ss=np.sort(z,axis=1)
    cs=np.cumsum(ss*ss,axis=1)
    kk=np.arange(1,n+1)
    sure=n-2*kk[None,:]+cs+(n-kk[None,:])*ss*ss
    ii=np.argmin(sure,axis=1)
    thresh=np.minimum(ss[np.arange(len(d)),ii],np.sqrt(2*np.log(n)))*sigma
    return np.sign(d)*np.maximum(np.abs(d)-thresh[:,None],0)


def soft(d,thresh):
    return np.sign(d)*np.maximum(np.abs(d)-thresh,0)


def bishrink(d,j,sigma,all_d):
    # Sendur-Selesnick 1D adapted parent, 7-point periodic local variance
    n=d.shape[1]
    if j==J0: parent=np.zeros_like(d)
    else: parent=np.repeat(all_d[:,2**(j-1):2**j],2,axis=1)
    local=sum(np.roll(d*d,lag,axis=1) for lag in range(-3,4))/7
    sx=np.sqrt(np.maximum(local-sigma[:,None]**2,1e-10))
    radius=np.sqrt(d*d+parent*parent)
    T=np.sqrt(3)*sigma[:,None]**2/sx
    return d*np.maximum(1-T/np.maximum(radius,1e-300),0)


def blockjs(d,sigma):
    L=max(1,int(np.floor(np.log(N))))
    z=np.zeros_like(d)
    for start in range(0,d.shape[1],L):
        block=d[:,start:start+L];m=block.shape[1]
        energy=np.sum(block*block,axis=1)
        w=np.maximum(1-4.50524*m*sigma*sigma/np.maximum(energy,1e-300),0)
        z[:,start:start+m]=block*w[:,None]
    return z


def evaluate_one_signal(name,seed,M,tables,hs_lik,hs_shrink,uq,uw,xq,wq):
    print('Signal',name,'M',M,flush=True)
    f=dj_signal(name)
    W=wavelet_matrix(N,FILTERS[name]);truth=W@f
    rng=np.random.default_rng(seed)
    err=rng.standard_normal((M,N))
    y=f[None,:]+err
    d=(W@y.T).T
    fine=d[:,N//2:]
    sigma=np.median(np.abs(fine-np.median(fine,axis=1)[:,None]),axis=1)/.6744897501960817
    assert np.all(np.isfinite(sigma)) and min(sigma)>.1
    # q methods coefficient storage for 16, all scaling coefficients retained.
    est=np.repeat(d[:,:,None],len(METHODS),axis=2)
    est[:,:,4]=bams_all(d,sigma)
    est[:,2**J0:,15]=soft(d[:,2**J0:],sigma[:,None]*np.sqrt(2*np.log(N)))
    for j in range(J0,J):
        a=2**j;b=2**(j+1);xx=d[:,a:b]
        m=tables[('mean',j)];md=tables[('median',j)]
        est[:,a:b,0]=lookup_shrink(xx,sigma[:,None],m)
        est[:,a:b,1]=lookup_shrink(xx,sigma[:,None],md)
        tau,dcrit,spx,spy=tables[('spom',j)]
        gamma=np.sqrt(np.log(N))/2
        zx=np.abs(xx)/sigma[:,None]
        est[:,a:b,2]=np.sign(xx)*sigma[:,None]*np.interp(np.minimum(zx,spx[-1]),spx,spy)* (zx>gamma*dcrit)
        est[:,a:b,3]=xx*(zx>tables[('bt',j)])
        lpm,lmed=eb_laplace(xx,sigma,a=.5)
        est[:,a:b,5]=lmed
        est[:,a:b,6]=lpm
        est[:,a:b,7]=eb_gaussian(xx,sigma)
        hs,ind=hs_select_and_estimate(xx,sigma,hs_lik,hs_shrink)
        est[:,a:b,8]=hs
        alpha=min(max(1-1/(j-J0+1)**2,0),1-1e-12)
        tauRC=np.maximum(np.max(np.abs(d[:,2**J0:]),axis=1),1e-8)
        est[:,a:b,9]=gauss_quad_bayes(xx,sigma,alpha,tauRC,np.zeros(M),xq,wq,0)
        # Variance-match the published Epanechnikov hyperparameter to MAD.
        lambdaEp=1/sigma**2+.5*np.exp(-sigma/2)
        betaEp=np.maximum(np.max(np.abs(xx),axis=1),1e-8)
        est[:,a:b,10]=gauss_quad_bayes(xx,sigma,alpha,betaEp,np.sqrt(2*lambdaEp),xq,wq,1)
        sigvar=sigma[:,None]**2
        sigy=np.mean(xx*xx,axis=1)
        sigx=np.sqrt(np.maximum(sigy-sigma*sigma,0))
        tbayes=np.divide(sigma*sigma,sigx,out=np.full(M,np.inf),where=(sigx>1e-7))
        est[:,a:b,11]=soft(xx,tbayes[:,None])
        est[:,a:b,12]=sureshrink(xx,sigma)
        est[:,a:b,13]=bishrink(xx,j,sigma,d)
        est[:,a:b,14]=blockjs(xx,sigma)
    assert np.all(np.isfinite(est)),('Nonfinite results for',name, METHODS[np.argwhere(~np.isfinite(est))[0][2]])
    mse=np.mean((est-truth[None,:,None])**2,axis=1)
    amse=mse.mean(axis=0);sd=mse.std(axis=0,ddof=1)
    assert np.all(amse>0),amse
    return amse,sd,mse,sigma,d,f,est


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--M',type=int,default=1000)
    parser.add_argument('--output',default=os.path.dirname(__file__));args=parser.parse_args()
    os.makedirs(args.output,exist_ok=True)
    print('Precomputing posterior lookup rules',flush=True)
    tables={}
    for j in range(J0,J):
        tables[('mean',j)]=caca_lookup(C_M*2**(-A_M*j),'mean')
        tables[('median',j)]=caca_lookup(C_MED*2**(-A_MED*j),'median')
        tables[('spom',j)]=spom_parameters(j)
        tables[('bt',j)]=caca_bt_threshold(j)
        print('j',j,'bt',tables[('bt',j)],'SPoM dcrit',tables[('spom',j)][1],flush=True)
    uq,uw=leggauss(128);uq=(uq+1)/2;uw=uw/2
    hs_tau=np.logspace(-1.5,1.5,28)
    hsc=[hs_lookup(t,uq,uw) for t in hs_tau]
    hs_lik=np.stack([v[0] for v in hsc]);hs_shrink=np.stack([v[1] for v in hsc])
    xq,wq=leggauss(64)
    S=['Blocks','Bumps','Doppler','HeaviSine']
    means=[];sds=[];mse_all={};sigmadiag=[]
    for k,name in enumerate(S):
        checkpoint=os.path.join(args.output,'checkpoint_%s_M%d_allMAD_v2.npz'%(name,args.M))
        if os.path.exists(checkpoint):
            saved=np.load(checkpoint)
            amse=saved['amse'];sd=saved['sd'];mse=saved['mse'];
            sigh=saved['sigmadiag']
            print('Resuming',name,'from checked simulation checkpoint',flush=True)
        else:
            res=evaluate_one_signal(name,20261104+100*k,args.M,tables,hs_lik,hs_shrink,uq,uw,xq,wq)
            amse,sd,mse,sigma,d,f,est=res
            sigh=np.array([sigma.mean(),sigma.std(ddof=1)])
            np.savez_compressed(checkpoint,amse=amse,sd=sd,mse=mse,
                                sigmadiag=sigh,methods=np.array(METHODS))
            if k==0:
                np.savez_compressed(os.path.join(args.output,'caucau_example_blocks.npz'),
                  f=f,ycoeff=d[0],sigmahat=sigma[0],methods=np.array(METHODS),
                  estimators=est[0,:,:])
            del res,est,d,f,sigma
        means.append(amse);sds.append(sd);mse_all[name]=mse
        sigmadiag.append((name,float(sigh[0]),float(sigh[1])))
        print(name,[(METHODS[i],round(float(v),5)) for i,v in enumerate(amse)],flush=True)
    means=np.asarray(means).T;sds=np.asarray(sds).T
    path=os.path.join(args.output,'caucau_dj_snr5_M%d_amse.csv'%args.M)
    with open(path,'w',newline='') as h:
        c=csv.writer(h);c.writerow(['Method']+[nm+'_AMSE' for nm in S]+[nm+'_SD_MSE' for nm in S])
        for i,nm in enumerate(METHODS):c.writerow([nm]+[f'{v:.8f}' for v in means[i]]+[f'{v:.8f}' for v in sds[i]])
    with open(os.path.join(args.output,'caucau_dj_snr5_M%d_sigmahat.csv'%args.M),'w',newline='') as h:
        c=csv.writer(h);c.writerow(['Signal','mean_sigmahat','sd_sigmahat']);c.writerows(sigmadiag)
    np.savez_compressed(os.path.join(args.output,'caucau_dj_snr5_M%d_mse.npz'%args.M),
          **{nm:mse_all[nm] for nm in S},method=np.asarray(METHODS))
    print('Output',path,flush=True)

if __name__=='__main__':main()
