import numpy as np
import pandas as pd
from lib.ecp import *

def test_FitCarrPelts():
    df = pd.read_csv("data/test_spxVol220603.csv")
    CP = FitCarrPelts(df,fixVol=True,optMethod='Evolution')
    print(CP)

def test_FitEnsembleCarrPelts():
    df = pd.read_csv("data/test_spxVol220603.csv")
    CP = FitEnsembleCarrPelts(df,fixVol=True,optMethod='Evolution')
    print(CP)

def test_ImpVolECP():
    np.set_printoptions(precision=7, suppress=True, linewidth=np.inf)

    df = pd.read_csv("data/test_spxVol220603.csv")

    Texp = df['Texp'].unique()
    Nexp = len(Texp)

    w0 = np.zeros(Nexp)
    T0 = df["Texp"].to_numpy()

    k = np.log(df["Strike"]/df["Fwd"])
    k = k.to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()
    midVar = (bid**2+ask**2)/2

    ### ATM vol
    for j,T in enumerate(Texp):
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[j] = spline(0).item()*T # ATM total variance

    sig0 = np.sqrt(w0/Texp)

    K = df['Strike'].to_numpy()
    T = df['Texp'].to_numpy()
    D = df['PV'].to_numpy()
    F = df['Fwd'].to_numpy()

    #### zgrid
    zcfg = (-100,150,50)

    zgrid = np.arange(*zcfg)
    N = len(zgrid)

    #### fixVol
    fixVol = True

    #### alpha/beta/gamma, fixVol=True
    params = np.array(
      [ 0.85543093,  0.05385916,  2.30915485,  1.81397698,  0.68191239,
        0.60301994,  2.51595808,  1.29461596, -0.566967  ,  1.87596517,
        4.90807089,  0.0453925 ,  0.03180923,  4.74765957,  0.61326273,
        0.09474174 ]
    )

    n = len(params)//(3+N+Nexp*(1-fixVol))

    tau_vec = list()
    h_vec   = list()
    ohm_vec = list()
    kwargs  = list()

    for k in range(n):
        alpha = params[(2+N)*k]
        beta  = params[(2+N)*k+1]
        gamma = params[(2+N)*k+2:(2+N)*k+2+N]

        alpha, beta, gamma = hParams(alpha,beta,gamma,zgrid)

        h   = hFunc(alpha,beta,gamma,zgrid)
        ohm = ohmFunc(alpha,beta,gamma,zgrid)

        if not fixVol:
            sig = params[(2+N)*n+Nexp*k:(2+N)*n+Nexp*k+Nexp]
            tau = tauFunc(sig,Texp)
        else:
            tau = tauFunc(sig0,Texp)

        tau_vec.append(tau)
        h_vec.append(h)
        ohm_vec.append(ohm)
        kwargs.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, 'method': 'Loop'})

    a = params[(2+N+Nexp*(1-fixVol))*n:]
    a /= sum(a)

    iv = EnsembleCarrPeltsImpliedVol(K, T, D, F, a, tau_vec, h_vec, ohm_vec, zgrid, kwargs=kwargs)
    df['Fit'] = iv

    print(df.head(20))

    PlotImpliedVol(df, "out/test_spx220603ECP.png", scatterFit=True, ncol=8, atmBar=True, baBar=True)

if __name__ == '__main__':
    test_FitCarrPelts()
    # test_FitEnsembleCarrPelts()
    # test_ImpVolECP()
