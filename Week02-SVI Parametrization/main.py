import numpy as np
import pandas as pd
from lib.svi import *

def test_FitArbFreeSimpleSVI():
    df = pd.read_csv(f'data/test_spxVol220603.csv').dropna()
    fit = FitArbFreeSimpleSVIWithSimSeed(df)
    Texp = df["Texp"].unique()
    dfnew = list()
    for t in Texp:
        dfT = df[df["Texp"]==t].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        w = svi(*fit.loc[t])(k)
        dfT["Fit"] = np.sqrt(w/t)
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, 'out/test_arbfree220603.png', ncol=8, atmBar=True, baBar=True, fitErr=True)
    PlotTotalVar(dfnew, 'out/test_arbfree220603w.png', xlim=[-0.5,0.5], ylim=[0,0.02])
    fit.to_csv('out/test_arbfree220603.csv', index=False)

def test_BatchFitArbFreeSimpleSVI():
    dfs = {T: pd.read_csv(f'data/test_spxVol{T}.csv').dropna() for T in ['050509','170424','191220']}
    fits = BatchFitArbFreeSimpleSVI(dfs)
    for T in dfs:
        df,fit = dfs[T],fits[T]
        Texp = df["Texp"].unique()
        dfnew = list()
        for t in Texp:
            dfT = df[df["Texp"]==t].copy()
            k = np.log(dfT["Strike"]/dfT["Fwd"])
            w = svi(*fit.loc[t])(k)
            dfT["Fit"] = np.sqrt(w/t)
            dfnew.append(dfT)
        dfnew = pd.concat(dfnew)
        PlotImpliedVol(dfnew, f'out/test_arbfree{T}.png', ncol=8, atmBar=True, baBar=True, fitErr=True)
        PlotTotalVar(dfnew, f'out/test_arbfree{T}w.png', xlim=[-0.2,0.2], ylim=[0,0.004])
        fit.to_csv(f'out/test_arbfree{T}.csv', index=False)
        print(f'---------------- T={T} ----------------')
        print(fits[T])

def test_LargeTimeFullHestonSmile():
    hesParams = {'vbar': 0.04, 'rho': -0.7, 'eta': 0.1, 'lda': 1}
    vbar = hesParams['vbar']
    rho = hesParams['rho']
    eta = hesParams['eta']
    lda = hesParams['lda']

    x = np.arange(-1,1,0.0002)
    xi = eta**2/(lda*vbar)
    a = rho*eta/lda
    m = -rho*eta/xi
    A = eta*np.sqrt(1-rho**2)
    B = rho*eta*(lda-rho*eta/2)
    C = np.sqrt(eta**2/4+(lda-rho*eta/2)**2)
    T2 = B**2-C**2*xi**2*(x-m)**2
    S2 = A**2+xi**2*(x-m)**2

    iu = (-B/A-np.sqrt((B/A)**2-T2/S2))/A # put
    psi = 1/xi*(np.sqrt(C**2+T2/S2)-lda*(1-a/2-a*iu))
    nu = x*iu + psi
    v = 4*(nu+np.sqrt(nu**2-x**2/4))
    sig = np.sqrt(v)

    # iuc = (-B/A+np.sqrt((B/A)**2-T2/S2))/A # call
    # psic = 1/xi*(np.sqrt(C**2+T2/S2)-lda*(1-a/2-a*iuc))
    # nuc = x*iuc + psic
    # vc = 4*(nuc-np.sqrt(nuc**2-x**2/4))
    # sigc = np.sqrt(vc)

    fig = plt.figure(figsize=(6,4))
    plt.scatter(x,sig,c='black',s=2)
    # plt.scatter(x,sigc,c='darkblue',s=2)
    plt.xlabel('time-scaled log-strike')
    plt.ylabel('implied volatility')
    plt.title(f'Heston Large-Time Smile: {hesParams}',fontsize=10)
    plt.xlim([-0.05,0.05])
    plt.ylim([0.15,0.25])
    fig.tight_layout()
    plt.savefig('out/hesSmile.png')
    plt.close()

def test_LargeTimeFullHestonSmileSVI():
    hesParams = {'vbar': 0.04, 'rho': -0.7, 'eta': 0.1, 'lda': 1}
    vbar = hesParams['vbar']
    rho = hesParams['rho']
    eta = hesParams['eta']
    lda = hesParams['lda']

    x = np.arange(-2,2,0.002)
    xi = eta**2/(lda*vbar)
    a = rho*eta/lda
    m = -rho*eta/xi
    A = eta*np.sqrt(1-rho**2)
    B = rho*eta*(lda-rho*eta/2)
    C = np.sqrt(eta**2/4+(lda-rho*eta/2)**2)
    D = np.sqrt((1/A)**2+(C/B)**2)
    K = np.sqrt(1+(a*A**2/(2*B))/(1-a/2))

    v0 = -lda/xi*(1-a/2)-B/A**2*(x-m)-B*D/xi*np.sqrt(1+(xi/A)**2*(x-m)**2)
    v1 = -lda/xi*(1-a/2)*K-B/A**2*K*(x-m)-B*D/xi/K*np.sqrt(1+(xi/A)**2*(x-m)**2)
    v = 4*(v0-v1)
    # v = 4*(v0-np.sqrt(v0**2-x**2/4))
    sig = np.sqrt(v)

    # K0 = lda/xi*(1-a/2)*K
    # K1 = B/A**2*K
    # K2 = B*D/(xi*K)
    # print(K0**2+K2**2, (lda/xi)**2*(1-a/2)**2+(B*D/xi)**2-m**2/4)
    # print(K1**2+K2**2*(xi/A)**2, (B/A**2)**2+(B*D/xi)**2*(xi/A)**2-1/4)

    fig = plt.figure(figsize=(6,4))
    plt.scatter(x,sig,c='black',s=2)
    plt.xlabel('time-scaled log-strike')
    plt.ylabel('implied volatility')
    plt.title(f'Heston Large-Time Smile: {hesParams}',fontsize=10)
    plt.xlim([-1,1])
    plt.ylim([0.1,0.4])
    fig.tight_layout()
    plt.savefig('out/hesSmile.png')
    plt.close()

def test_LargeTimeFullHestonSmileAnalysis():
    hesParams = {'vbar': 0.04, 'rho': -0.7, 'eta': 0.1, 'lda': 1}
    vbar = hesParams['vbar']
    rho = hesParams['rho']
    eta = hesParams['eta']
    lda = hesParams['lda']

    x = np.linspace(-0.03,0.03,200)
    xi = eta**2/(lda*vbar)
    a = rho*eta/lda
    m = -rho*eta/xi
    A = eta*np.sqrt(1-rho**2)
    B = rho*eta*(lda-rho*eta/2)
    C = np.sqrt(eta**2/4+(lda-rho*eta/2)**2)
    D = np.sqrt((1/A)**2+(C/B)**2)

    v0 = -lda/xi*(1-a/2)-B/A**2*(x-m)-B*D/xi*np.sqrt(1+(xi/A)**2*(x-m)**2)
    v1 = np.sqrt(v0**2-x**2/4)
    v1sq = v0**2-x**2/4

    fig = plt.figure(figsize=(6,4))
    # plt.scatter(x,v0,c='k',s=2)
    # plt.scatter(x,v1,c='r',s=2)
    plt.scatter(x,v1sq,c='k',s=2)
    plt.xlabel('time-scaled log-strike')
    plt.ylabel('implied volatility')
    plt.title(f'Large-Time Heston ATM: {hesParams}',fontsize=10)
    # plt.xlim([-0.05,0.05])
    # plt.ylim([0.1,0.4])
    fig.tight_layout()
    plt.savefig('out/heston.png')
    plt.close()

if __name__ == '__main__':
    # test_FitArbFreeSimpleSVI()
    # test_BatchFitArbFreeSimpleSVI()
    # test_LargeTimeFullHestonSmile()
    # test_LargeTimeFullHestonSmileSVI()
    test_LargeTimeFullHestonSmileAnalysis()
