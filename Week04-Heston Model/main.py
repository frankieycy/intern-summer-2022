import numpy as np
import pandas as pd
from scipy.stats import norm
from lib.params import *
from lib.pricer import *

def test_CalibrateHES():
    df = pd.read_csv("data/test_spxVol220603.csv").dropna()
    T = df["Texp"]
    k = np.log(df["Strike"]/df["Fwd"]).to_numpy()
    w = 1/(df["Ask"]-df["Bid"]).to_numpy()*norm.pdf(k, scale=0.2)
    iv = df[["Bid","Ask"]]
    x = CalibrateModelToImpliedVolFast(k, T, iv, HestonCharFunc, paramsBCCval, paramsBCCkey, bounds=paramsBCCbnd, w=w, optionType="call", inversionMethod="Bisection_jit", useGlobal=True, curryCharFunc=True, formulaType="COS", optMethod="Evolution")
    x = pd.DataFrame(x.reshape(1,-1), columns=paramsBCCkey)
    x.to_csv("out/test_spx220603HES.csv", index=False)

def test_ImpVolHES():
    cal = pd.read_csv("out/test_spx220603HES.csv")
    df = pd.read_csv("data/test_spxVol220603.csv").dropna()
    Texp = df["Texp"].unique()
    params = cal[paramsBCCkey].iloc[0].to_dict()
    impVolFunc = CharFuncImpliedVol(HestonCharFunc(**params), optionType="call", formulaType="COS")
    dfnew = list()
    for T in Texp:
        dfT = df[df["Texp"]==T].copy()
        k = np.log(dfT["Strike"]/dfT["Fwd"]).to_numpy()
        iv = impVolFunc(k,T)
        dfT["Fit"] = iv
        dfnew.append(dfT)
    dfnew = pd.concat(dfnew)
    PlotImpliedVol(dfnew, "out/test_spx220603HES.png", ncol=8, atmBar=True, baBar=True, fitErr=True)

def test_LargeTimeHestonSmile():
    hesParams = {'vbar': 0.04, 'rho': -0.7, 'eta': 0.1, 'lda': 1}
    params = {"meanRevRate": 1, "correlation": -0.7, "volOfVol": 0.1, "meanVar": 0.04, "currentVar": 0.04}
    ivFunc = CharFuncImpliedVol(HestonCharFunc(**params), optionType="call", FFT=True)
    x = np.arange(-1,1,0.002)
    T = 20
    k = x*T
    sig = ivFunc(k,T)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(x,sig,c='k',s=2)
    plt.xlabel('time-scaled log-strike')
    plt.ylabel('implied volatility')
    plt.title(f'Heston Large-Time Smile: {hesParams}',fontsize=10)
    plt.xlim([-1,1])
    plt.ylim([0.05,0.40])
    fig.tight_layout()
    plt.savefig('out/hesSmile.png')
    plt.close()

if __name__ == '__main__':
    # test_CalibrateHES()
    # test_ImpVolHES()
    test_LargeTimeHestonSmile()
