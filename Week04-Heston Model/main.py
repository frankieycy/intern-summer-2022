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

if __name__ == '__main__':
    test_CalibrateHES()
    test_ImpVolHES()
