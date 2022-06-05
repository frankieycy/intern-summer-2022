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

if __name__ == '__main__':
    test_FitArbFreeSimpleSVI()
    # test_BatchFitArbFreeSimpleSVI()
