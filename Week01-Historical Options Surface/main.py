import numpy as np
import pandas as pd
from lib.option import *

def test_GenIvDatasetSPY():
    df = pd.read_csv('data/test_SPY220414.csv')
    df = StandardizeOptionsChainDataset(df,'2022-04-14')
    ivdf = GenerateImpVolDatasetFromStdDf(df,volCorrection='delta')
    ivdf.to_csv('out/test_spyVol220414.csv',index=False)
    PlotImpliedVol(ivdf,"out/test_spyVol220414.png",ncol=8,baBar=True)

def test_GenIvDatasetQQQ():
    df = pd.read_csv('data/test_QQQ220414.csv')
    df = StandardizeOptionsChainDataset(df,'2022-04-14')
    ivdf = GenerateImpVolDatasetFromStdDf(df,volCorrection='delta')
    ivdf.to_csv('out/test_qqqVol220414.csv',index=False)
    PlotImpliedVol(ivdf,"out/test_qqqVol220414.png",ncol=8,baBar=True)

def test_GenIvDatasetSPX():
    df = pd.read_csv('data/test_SPX220603_cboe.csv')
    df = CleanCboeOptionsChainDataset(df,'data/test_SPX220603.csv')
    df = StandardizeOptionsChainDataset(df,'2022-06-03')
    ivdf = GenerateImpVolDatasetFromStdDf(df,volCorrection='delta')
    ivdf = SimplifyDatasetByPeriod(ivdf,select='latest')
    ivdf.to_csv('out/test_spxVol220603.csv',index=False)
    PlotImpliedVol(ivdf,"out/test_spxVol220603.png",ncol=9,baBar=True)

if __name__ == '__main__':
    # test_GenIvDatasetSPY()
    # test_GenIvDatasetQQQ()
    test_GenIvDatasetSPX()
