import numpy as np
import pandas as pd
from lib.option import *

DATA_DIR = 'data/'
OUT_DIR  = 'out/'

def test_SimplifyDatasetByPeriod():
    df = pd.read_csv(DATA_DIR+'test_SPY220414.csv')
    std = StandardizeOptionsChainDataset(df,'2022-04-14')
    print(std['Texp'].unique())
    sim = SimplifyDatasetByPeriod(std)
    print(sim['Texp'].unique())

def test_GenerateImpVolDatasetFromStdDf():
    df = pd.read_csv(DATA_DIR+'test_SPY220414.csv')
    std = StandardizeOptionsChainDataset(df,'2022-04-14')
    # iv = GenerateImpVolDatasetFromStdDf(std)
    iv = GenerateImpVolDatasetFromStdDf(std,volCorrection='delta')
    iv.to_csv(OUT_DIR+'test_volSPY220414.csv',index=False)
    print(iv.head(50))

def main():
    pass

if __name__ == '__main__':
    # test_SimplifyDatasetByPeriod()
    # test_GenerateImpVolDatasetFromStdDf()
    main()
