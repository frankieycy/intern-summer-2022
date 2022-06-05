import numpy as np
import pandas as pd
from lib.ecp import *

def test_FitCarrPelts():
    df = pd.read_csv("data/test_spxVol170424.csv")
    CP = FitCarrPelts(df,fixVol=True,optMethod='Evolution')
    print(CP)

def test_FitEnsembleCarrPelts():
    df = pd.read_csv("data/test_spxVol170424.csv")
    CP = FitEnsembleCarrPelts(df,fixVol=True,optMethod='Evolution')
    print(CP)

if __name__ == '__main__':
    # test_FitCarrPelts()
    test_FitEnsembleCarrPelts()
