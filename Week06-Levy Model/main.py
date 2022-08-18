import numpy as np
import pandas as pd
from lib.pricer import *

def test_LargeTimeVGSmileApprox():
    params = {"sigma": 0.12, "theta": -0.14, "nu": 0.17}
    sig = params['sigma']
    tht = params['theta']
    nu = params['nu']

    x = np.arange(-2,2,0.0002)
    a = sig**2*nu/2
    b = 1-tht*nu/2-sig**2*nu/8
    xi = nu*(tht+sig**2/2)
    S = nu*x-np.log(1-xi)

    iu = -(1/S+xi/(2*a))-np.sqrt((1/S+xi/(2*a))**2-(xi/S-b)/a)
    psi = (np.log(b-xi*iu-a*iu**2)-(iu+0.5)*np.log(1-xi))/nu
    vv = iu*x+psi
    v = 4*(vv+np.sqrt(vv**2-x**2/4))
    vol = np.sqrt(v)

    # iuc = -(1/S+xi/(2*a))+np.sqrt((1/S+xi/(2*a))**2-(xi/S-b)/a)
    # psic = (np.log(b-xi*iuc-a*iuc**2)-(iuc+0.5)*np.log(1-xi))/nu
    # vvc = iuc*x+psic
    # vc = 4*(vvc-np.sqrt(vvc**2-x**2/4))
    # volc = np.sqrt(vc)

    fig = plt.figure(figsize=(6,4))
    plt.scatter(x,vol,c='black',s=2)
    # plt.scatter(x,volc,c='darkblue',s=2)
    plt.xlabel('time-scaled log-strike')
    plt.ylabel('implied volatility')
    plt.title(f'VG Smile: {params}',fontsize=10)
    plt.xlim([-0.05,0.05])
    plt.ylim([0.12,0.14])
    fig.tight_layout()
    plt.savefig('out/vgSmile.png')
    plt.close()

def test_LargeTimeVGSmile():
    vgParams = {"vol": 0.12, "drift": -0.14, "timeChgVar": 0.17}
    params = {"sigma": 0.12, "theta": -0.14, "nu": 0.17}
    ivFunc = CharFuncImpliedVol(VarianceGammaCharFunc(**vgParams), optionType="call", FFT=True)
    x = np.arange(-2,2,0.002)
    T = 1
    k = x*T
    vol = ivFunc(k,T)
    fig = plt.figure(figsize=(6,4))
    plt.scatter(x,vol,c='k',s=2)
    plt.xlabel('time-scaled log-strike')
    plt.ylabel('implied volatility')
    plt.title(f'VG Smile $T={T}$: {params}',fontsize=10)
    plt.xlim([-1,1])
    plt.ylim([0.1,0.25])
    fig.tight_layout()
    plt.savefig('out/vgSmile-act.png')
    plt.close()

if __name__ == '__main__':
    test_LargeTimeVGSmileApprox()
    # test_LargeTimeVGSmile()
