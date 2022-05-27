import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from dateutil import parser
from scipy.special import ndtr
from scipy.optimize import minimize

#### Black-Scholes #############################################################

def BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black-Scholes formula for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    discountFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol

    price = np.where(optionType == "call",
        spotPrice * ndtr(d1) - discountFactor * strike * ndtr(d2),
        discountFactor * strike * ndtr(-d2) - spotPrice * ndtr(-d1))
    return price

def BlackScholesDelta(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black-Scholes delta for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    return np.where(optionType == "call", ndtr(d1), -ndtr(-d1))

def BlackScholesImpliedVol(spotPrice, strike, maturity, riskFreeRate, priceMkt, optionType="OTM"):
    # Black-Scholes implied volatility for call/put/OTM
    forwardPrice = spotPrice*np.exp(riskFreeRate*maturity)
    impVol = np.zeros(len(strike))

    if optionType == "OTM":
        optionType = np.where(strike > forwardPrice, "call", "put")

    impVol0 = np.repeat(1e-10, len(strike))
    impVol1 = np.repeat(10., len(strike))
    price0 = BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impVol0, optionType)
    price1 = BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impVol1, optionType)

    while np.mean(impVol1-impVol0) > 1e-10:
        impVol = (impVol0+impVol1)/2
        price = BlackScholesFormula(spotPrice, strike, maturity, riskFreeRate, impVol, optionType)
        price0 += (price<priceMkt)*(price-price0)
        impVol0 += (price<priceMkt)*(impVol-impVol0)
        price1 += (price>=priceMkt)*(price-price1)
        impVol1 += (price>=priceMkt)*(impVol-impVol1)
    return impVol

#### Options Chain #############################################################

def StandardizeOptionsChainDataset(df, onDate):
    # Standardize options chain dataset
    onDate = parser.parse(onDate)
    cols = ["Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"]
    getDateFromContractName = lambda n: re.match(r'([a-z.]+)(\d+)([a-z])(\d+)',n,re.I).groups()[1]
    df["Expiry"] = pd.to_datetime(df["Contract Name"].apply(getDateFromContractName),format='%y%m%d')
    df["Texp"] = (df["Expiry"]-onDate).dt.days/365.25
    df = df[df['Texp']>0][cols].reset_index(drop=True)
    return df

def SimplifyDatasetByPeriod(df, period='month', select='earliest'):
    # Simplify options chain dataset according to specified period
    Tdict = dict()
    Texp = df['Texp'].unique()
    for T in Texp:
        if period == 'day':
            T0 = np.ceil(T*365.25).astype('int') # Almost never used
        elif period == 'month':
            T0 = np.ceil(T*12).astype('int')
        elif period == 'year':
            T0 = np.ceil(T).astype('int')
        if T0 in Tdict:
            Tdict[T0] += [T]
        else:
            Tdict[T0] = [T]
    # print(Tdict)
    dfnew = list()
    for T0 in Tdict:
        if select == 'earliest':
            dfnew.append(df[df['Texp']==Tdict[T0][0]])
        elif select == 'latest':
            dfnew.append(df[df['Texp']==Tdict[T0][-1]])
    dfnew = pd.concat(dfnew)
    return dfnew

#### Implied Vol Dataset #######################################################

def GenerateImpVolDatasetFromStdDf(df, Nntm=6, volCorrection=None):
    # Generate implied vol dataset from standardized options chain df
    # Fi,PVi are implied from put-call parity, for each maturity i
    # Columns: "Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask" ("Fwd","PV")
    # Output: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    Texp = df['Texp'].unique()
    ivdf = list()
    for T in Texp:
        dfT = df[df['Texp']==T].dropna()
        dfTc = dfT[dfT['Put/Call']=='Call']
        dfTp = dfT[dfT['Put/Call']=='Put']
        expiry = dfT['Expiry'].iloc[0]
        Kc = dfTc['Strike']
        Kp = dfTp['Strike']
        K = Kc[Kc.isin(Kp)] # common strikes
        dfTc = dfTc[Kc.isin(K)]
        dfTp = dfTp[Kp.isin(K)]
        dfTc['Mid'] = (dfTc['Bid']+dfTc['Ask'])/2 # add Mid col
        dfTp['Mid'] = (dfTp['Bid']+dfTp['Ask'])/2
        if len(K) >= Nntm:
            K = K.to_numpy()
            bidc = dfTc['Bid'].to_numpy()
            bidp = dfTp['Bid'].to_numpy()
            askc = dfTc['Ask'].to_numpy()
            askp = dfTp['Ask'].to_numpy()
            if "Fwd" in dfT and "PV" in dfT: # American-style (via de-Americanization)
                F = dfT['Fwd'].iloc[0]
                PV = dfT['PV'].iloc[0]
            else: # European-style (via put-call parity)
                midc = dfTc['Mid'].to_numpy()
                midp = dfTp['Mid'].to_numpy()
                mids = midc-midp # put-call spread
                Kntm = K[np.argsort(np.abs(midc-midp))[:Nntm]] # ntm strikes
                i = np.isin(K,Kntm)
                def objective(params):
                    F,PV = params
                    return sum((mids[i]-PV*(F-K[i]))**2)
                opt = minimize(objective,x0=(np.mean(mids[i]+K[i]),1))
                F,PV = opt.x
            print(f"T={expiry.date()} F={F} PV={PV}")
            ivcb = BlackScholesImpliedVol(F,K,T,0,bidc/PV,"call")
            ivca = BlackScholesImpliedVol(F,K,T,0,askc/PV,"call")
            ivpb = BlackScholesImpliedVol(F,K,T,0,bidp/PV,"put")
            ivpa = BlackScholesImpliedVol(F,K,T,0,askp/PV,"put")
            if volCorrection == "delta": # wrong-spot correction
                dcb = BlackScholesDelta(F,K,T,0,ivcb,"call")
                dca = BlackScholesDelta(F,K,T,0,ivca,"call")
                dpb = BlackScholesDelta(F,K,T,0,ivpb,"put")
                dpa = BlackScholesDelta(F,K,T,0,ivpa,"put")
                ivb = (dcb*ivpb-dpb*ivcb)/(dcb-dpb)
                iva = (dca*ivpa-dpa*ivca)/(dca-dpa)
            else: # otm imp vols
                ivb = ivpb*(K<=F)+ivcb*(K>F)
                iva = ivpa*(K<=F)+ivca*(K>F)
            ivb[ivb<1e-8] = np.nan
            iva[iva<1e-8] = np.nan
            callb = BlackScholesFormula(F,K,T,0,ivb,"call")
            calla = BlackScholesFormula(F,K,T,0,iva,"call")
            callm = (callb+calla)/2
            ivdfT = pd.DataFrame({
                "Expiry":   expiry,
                "Texp":     T,
                "Strike":   K,
                "Bid":      ivb,
                "Ask":      iva,
                "Fwd":      F,
                "CallMid":  callm,
                "PV":       PV,
            })
            ivdf.append(ivdfT)
    ivdf = pd.concat(ivdf)
    return ivdf
