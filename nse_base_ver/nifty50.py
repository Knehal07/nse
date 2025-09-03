#!/usr/bin/env python
# coding: utf-8

import requests
import json
import math
import pandas as pd
import time
from datetime import datetime
import datetime

# Python program to print
# colored text and background
def strRed(skk):         return "\033[91m {}\033[00m".format(skk)
def strGreen(skk):       return "\033[92m {}\033[00m".format(skk)
def strYellow(skk):      return "\033[93m {}\033[00m".format(skk)
def strLightPurple(skk): return "\033[94m {}\033[00m".format(skk)
def strPurple(skk):      return "\033[95m {}\033[00m".format(skk)
def strCyan(skk):        return "\033[96m {}\033[00m".format(skk)
def strLightGray(skk):   return "\033[97m {}\033[00m".format(skk)
def strBlack(skk):       return "\033[98m {}\033[00m".format(skk)
def strBold(skk):        return "\033[1m {}\033[0m".format(skk)

# Urls for fetching Data
url_oc      = "https://www.nseindia.com/option-chain"
url_nf      = 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
url_indices = "https://www.nseindia.com/api/allIndices"

# Headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/javascript,*/*;q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
}

sess = requests.Session()
cookies = dict()

# Local methods
def set_cookie():
    request = sess.get(url_oc, headers=headers, timeout=5)
    cookies = dict(request.cookies)

def get_data(url):
    set_cookie()
    response = sess.get(url, headers=headers, timeout=5, cookies=cookies)
    if(response.status_code==401):
        set_cookie()
        response = sess.get(url_nf, headers=headers, timeout=5, cookies=cookies)
    if(response.status_code==200):
        return response.text
    return ""

def set_header():
    global nf_ul
    global nf_nearest
    response_text = get_data(url_indices)
    data = json.loads(response_text)
    for index in data["data"]:
        if index["index"]=="NIFTY 50":
            nf_ul = index["last"]
            print("nifty")

def print_oi(url,l1,dict1):
    ceoi = []
    cevolume = []
    ceiv = []
    celtp=[]
    cechangeoi = []
    cechange = []
    cebidqty = []
    cebidprice = []
    ceaskqty = []
    ceaskprice = []
    underlyingvalue = []
    time_list =[]
    peoi = []
    pevolume = []
    peiv = []
    peltp=[]
    pechangeoi = []
    pechange = []
    pebidqty = []
    pebidprice = []
    peaskqty = []
    peaskprice = []
    strikeprice = []
    Expirydate = []

    response_text = get_data(url)
    data = json.loads(response_text)
    currExpiryDate = data["records"]["expiryDates"][l1]
    print("Fetching for expiry:", currExpiryDate)

    for item in data['records']['data']:
      if item["expiryDate"] == str(currExpiryDate):
        ceoi.append(item["CE"]["openInterest"])
        cechangeoi.append(item["CE"]["changeinOpenInterest"])
        cevolume.append(item["CE"]["totalTradedVolume"])
        ceiv.append(item["CE"]["impliedVolatility"])
        celtp.append(item["CE"]["lastPrice"])
        cechange.append(item["CE"]["change"])
        cebidqty.append(item["CE"]["bidQty"])
        cebidprice.append(item["CE"]["bidprice"])
        ceaskprice.append(item["CE"]["askPrice"])
        ceaskqty.append(item["CE"]["askQty"])
        strikeprice.append(item["strikePrice"]) 
        pebidqty.append(item["PE"]["bidQty"])
        pebidprice.append(item["PE"]["bidprice"])
        peaskprice.append(item["PE"]["askPrice"])
        peaskqty.append(item["PE"]["askQty"])
        pechange.append(item["PE"]["change"])
        peltp.append(item["PE"]["lastPrice"])
        peiv.append(item["PE"]["impliedVolatility"])
        pevolume.append(item["PE"]["totalTradedVolume"])
        pechangeoi.append(item["PE"]["changeinOpenInterest"])
        peoi.append(item["PE"]["openInterest"])
        underlyingvalue.append(item["PE"]["underlyingValue"])
        time_list.append(datetime.datetime.now())
        Expirydate.append(currExpiryDate)

    dict1.update({
        'CE_OI': ceoi, 
        'CE_changeoi': cechangeoi,
        'CE_Volume': cevolume,
        'CE_IV': ceiv, 
        'CE_LTP': celtp,
        'CE_change': cechange,
        'CE_bidqty': cebidqty,
        'CE_bidprice': cebidprice,
        'CE_askprice': ceaskprice,
        'CE_askqty': ceaskqty,
        'Strikeprice': strikeprice,
        'PE_bidqty': pebidqty,
        'PE_bidprice': pebidprice,
        'PE_askprice': peaskprice,
        'PE_askqty': peaskqty,
        'PE_change': pechange,
        'PE_LTP': peltp,
        'PE_IV': peiv,
        'PE_Volume': pevolume,
        'PE_changeoi': pechangeoi,
        'PE_OI': peoi ,
        'underlyingvalue' :underlyingvalue ,
        'time':time_list,
        'Expirydate':Expirydate
    })

# -------------------
# Main loop
# -------------------
def run_job():
    dict1 = {'graph': 'image'}
    set_header()
    print_oi(url_nf,0,dict1)
    df = pd.DataFrame(dict1)

    s = pd.Series([
        'Tot',
        df['CE_OI'].sum(axis = 0, skipna = True),
        ' ',
        df['CE_Volume'].sum(axis = 0, skipna = True),
        ' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',
        df['PE_Volume'].sum(axis = 0, skipna = True),
        ' ',
        df['PE_OI'].sum(axis = 0, skipna = True),
        ' ',' ',' '
        ],
        index=['graph','CE_OI', 'CE_changeoi', 'CE_Volume', 'CE_IV', 'CE_LTP', 'CE_change',
           'CE_bidqty', 'CE_bidprice', 'CE_askprice', 'CE_askqty', 'Strikeprice',
           'PE_bidqty', 'PE_bidprice', 'PE_askprice', 'PE_askqty', 'PE_change',
           'PE_LTP', 'PE_IV', 'PE_Volume', 'PE_changeoi', 'PE_OI','underlyingvalue','time','Expirydate'])

    df1 = pd.concat([df,s])
    df1.to_csv('D:\\nse\\nse_base_ver\\optiondata_Test.csv', index=False)
    print("File updated at", datetime.datetime.now())

# Run every 30 sec
if __name__ == "__main__":
    while True:
        try:
            run_job()
        except Exception as e:
            print("Error:", e)
        time.sleep(7)

