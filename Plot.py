# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:16:54 2021

@author: Lab
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import pandas as pd
from PreProcess import _Plot
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-T','--training',
                   default='training_data.csv',
                   help='input training data file name')

parser.add_argument('-P','--plot',
                   default='Fig_02',
                   help='figure to plot')

args = parser.parse_args()

if args.plot == "Fig_01":
    # Read data
    MW, perc, PP_MW, P_MW, _ = _Plot(args.training)
    dates = pd.date_range('2019-01-02', '2021-03-16').strftime("%Y-%m-%d").to_list()
    dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(dates, PP_MW, color='dodgerblue', label='淨尖峰供電能力 (MW)')
    ax.plot(dates, P_MW, color='darkorange', label='尖峰負載 (MW)')
    ax.plot(dates, MW, color='yellowgreen', label='備轉容量 (MW)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator([2]))
    ax.set_xlim(dates[0], dates[-1])
    plt.xticks(rotation=45)
    
    plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
    ax.legend(fontsize=10, loc=4)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Fig_01.png")
    #plt.show()
    
elif args.plot == "Fig_02":
    _, _, _, _, VAL_MW = _Plot(args.training)
    dates = pd.date_range('2021-03-02', '2021-03-15').strftime("%Y-%m-%d").to_list()
    dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
    val_pred = np.load("val_pred.npy")
    
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(dates, VAL_MW, color='dodgerblue', label='Real (MW)')
    ax.plot(dates, val_pred.squeeze(), color='darkorange', label='Pred (MW)')
    ax.xaxis.set_major_locator(MaxNLocator(15)) 
    ax.legend(fontsize=10, loc=4)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("Fig_02.png")
    #plt.show()    
    


    


