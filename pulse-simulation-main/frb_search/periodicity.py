from __future__ import division
import sys,os,numpy as np, matplotlib.pyplot as plt, pylab, matplotlib.mlab as mlab, matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.dates as md
import dateutil
import matplotlib
from matplotlib import pyplot as plt
import math
from datetime import datetime
import time
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
#from astropy.stats import LombScargle
from astropy.coordinates import *
from astropy.timeseries import LombScargle
#from gammapy  import *
from astropy import units as au
import astropy.time as at
from astropy.coordinates import SkyCoord, EarthLocation
from scipy.optimize import curve_fit
import argparse

# Define a Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def read_obsinfo(file, det_model):
    data = np.genfromtxt(file)
    years = data[:,0]
    months = data[:,1]
    days = data[:,2]
    hours = data[:,3]
    minutes = data[:,4]
    seconds = data[:,5]
    duration = data[:,8]
    nFRB = data[:,9]
    tel = data[:,6]
    freq = data[:,7]
    dmval = data[:,10]
    obsset= data[:,11]
    starts = []
    det = []
    FRB_loc = SkyCoord('05:31:59','+33:08:50',unit=(au.hourangle,au.deg),equinox='J2000')
    AO_loc = EarthLocation.from_geodetic(lon='-66.7528',lat='18.3464') 
    GBT_loc = EarthLocation.from_geodetic(lon='-79.8398',lat='38.4322') 
    Eff_loc = EarthLocation.from_geodetic(lon='6.882778',lat='50.52472')
    VLA_loc = EarthLocation.from_geodetic(lon='-107.6184',lat='34.0784') 
    Lov_loc = EarthLocation.from_geodetic(lon='-2.3085',lat='53.2367')
    WSRT_loc = EarthLocation.from_geodetic(lon='6.6033',lat='52.91472')
    FAST_loc = EarthLocation.from_geodetic(lon='106.9283',lat='25.6528')
    MeerKAT_loc = EarthLocation.from_geodetic(lon='21.4431',lat='-30.7135')
    dss_43_loc = EarthLocation.from_geodetic(lon='148.965',lat='-35.3708')
    tel_locs = [AO_loc,Eff_loc,GBT_loc,VLA_loc,Lov_loc, FAST_loc, WSRT_loc, MeerKAT_loc, dss_43_loc]
    for i in range(len(years)):
        startstr = '%04i-%02i-%02iT%02i:%02i:%02i'%(years[i],months[i],days[i],
                                                    hours[i],minutes[i],
                                                    seconds[i])
        if det_model == 'binary':
            if nFRB[i]!=0:
                det.append(1)
            else:
                det.append(0)
        if det_model == 'norm_rate':
            if nFRB[i]!=0:
                det.append(nFRB[i])
            else:
                det.append(0)
        start = at.Time(startstr,format='isot',scale='utc',location=tel_locs[int(tel[i])])
        starts.append(start.mjd)
    return starts,det,duration,obsset,tel,nFRB
def read_toas(file):
    data = np.genfromtxt(file)
    toas = data[:,0]
    dataset = data[:,1]
    return toas,dataset

def periodogram(time,y,data_type='obs',plot=True,top_vals=10):
    #Lomb-Scargle for unevenly sampled data from astropy
    if data_type=='obs':
        print("The analysis will be carried for observations")
        frequency, power = LombScargle(time, y,center_data=True,
                                       fit_mean=True).autopower(normalization='psd',nyquist_factor=16)
    elif data_type=='win':
        print("The analysis will be carried for the window")
        frequency, power = LombScargle(time, y,center_data=False,
                                       fit_mean=False).autopower(normalization='psd',nyquist_factor=16)
           
    i=np.argmax(power)
    period_l = float("{:.2f}".format(1./frequency[np.argmax(power)]))
    print("Lomb-Scargle prediction: ",period_l)
    if top_vals is not None:
        print("Top "+str(top_vals)+" periods:")
        for i in range(top_vals):
            idx = (-power).argsort()[:top_vals]
            print(i+1,") ",float("{:.2f}".format(1./frequency[idx[i]]))," Power: ",np.round(power[idx[i]],1))


  
    #plt.savefig('FRB121102_periodogram_Eff.png',dpi=300)
    
    if plot:  
        fig, axs = plt.subplots(1, 2,facecolor='w',figsize=(15,5))
        #Data vizualization
        axs[0].plot(time, y,'ko')
        axs[0].set_title("Data",size=14)
        axs[0].set_xlabel("Time",size=14)
        axs[0].set_ylabel("",size=14)
        axs[0].set_yticks([0])
        
        axs[1].plot(frequency, power,'-',label="Prediction: "+str(1./frequency[i])+" days")
        axs[1].set_title("Lomb-Scargle periodogram",size=14)
        axs[1].set_xlabel("Freq, Hz",size=14)
        axs[1].set_ylabel("Power",size=14)
        
        plt.tight_layout()
        return period_l
    else:
        return period_l,frequency, power
    
def folding(epochs,detection,period):
    A = []
    A_i= []
    B = []
    B_i =[]
    e_pos = []
    e_neg = []
    i=0
    while i<len(epochs):
        phase=epochs[i]/period-int(epochs[i]/period)
        if detection[i]==1:
            A.append(phase)
            A_i.append(i)
            e_pos.append(epochs[i])
        elif detection[i]==0:
            B.append(phase)
            B_i.append(i)
            e_neg.append(epochs[i])
        i+=1
    return A,A_i,e_pos,B,B_i,e_neg


def folding_norm(epochs,detection,period):
    A = []
    A_i= []
    B = []
    B_i =[]
    e_pos = []
    e_neg = []
    i=0
    while i<len(epochs):
        phase=epochs[i]/period-int(epochs[i]/period)
        if detection[i]>0:
            A.append(phase)
            A_i.append(i)
            e_pos.append(epochs[i])
        elif detection[i]==0:
            B.append(phase)
            B_i.append(i)
            e_neg.append(epochs[i])
        i+=1
    return A,A_i,e_pos,B,B_i,e_neg


def phaseshift(array,shift):
    i = 0
    arraymod = []
    while i < len(array):
        a_shifted=array[i]+shift
        if a_shifted < 0:
            a_shifted=1+a_shifted
        elif a_shifted > 1:
            a_shifted=a_shifted-1
        arraymod.append(a_shifted)
        #print array[i],arraymod[i]
        i+=1
    return arraymod

def periodicity(obsinfo,toas, amplitude, mean, sig, data_type='obs',top_cand=10,shift=0,period='auto',plot='all',significance=True, det_model='binary'):
    
    epochs,det,duration,obsset,tel,bursts=read_obsinfo(obsinfo, det_model)
    rate=bursts/duration*24*3600
    if data_type=='obs':
        y=det       #-np.mean(det)
    elif data_type=='win':
        y=np.ones(len(det))
        plot=='all'
    
    if plot=='all':
        period_l=periodogram(epochs,y,data_type=data_type,top_vals=top_cand,plot=True)
    else:
        period_l,F_l,A_l=periodogram(epochs,y,data_type=data_type,top_vals=top_cand,plot=None)
    
    if data_type=='win':
        print("***Window analysis done***")
        return
    
    peaks = np.zeros(10000)  # try 20000
    for i in range(10000):  # original 10000
        y_boostraped = np.random.choice(y, size=len(y), replace=True)
        p,a = LombScargle(epochs,y_boostraped,center_data=True, fit_mean=False).autopower(normalization='psd')
        peaks[i] = a.max()
    sig3, sig2, sig1 = np.percentile(peaks, [99.7, 95.4, 68.2])
    print(sig3,sig2,sig1)
    
    if period=='auto':
        period=period_l
    else:
        period=period
    if det_model == 'binary':
        pos,pos_i,e_pos,neg,neg_i,e_neg = folding(epochs,det,period)
    elif det_model == 'norm_rate':
        pos,pos_i,e_pos,neg,neg_i,e_neg = folding_norm(epochs,det,period)
    bins=15
    ref=np.array([57075])
    ref_arr = ref/period-int(ref/period)
    ref_shifted=phaseshift(ref_arr,shift)
    pos_shifted=phaseshift(pos,shift)
    neg_shifted=phaseshift(neg,shift)
    times, dataset = read_toas(toas)
    if plot=='all':
        fig, axs = plt.subplots(1, 2,facecolor='w',figsize=(18,5))
        s3=axs[1].hist(neg_shifted,bins=bins,label='no detections')
        s3=axs[1].hist(pos_shifted,bins=bins,label='detections')
        for l in range(len(neg_shifted)):
            axs[0].bar(neg[l],duration[neg_i[l]]/3600,color='grey',width=0.01)
           
        
        for j in range(len(pos_shifted)):
            if obsset[pos_i[j]]!=0:
                axs[0].bar(pos[j],duration[pos_i[j]]/3600,color='mediumvioletred',width=0.01)
                for idx,i in enumerate(dataset):
                    if i == obsset[pos_i[j]]:
                        position=abs(times[idx]-e_pos[j])*24
                        axs[0].plot(pos[j],position,marker='*',color='goldenrod')
            else:
                axs[0].bar(pos[j],duration[pos_i[j]]/3600,color='goldenrod',width=0.01)
    
        axs[0].axvline(x=ref_arr[0],color='red',linewidth=1,ls='--')
        
        axs[0].set_ylabel(r'Obs. length [hr]',size=14)
        axs[0].set_ylim(0.,12.)
        axs[1].set_title(r'Phase shifted',size=14)
        axs[1].set_ylabel(r'Obs. count',size=14)
        axs[1].set_xlim(0.,1.)
        axs[1].set_xlabel(r'$\phi$',size=14)
        axs[1].legend(numpoints=1,fontsize=10,loc=2)
       
        

   
    
    elif plot=='paper':
        fig, axs = plt.subplots(2, 1,facecolor='w',figsize=(12,6))
        i=np.argmax(A_l)
        axs[0].axvline(x=period,c='mediumvioletred',linewidth=1.5,ls='--')
        #axs[0].axvline(x=1/period,c='mediumvioletred',linewidth=1.5,ls='--')
        axs[0].axhline(y=sig1,c='k',linewidth=1.5,ls=':')
        axs[0].text(period-0.2*period,sig1,r'$1\sigma$',size=10)
        axs[0].axhline(y=sig2,c='k',linewidth=1.5,ls=':')
        axs[0].text(period-0.2*period,sig2,r'$2\sigma$',size=10)
        axs[0].axhline(y=sig3,c='k',linewidth=1.5,ls=':')
        axs[0].text(period-0.2*period,sig3,r'$3\sigma$',size=10)
        axs[0].plot(1/F_l, A_l,'-')
        #guess1 = [14, 160, 1]
        guess = [amplitude, mean, sig]
        popt, _ = curve_fit(gaussian, 1/F_l, A_l, p0=guess)
        axs[0].plot(1/F_l, gaussian(1/F_l, *popt), label='fit')
        print(f'Parameters for peak: A={popt[0]}, mu={popt[1]}, sigma={popt[2]}')
        #axs[0].plot(F_l, A_l,'-')
        #axs[0].text(period+0.05*period,np.max(A_l)+0.05*np.max(A_l),'P='+str(int(period_l))+' days',size=10,backgroundcolor='1')
        harmonics = [161 / 2, 161/3, 161 / 4] + [n * 161 for n in range(2, 7)]  # Harmonics 161/2, 161/4, and up to 6*161
        for harmonic in harmonics:
            axs[0].axvline(x=harmonic, c='purple', linewidth=1, linestyle='-.', alpha=0.5)  
        axs[0]
        if det_model == 'binary':
            for l in range(len(neg_shifted)):
                axs[1].bar(neg_shifted[l],duration[neg_i[l]]/3600,color='grey',width=0.01)
            
            for j in range(len(pos_shifted)):
                if obsset[pos_i[j]]!=0:
                    axs[1].bar(pos_shifted[j],duration[pos_i[j]]/3600,color='mediumvioletred',width=0.015)
                    for idx,i in enumerate(dataset):
                        if i == obsset[pos_i[j]]:
                            position=abs(times[idx]-e_pos[j])*24
                            axs[1].plot(pos_shifted[j],position,marker='*',color='gold',ms=10)
        elif det_model == 'norm_rate':
            for l in range(len(neg_shifted)):
                axs[1].bar(neg_shifted[l],bursts[neg_i[l]],color='grey',width=0.01)
            
            for j in range(len(pos_shifted)):
                if obsset[pos_i[j]]!=0:
                    axs[1].bar(pos_shifted[j],bursts[pos_i[j]],color='mediumvioletred',width=0.015)
    
        axs[0].set_xlabel(r"Period (days)",size=14)
        axs[0].set_ylabel("Power",size=14)
        axs[0].set_ylim(0.,np.max(A_l)+0.2*np.max(A_l))
        axs[0].set_xscale('log')
        axs[0].set_xlim(5.,1000)
        axs[1].set_ylabel(r'Observation (hours)',size=12)
        axs[1].set_xlabel(r'$\phi$',size=14)
        axs[1].set_xlim(0.,1.)
        axs[1].set_ylim(0.,12.)
        axs[0].tick_params(axis="x", labelsize=14)
        axs[0].tick_params(axis="y", labelsize=14)
        axs[1].tick_params(axis="x", labelsize=14)
        axs[1].tick_params(axis="y", labelsize=14)
        plt.tight_layout()
        
        
  
    active_per = float("{:.2f}".format((np.max(pos_shifted)-np.min(pos_shifted))*100))
    active_days = float("{:.2f}".format((np.max(pos_shifted)-np.min(pos_shifted))*160.52))
    print("Middle phase:",float("{:.2f}".format((np.max(pos_shifted)+np.min(pos_shifted))/2)))
    print("Active window (%): ", active_per)
    print("Active window (days):", active_days)
    print("Epochs in total: ",len(epochs))
    print("Non-detections:",len(neg))
    print("Detections:",len(pos))
    print("Time span (days):",int(np.max(epochs)-np.min(epochs)))
    return


def arg_parser():
    parser = argparse.ArgumentParser(description='Periodicity analysis for FRB121102')
    parser.add_argument('--obsinfo', type=str, help='File with observation info')
    parser.add_argument('--obstimes', type=str, help='File with TOAs')
    parser.add_argument('--data_type', type=str, default='obs', help='Type of data to analyze (obs or win)')
    parser.add_argument('--top_cand', type=int, default=15, help='Number of top candidates to show')
    parser.add_argument('--shift', type=float, default=0, help='Phase shift')
    parser.add_argument('--period', default='auto', help='Period to analyze')
    parser.add_argument('--plot', type=str, default='all', help='Plot type (all or paper)')
    parser.add_argument('--guess', type=float, nargs='+', help='Guess for Gaussian fit, in the form A mu sigma')
    parser.add_argument('--det_model', type=str, default='binary', help='Detection model (binary or norm_rate)')
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    if args.period == 'auto':
        period=args.period
    else:
        period=int(args.period)
    periodicity(args.obsinfo,args.obstimes,amplitude=args.guess[0], mean=args.guess[1], sig=args.guess[2], 
                data_type=args.data_type,top_cand=args.top_cand,shift=args.shift,period=period,plot=args.plot, det_model=args.det_model)
    plt.show()


if __name__ == '__main__':
    main()