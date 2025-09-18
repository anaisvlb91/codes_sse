# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:29:10 2025

@author: alrouaud
"""

import numpy as np
import lvpyio as lv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math as math
import os
import scipy.io
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit




def drawing(A,r,x_mm,y_mm,sf,label):
    for i in A:
        p,q=i
        plt.plot(r,sf[:,p,q],label="x="+str(round(x_mm[q]*1000,1))+"[mm] y="+str(round(y_mm[p]*1000,1))+"[mm]")
    plt.xlabel(r'$r ~ [m]$')
    plt.ylabel(r'$AutoCorrélation Function ~ \mathrm{m^2.s^{-2}}$')
    plt.title(label)
    plt.legend(loc="upper right", fontsize=8)



def curves(b,E,label,x_mm,y_mm):
    for i in range (b):
        k=-4+i*2
        mid=int(len(y_mm)/2)
        e=[]
        for i in range (len(x_mm)):
            e=e+[E[mid+k,i]]
        plt.plot(x_mm,e,label="y="+str(round(y_mm[mid+k]*1000,1))+"[mm]")
    plt.xlabel(r'$x in [m]$ from the cylinder')
    plt.ylabel(label)
    plt.title(label+" depending on x, along the wake, at different y")
    plt.legend(loc="upper right", fontsize=8)

    
def drawing_log(A,r,sf,M,label,x_mm,y_mm):
    for i in A:
        p,q=i
        plt.loglog(r,sf[:,p,q],label="x="+str(round(x_mm[q]*1000,1))+"[mm] y="+str(round(y_mm[p]*1000,1))+"[mm]")
        e=regression(r,sf,p,q,M,2)
        plt.loglog(r[e[1]:e[2]],2*((e[0]*r[e[1]:e[2]])**(2/3)))
        print("M=",e[1],e[2])
    plt.xlabel(r'$r ~ [m]$')
    plt.ylabel(r'$S^2$')
    plt.title(label)
    plt.legend(loc="lower right", fontsize=8)


def show_frame(image, X, Y, label='velocity'):
    vmin, vmax = np.nanmin(image),np.nanmax(image) # minimum and maximum value of the colorbar
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap.set_bad(color='white') 
    extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)]
    fig = plt.figure(figsize=(6.2, 4.0))  # define the figure and its size
    ax = fig.add_axes((0.13, 0.10, 0.85, 0.88))  # add an axis to the figure
    ax.set_xlabel(r'$x ~ [m]$')  # set label x axis
    ax.set_ylabel(r'$y ~ [m]$')  # set label y axis
    ax.grid(False)  # remove grid
    im = ax.imshow(image, extent=extent, cmap=cmap, norm=norm,interpolation='bilinear')  # plot the data
    cbar = fig.colorbar(im, pad=0.02)  # add a colorbar
    cbar.set_label(label) 
    ax.set_aspect('equal', adjustable='box')
    return fig, ax, im



def wake_lines(Ucenter,b,x_mm,y_mm):
    show_frame(Ucenter,x_mm,y_mm, label='U-velocity')
    for i in range (b):
        k=-4+2*i
        mid=int(len(y_mm)/2)
        y=[y_mm[mid+k]]*len(x_mm)
        plt.plot(x_mm,y,color='black')
    plt.show()
    


## EPSILON ux
def inertial_regression(r,sf,a,b,C):
    xMin = 2
    xMax = len(sf)-8
    sf=sf[:,a,b]
    err=-5
    if np.isnan(np.nanmean(sf))==False  :
        err=(math.log(np.nanmean(sf),10))-1
    error=10**(err)
    slope = np.nanmean(sf[xMin:xMax] / (r[xMin:xMax]**(2/3)))
    fTest = slope * r[xMin:xMax]**(2/3)
    tmpDiff = np.abs(sf[xMin:xMax] - fTest)
    tmpErr = np.sqrt(np.nansum(tmpDiff**2))
    while np.isnan(sf[xMax])and(xMax-xMin) > 3:
        xMax = xMax - 1
    while tmpErr > error and (xMax-xMin) >= 0:  
        if (tmpDiff[-1] > tmpDiff[1]).all():
            xMax = xMax - 1
        else:
            xMin = xMin + 1
        slope =  np.nanmean(sf[xMin:xMax] / (r[xMin:xMax]**(2/3)))
        fTest = slope * r[xMin:xMax]**(2/3)
        tmpDiff =  np.abs(sf[xMin:xMax] - fTest)
        tmpErr = np.sqrt(sum(tmpDiff**2))
    epsilon = (slope / C)**(3/2)
    return epsilon, xMin,xMax

def inertial(sf,r,x_mm,y_mm):
    min=np.full(sf[0].shape,np.nan)
    max=np.full(sf[0].shape,np.nan)
    for i in range (len(y_mm)) :
        for j in range (len(x_mm)) :
            min[i,j]=int(inertial_regression(r,sf,i,j,2)[1])
            max[i,j]=int(inertial_regression(r,sf,i,j,2)[2])
            if min[i, j] < 0 :
                min[i,j]=np.nan
            if max[i, j] > 20 :
                max[i,j]=np.nan
            if max[i, j]<6:
                max[i,j]=np.nan
    xMin=int(np.nanmean(min))+1
    xMax=int(np.nanmean(max))+1
    return xMin,xMax


def regression(r,sf,a,b,M,C):
    xMin=M[0]
    xMax=M[1]
    sf=sf[:,a,b]
    slope = np.nanmean(sf[xMin:xMax] / (r[xMin:xMax]**(2/3)))
    epsilon = (slope / C)**(3/2)
    return epsilon, xMin,xMax

## IINTEGRALE RANGE

#LI normal
def Li(R,x_mm):
    r,y,x=R.shape
    Li=np.full((y,x),np.nan)
    for yi in range (y):
        for xi in range(x):
            Lr=np.full((r),np.nan)
            ri=0
            while R[ri,yi,xi]>0 and ri<r:
                Lr[ri]=R[ri,yi,xi]
                ri=ri+1
            if ri<r:
                mask = ~np.isnan(Lr)  # Crée un masque pour exclure les NaN
                x_cleanr, Lr_cleanr= x_mm[:len(R)][mask],Lr[mask]
                Li[yi,xi]= np.trapz(Lr_cleanr,x_cleanr)
    return Li


#LI fit exponentiel sur toute la longueur
def model(x, a,b):
    return a *np.exp(b*x)
def Li_fit(R,x_mm):
    r,y,x=R.shape
    Li=np.full((y,x),np.nan)
    X = x_mm[:len(R)].astype(np.float32)
    for yi in range (y):
        for xi in range(x):
            if np.sum(np.isnan(R[:,yi,xi]))<=len(R)-2:  
                mask = ~np.isnan(R[:,yi,xi])
                y = R[mask,yi,xi].astype(np.float32)
                a_opt,b_opt=curve_fit(model, X[mask], y,maxfev=5000)[0]
                y_fit = model(X, a_opt,b_opt)
                Li[yi,xi]= np.trapz(y_fit,X)
            else:
               Li[yi,xi]=np.nan
    return Li


## Cesilon 
def Cep(E,L,urms):
    Cep=(E*L)/(urms**3)
    return Cep
  
def Cep_mean(Cepux,x_mm,y_mm):
    Cmean=np.nanmean(Cepux, axis=0)
    plt.plot(x_mm,Cmean,'x',label="y in the wake, mean")
    plt.xlabel(r'$x in [m]$ from the cylinder')
    plt.ylabel(r"$C_{\varepsilon}$")
    plt.title(r"$C_{\varepsilon}$  ux mean on y depending on x, along the wake")
    plt.legend(loc="upper right", fontsize=8)

    


#évhelles de la turbulence
def lambdas(urms,mu,E):
    lambdas=np.sqrt((15*mu*(urms**2))/E)
    return lambdas



def f_ReynoldsL(urms,L,mu):
    R=(urms*L)/mu
    return R


def Kolmogorov_Scale(epsilon, mu):
    # Length
    eta = (mu**3/epsilon)**(1/4)
    # Time
    tau_eta = (mu/epsilon)**(1/2)
    # Velocity
    u_eta = (epsilon*mu)**(1/4)
    # ${{Re_\lambda}}$ #optionel
    Re_eta = eta*u_eta/mu
 
    return eta, tau_eta, u_eta


#Plot lambda, Re et urms
def curves5(b,x,E,label,x_mm,y_mm,A):
    for i in range (b):
        k=-4+2*i
        mid=int(len(y_mm)/2)
        e=[]
        for i in range (len(x_mm)):
            e=e+[E[mid+k,i]]
        plt.plot(x,e,label="y="+str(round(y_mm[A[0][0]],3))+"m")
    plt.xlabel(r'$x in [m]$ from the cylinder')
    plt.ylabel(label)
    plt.title(label)
    plt.legend(loc="upper right", fontsize=8)


#plot nuage de points, des trois grandeurs dépandant de Reynolds


def curves2(x,E,label):
    plt.plot(x,E,"x")
    plt.xlabel(r'${{Re_\lambda}}$')
    plt.ylabel(label)
    plt.title(label)
    plt.legend(loc="upper right", fontsize=8)

#Plot moyenne par bin
def moyenne(E,ReynoldsL,y_mm):
    mid=int(len(y_mm)/2)
    a=int(np.nanmax(ReynoldsL[mid]))+1
    b=int(np.nanmin(ReynoldsL[mid]))
    R=np.full((a-b)+1,np.nan)
    print(R.shape)
    for i in range ((a-b)+1):
        tranche=[]
        for x in range (len(ReynoldsL[mid])):
            if int(np.nanmin(ReynoldsL[mid]))+i<ReynoldsL[mid][x]<int(np.nanmin(ReynoldsL[mid]))+1+i :
                    tranche=tranche+[E[mid][x]]
        M=np.nanmean(tranche)
        R[i]=M
    return R


                    
