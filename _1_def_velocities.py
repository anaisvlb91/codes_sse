# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 06:30:43 2025

@author: alrouaud
"""
import sys
from pathlib import Path
import numpy as np
import lvpyio as lv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
import math
from scipy import signal, datasets, ndimage
from tqdm import tqdm


## DEFINITION DES DONNEES
def BIG(s):
    vector=s[0].as_masked_array()
    h,l=vector.shape
    Tot=np.empty((len(s),3,h,l))
    i=-1
    for frame in s:
        i=i+1
        vector=frame.as_masked_array()
        Tot[i,0]=np.where(vector['u'].mask, np.nan, vector['u'].data)
        Tot[i,1]=np.where(vector['v'].mask, np.nan, vector['v'].data)
        Tot[i,2]=np.where(vector['w'].mask, np.nan, vector['w'].data)
    return Tot

## RECADRAGE
def rotation(T):
    A=[]
    for i in range (len(T[0])):
        if np.isnan(T[:,i]).all()==False:
            A=A+[(np.nanargmin(T[:,i]),i)]
        points_x = [a[1] for a in A]
        points_y = [a[0] for a in A]
    coeffs = np.polyfit(points_x, points_y, 1)
    pente, intercept = coeffs
    theta_rad = np.arctan(pente)
    theta_deg = np.degrees(theta_rad)
    return theta_deg, theta_rad

def clean(v,MAX):
    b=len(v)-1
    q=0
    while v[b,q]<=0.2 or v[b,q]>=MAX :
        b=b-1
        if b <= 0:
            q=q+1
            b=len(v)-1

    d=len(v[0])-1
    while v[b,d]<=0.2 or v[b,d]>=MAX:
        d=d-1
    a=0
    while v[a,d]<=0.2 or v[a,d]>=MAX:
        a=a+1
    c=0
    while v[a,c]<=0.2 or v[a,c]>=MAX:
        c=c+1
    return(a,b,c,d)

def centered(v_clean,a,b):
    center=np.nanargmin(v_clean[1:,5])+a
    height=min(center-a,b-center)
    a=center-height
    b=center+height
    return a,b

def get_PIV_coordinates(frame):
    sc = frame.scales  # scaling from the calibration
    h, w = frame.shape  # height and width, numpy convention
    X_mm = np.array([sc.x.slope * frame.grid.x * i + sc.x.offset for i in range(w)])
    Y_mm = np.array([sc.y.slope * frame.grid.y * j + sc.y.offset for j in range(h)])
    return X_mm, Y_mm

def velocities(Tot,theta,U_mean,frame,MAX):
    U_mean=np.nan_to_num(U_mean)
    U_mean.astype(np.float32) 
    Urot = ndimage.rotate(U_mean, theta,  axes=(1,0), reshape=False, mode='nearest')
    a,b,c,d=clean(Urot,MAX)
    A,B=centered(Urot[a:b,c:d],a,b)
    Sizes=A,B,c,d
    clean_Tot=np.full((len(Tot),3,B-A,d-c),0,dtype=float)
    for time in range (len(Tot)):
        clean_Tot[time,0,:,:]=ndimage.rotate(np.nan_to_num(Tot[time,0,:,:]), theta,   reshape=False, mode='nearest')[A:B,c:d]
        clean_Tot[time,1,:,:]=ndimage.rotate(np.nan_to_num(Tot[time,1,:,:]), theta,   reshape=False, mode='nearest')[A:B,c:d]
        clean_Tot[time,2,:,:]=ndimage.rotate(np.nan_to_num(Tot[time,2,:,:]), theta,  reshape=False, mode='nearest')[A:B,c:d]
    #difinition of U,V and W
    U=clean_Tot[:,0,:,:]
    V=clean_Tot[:,1,:,:]
    W=clean_Tot[:,2,:,:]
    x_mm, y_mm = get_PIV_coordinates(frame)
    x_mm=x_mm[c:d]/math.cos(math.radians(theta))
    y_mm=y_mm[A:B]
    coord=x_mm,y_mm
    return U,V, W,clean_Tot,coord

def velocities2(U,V,W,h,theta,U_mean,x_crop,y_crop,MAX):
    U_mean=np.nan_to_num(U_mean,nan=0.0)
    U_mean.astype(np.float32) 
    Urot = ndimage.rotate(U_mean, theta,  axes=(1,0), reshape=False, mode='nearest')
    a,b,c,d=clean(Urot,MAX)
    A,B=centered(Urot[a:b,c:d],a,b)
    Sizes=A,B,c,d
    clean_U=np.full((len(U),B-A,d-c),0,dtype=float)
    clean_V=np.full((len(U),B-A,d-c),0,dtype=float)
    clean_W=np.full((len(U),B-A,d-c),0,dtype=float)
    clean_h=np.full((len(h),B-A,d-c),0,dtype=float)
    for time in range (len(U)):
        clean_U[time,:,:]=ndimage.rotate(np.nan_to_num(U[time,:,:]), theta,  reshape=False, mode='nearest')[A:B,c:d]
        clean_V[time,:,:]=ndimage.rotate(np.nan_to_num(V[time,:,:]), theta,  reshape=False, mode='nearest')[A:B,c:d]
        clean_W[time,:,:]=ndimage.rotate(np.nan_to_num(W[time,:,:]), theta,  reshape=False, mode='nearest')[A:B,c:d]
    for t in range (len(h)):     
        clean_h[t,:,:]=ndimage.rotate(np.nan_to_num(h[t,:,:]), theta,  reshape=False, mode='nearest')[A:B,c:d]
    f=0
    while clean_h[0,0,f]<=1e-3:
        f=f+1 
    g=-1
    while clean_h[0,-1,g]<=1e-3:
        g=g-1
    clean_U=clean_U[:,:,f+1:g]
    clean_V=clean_V[:,:,f+1:g]
    clean_W=clean_W[:,:,f+1:g]
    clean_h=clean_h[:,:,f+1:g]
    x_mm=x_crop[c+f+1:d+g]/math.cos(math.radians(theta))
    O=np.argmin(np.abs(y_crop))
    m=(B-A-1)//2
    y_mm=y_crop[O-m-1:O+m+1]
    coord=x_mm,y_mm
    return clean_U,clean_V,clean_W,clean_h,coord

#plot
def show_frame(image, X, Y, label='velocity'):
    vmin, vmax = np.nanmin(image),np.nanmax(image) # minimum and maximum value of the colorbar
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap.set_bad(color='white') 
    fig = plt.figure(figsize=(6.2, 4.0))  # define the figure and its size
    ax = fig.add_axes((0.13, 0.10, 0.85, 0.88))  # add an axis to the figure
    ax.set_xlabel(r'$x/D$')  # set label x axis
    ax.set_ylabel(r'$y/D$')  # set label y axis
    ax.grid(False)  # remove grid
    im = plt.pcolormesh(X,Y,image, cmap=cmap, norm=norm)  # plot the data
    cbar = fig.colorbar(im, pad=0.02)  # add a colorbar
    cbar.set_label(label) 
    ax.set_aspect('equal', adjustable='box')
    return fig, ax, im

def SVD(tot):
    t,s,l,L=tot.shape
    tot_flat=tot.reshape(t,s*l,L)
    v_2d = tot_flat.reshape(t, -1).T
    lenght,width=tot_flat[0].shape
    U, S, VT = np.linalg.svd(v_2d,full_matrices=False)
    Uapprox=U[:,:len(tot_flat)]
    return Uapprox,S,VT,(lenght,width)

    
def modes2(image, X, Y,l,L,S,label,exp_case,folder): # echelles differentes pour chacune des images
    name=["u","v","w"]
    lab=label
    for k in range(3):     
        cmap = plt.cm.jet
        cmap.set_bad(color='white') 
        extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)]
        fig, axes = plt.subplots(2,5, figsize=(20, 6), constrained_layout=True)
        plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.2)  
        axes = axes.ravel()
        label=name[k]+lab
        for i in range(10):
            svdU=image[k*L*l:(k+1)*L*l,i].reshape((l,L))
            vmin, vmax = np.nanmin(svdU),np.nanmax(svdU) # minimum and maximum value of the colorbar
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            im = axes[i].imshow(svdU, extent=extent, cmap=cmap, norm=norm, interpolation='bilinear')
            axes[i].set_title(label + " mode " + str(i + 1) + rf", poids : {round((S[i]/np.nansum(S))*100,3)} %")
            axes[i].set_xlabel(r'$x/D$')
            axes[i].set_ylabel(r'$y/D$')
            axes[i].set_aspect('equal', adjustable='box')
            plt.title(exp_case)
            #fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)
        plt.savefig(f"{folder}/{exp_case}/svd_{name[k]}.pdf")
    plt.show()

#energy
def energy(SSVD):
    plt.figure()
    plt.semilogy(SSVD[:-1],"x",markersize=3)
    plt.title('Singular Values')
    plt.xlabel("r")
    plt.ylabel("singular values"+r'$\sigma_r$')


#vorticity
def blurred(v_clean,sigma,radius):
    v_blurred=gaussian_filter(v_clean.astype(np.float32), sigma,radius=radius)
    return v_blurred

def rotz(u,v,x_mm,y_mm):
    dy=np.diff(y_mm)[0].astype(np.float32)
    dx=np.diff(x_mm)[0].astype(np.float32)
    duy=np.gradient(u.astype(np.float32),dy,axis=1).astype(np.float32)
    dvx=np.gradient(v.astype(np.float32),dx,axis=2).astype(np.float32)
    wz=dvx-duy
    return wz,duy,dvx,wz[0].shape


def modes1(tot, X, Y,label): # echelles differentes pour chacune des images
    v_2d = tot.reshape(len(tot), -1).T
    lenght,width=tot[0].shape
    U, S, VT = np.linalg.svd(v_2d,full_matrices=False)
    
    cmap = plt.cm.jet
    cmap.set_bad(color='white') 
    extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)]
    fig, axes = plt.subplots(2,5, figsize=(20, 6), constrained_layout=True)
    plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.2)  
    axes = axes.ravel()
    for i in range(10):
        svdU=np.reshape(U[:,i],(lenght,width))
        vmin, vmax = np.nanmin(svdU),np.nanmax(svdU) # minimum and maximum value of the colorbar
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = axes[i].imshow(svdU, extent=extent, cmap=cmap, norm=norm, interpolation='bilinear')
        axes[i].set_title(label + " mode " + str(i + 1)+ rf", poids : {round((S[i]/np.nansum(S))*100,3)} %")
        axes[i].set_xlabel(r'$x ~ [m]$')
        axes[i].set_ylabel(r'$y ~ [m]$')
        axes[i].set_aspect('equal', adjustable='box')
        fig.colorbar(im, ax=axes[i], orientation='vertical', fraction=0.046, pad=0.04)

# Construction de l’ondelette Mexican Hat 2D = same as Omer's one
def mexican_hat_2d(size, sigma=1.0):
    print(size)
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4,4, size)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    factor = (2-r2) / (sigma**2)
    mh = factor * np.exp(-r2 / (2 * sigma**2))
    return mh

def W_transform(h,wavelet_size,sigma):
    wavelet = mexican_hat_2d(wavelet_size, sigma)
    wavelet_T=[]
    for i in tqdm(range(len(h)), desc="wavelet transform", unit="snapshot"):
        A=h[i]
        A_conv = signal.convolve2d(A, wavelet, mode='same', boundary='symm')
        wavelet_T=wavelet_T+[A_conv]
    wavelet_T = np.array(wavelet_T, dtype=float)
    return wavelet_T
