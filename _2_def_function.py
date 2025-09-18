# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:21:52 2025

@author: alrouaud
"""

import numpy as np
import math as math

def autocorr_fct_rx(u,lenght):
    max_rx=math.floor(min(len(u[0,0,:])*lenght,len(u[0,:,0])*lenght))
    nt, ny, nx= u.shape
    tmpx = np.full((max_rx+1, ny, nx), fill_value=np.nan,dtype=np.float16)
    tmpx[0,:,:]=np.nanmean(np.multiply(u[:,:, 0:], u[: ,:,:]),axis=0)/np.nanmean(u[:,:,:]**2,axis=0)
    for rx in range ( 1,max_rx ):
        for x in range (nx-1):
            if (x+rx)>(nx-1) :
                uplus=np.nan 
            else :
                uplus=u[:,:, x+rx]
            if x-rx<0:
                umoins=np.nan
            else :
                umoins=u[:,:, x-rx]
            ur=np.nanmean([np.multiply(uplus,u[:,:, x]),np.multiply(umoins,u[:,:, x])],axis=0)
            tmpx[rx,:,x]=np.nanmean(ur,axis=0)/np.nanmean(u[:,:,x]**2,axis=0)
    Rx=tmpx
    return Rx


def autocorr_fct_ry(u,lenght):
    max_ry=math.floor(min(len(u[0,0,:])*lenght,len(u[0,:,0])*lenght))
    nt, ny, nx= u.shape    
    tmpy = np.full((max_ry+1,ny,nx), fill_value=np.nan,dtype=np.float16)
    tmpy[0,:,:]=np.nanmean(np.multiply(u[:,0:, :], u[: ,:,:]),axis=0)/np.nanmean(u[:,:,:]**2,axis=0)
    for ry in range (1, max_ry ):
        for y in range (ny-1):
            if (y+ry)>(ny-1) :
                uplus=np.nan 
            else :
                uplus=u[:, y+ry,:]
            if y-ry<0:
                umoins=np.nan
            else :
                umoins=u[:, y-ry,:]
            ur=np.nanmean([np.multiply(uplus,u[:, y,:]),np.multiply(umoins,u[:,y,:])],axis=0)
            tmpy[ry,y,:]=np.nanmean(ur,axis=0)/np.nanmean(u[:,y,:]**2,axis=0)
    Ry=tmpy
    return Ry

def structure_fct_rx(u,lenght, order=2):
    max_rx=math.floor(min(len(u[0,0,:])*lenght,len(u[0,:,0])*lenght))
    nt, ny, nx= u.shape
    tmpx = np.full([nt,max_rx, ny, nx], fill_value=np.nan,dtype=np.float16)
    for rx in range (1, max_rx ):
        tmpx[: ,rx,:,:-rx]=(u[:,:, rx:] - u[: ,:,:-rx]) ** order
    sfx=np.nanmean(tmpx,axis=0)
    return sfx


def structure_fct_ry(u,lenght, order=2):
    max_ry=math.floor(min(len(u[0,0,:])*lenght,len(u[0,:,0])*lenght))
    nt, ny, nx= u.shape    
    tmpy = np.full([nt,max_ry,ny,nx], fill_value=np.nan,dtype=np.float16)
    for ry in range(1, max_ry ):
        tmpy [: ,ry,:-ry,:] = (u[:, ry:,:] - u[:, :-ry,:]) ** order
    sfy=np.nanmean(tmpy,axis=0)
    return sfy

