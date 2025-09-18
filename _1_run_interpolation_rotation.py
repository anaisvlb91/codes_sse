# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:05:05 2025

@author: alrouaud
"""

import sys
from pathlib import Path
import numpy as np
from skimage import io
from skimage.measure import label, regionprops
import lvpyio as lv
from tqdm import tqdm  

import _1_def_velocities as velo
import _1_def_GIF_3D

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
import math
import h5py
from scipy import signal, datasets, ndimage
from scipy.interpolate import griddata 

import psutil
mem = psutil.virtual_memory()
print(f"Total memory: {mem.total / 1e9:.2f} GB")
print(f"Available memory: {mem.available / 1e9:.2f} GB")
print(f"Used by system: {mem.used / 1e9:.2f} GB")

folder_fig="C:/Users/alrouaud/OneDrive - NTNU/Documents/NTNU experiment/comparaison/figures"
folder_data="F:/CYLINDER/data_comparaison"

def get_CALIB_coordinates(frame):
    sc = frame.scales  # scaling from the calibration
    h, w = frame.shape  # height and width, numpy convention
    X_mm = np.array([sc.x.slope  * i + sc.x.offset for i in range(w)])
    Y_mm = np.array([sc.y.slope  * j + sc.y.offset for j in range(h)])
    return X_mm, Y_mm

pathPIV=Path(r"F:/CYLINDER/Plane10mm_Cylinder_PIV")
pathPROF=Path(r"F:/CYLINDER")
CALIB=lv.read_set(r"F:/CYLINDER/ImageCorrection")
pathsPIV = [p for p in pathPIV.glob(r'S*Cylinder*') if p.is_dir()]
pathsPROF=[p for p in pathPIV.glob(r'S*Cylinder*_Prof.mat') if p.is_dir()]
case_names = [p.name for p in pathsPIV]

##CALIB
frame_CALIB=CALIB[0][0]
x_CALIB,y_CALIB=get_CALIB_coordinates(frame_CALIB)
calibration=CALIB[0].as_masked_array().astype(np.float32)

D=12 #diameter of the cylinder in mm
###if you want to choose the calibration point yourself 
fig, ax, im = velo.show_frame(calibration,x_CALIB,y_CALIB,'Select the 30cm point of the ruler on its down side')    
plt.pause(2)
plt.show()
print('Select the 30cm point of the ruler on its down side')
pt = fig.ginput(n=1)
# Coordinates of the point 
x0_m = np.copy(pt[0][0])
y0_m = np.copy(pt[0][1])
print("point of calibration in mm : (",x0_m,",",y0_m,")")
plt.pause(2)
    
# Look at the final result
fig, ax, im = velo.show_frame(calibration,x_CALIB,y_CALIB,"calibration")
ax.plot(x0_m, y0_m,'ob', marker='x',color="red", lw=0.5, markersize=6)
plt.pause(2)

"""x0_m =  -29.8850828566568 
y0_m = 14.35493338571112 """
print("point of calibration in mm : (",x0_m,",",y0_m,")")

##PROF
name=case_names[0].replace("PIV", "Prof")
with h5py.File(f"{pathPROF}/{name}.mat") as f:
        x_PROF=f['xMesh'][:].astype(np.float32)
        y_PROF=f['yMesh'][:].astype(np.float32)
print("access to PROF data : done")


#PIV
path_dat = pathPIV / case_names[0] /'StereoPIV_MPd(2x64x64_50%ov)'
s=lv.read_set(path_dat)
frame_PIV=s[0][0]
x_PIV, y_PIV = velo.get_PIV_coordinates(frame_PIV)
x_PIV, y_PIV=x_PIV.astype(np.float32), y_PIV.astype(np.float32)    


argX0=np.argmin((x_PIV-x0_m)**2)
argY0=np.argmin((y_PIV-y0_m)**2)
dx=np.nanmean(np.gradient(x_PIV))
dy=np.nanmean(np.gradient(y_PIV))
r=D/2
X0=np.arange(-dx*argX0+300+r,dx*(len(x_PIV)-argX0)+300+r,dx)
print("comparaison : ", X0[argX0])
Y0=np.arange(-dy*argY0,dy*(len(y_PIV)-argY0),dy)
print("comparaison : ", Y0[argY0])


##INTERPOLATION GRID
points = np.column_stack((x_PROF.flatten(), (y_PROF).flatten()))
gridx,gridy=np.meshgrid(x_PIV, y_PIV)
del x_CALIB
del y_CALIB
del CALIB
del calibration
del frame_CALIB

print("lets go")

for exp_case in case_names:
    # Path to save the figures, I usually make one folder for each case to save the figures
    path_save_fig = Path(f"{folder_fig}/{exp_case}")  # the folder for the figures has the name of the experimental case
    path_save_fig.mkdir(parents=True, exist_ok=True)  # this creates the folder if it doesnt exist
    path_save_fig = Path(f"{folder_data}/{exp_case}")  # the folder for the figures has the name of the experimental case
    path_save_fig.mkdir(parents=True, exist_ok=True)  # this creates the folder if it doesnt exist
    print(exp_case)
    plt.close('all')
    ##PIV
    path_dat = pathPIV / exp_case /'StereoPIV_MPd(2x64x64_50%ov)'
    s=lv.read_set(path_dat)
    #separation u,v,w
    Tot=velo.BIG(s)
    rawU=Tot[:,0,:,:].astype(np.float32)
    rawV=Tot[:,1,:,:].astype(np.float32)
    rawW=Tot[:,2,:,:].astype(np.float32)
    del Tot
    
    ##PROF
    name=exp_case.replace("PIV", "Prof")
    with h5py.File(f"{pathPROF}/{name}.mat") as f:
        elevation = f['surfData'][:].astype(np.float32)
    print("access to PROF data : done")
    

    
    ##INTERPOLATION         
    def interpolate_layer(i):
        return griddata(points, elevation[i].flatten(), (gridx, gridy), method='linear').astype(np.float32)
    from joblib import Parallel, delayed
    # Exécution parallèle sur tous les cœurs dispo
    ITP_list = Parallel(n_jobs=-1, verbose=5)(delayed(interpolate_layer)(i) for i in tqdm(range(len(elevation))))
    # Assemblage en tableau final
    ITP_elevation = np.stack(ITP_list)
    del ITP_list
    del elevation

    """interpolates then rotates PROF & PIV together before computing SVD, GIV and wavelet transform"""

    plt.close('all')
    Uvelocity=np.nanmean(rawU,axis=0)
    U_mean=np.where(Uvelocity>0.1,Uvelocity,np.nan)
    if r"TimeResolved" not in exp_case:
        theta,THETArad=velo.rotation(U_mean)
    else: 
        name=exp_case.replace("TimeResolved", "FS1Hz")
        theta=np.load(f"{folder_data}/{name}/rotated_cleaned_data.npz")["theta"]
    print(theta)
    #rotation and cleaning
    if "VFD10" in exp_case:
        MAX=0.25
    else:
        MAX=0.4
    U_prime,V_prime,W,h,coord=velo.velocities2(rawU,rawV,rawW,ITP_elevation,theta,U_mean,X0,Y0,MAX) 
    U=math.cos(theta)*U_prime+math.sin(theta)*V_prime
    V=math.sin(-theta)*U_prime+math.cos(theta)*V_prime
    np.savez_compressed(f"{folder_data}/{exp_case}/rotated_cleaned_data.npz",U=U,V=V,W=W,h=h,theta=theta)
    del ITP_elevation
    del rawU
    del rawV
    del rawW
    
    #plot of the mean velocity
    x_mm, y_mm = coord
    np.savez_compressed(f"{folder_data}/{exp_case}/frame_cropped_rotated.npz",x_mm=x_mm,y_mm=y_mm)
    
    Umean, Vmean, Wmean = (np.nanmean(arr, axis=0) for arr in [U, V, W])
    velo.show_frame(Umean,x_mm/D,y_mm/D,"mean U velocity")
    plt.savefig(f"{folder_fig}/{exp_case}/mean_velocity_U.pdf")
    plt.show()
    velo.show_frame(Vmean,x_mm/D,y_mm/D, "mean V velocity")
    plt.savefig(f"{folder_fig}/{exp_case}/mean_velocity_V.pdf")
    plt.show()
    u, v, w = (arr - mean for arr, mean in zip([U, V, W], [Umean, Vmean, Wmean]))
    np.savez_compressed(f"{folder_data}/{exp_case}/fluctuating_velocities.npz",u=u,v=v,w=w)
    del U
    del V
    del W
    
    #Vorticity
    omegaZ=velo.rotz(velo.blurred(u.astype(np.float32),1,1),velo.blurred(v.astype(np.float32),1,1),x_mm,y_mm)[0]
    mean_t_wz=np.nanmean(omegaZ**2,0)

    velo.show_frame(mean_t_wz.astype(np.float32),x_mm/D,y_mm/D, label='fluctuating mean t wz**2')
    plt.savefig(f"{folder_fig}/{exp_case}/mean_vorticity_squared.pdf")
    plt.show()
  
    
    #SVD
    if r"TimeResolved" not in exp_case:
        tot=np.stack([u.astype(np.float32),v.astype(np.float32),w.astype(np.float32)], axis=1)
        #SVD
        t,s,l,L=tot.shape
        SVDtot=velo.SVD(tot.astype(np.float32))
        #U
        USVDu=np.reshape(SVDtot[0][:L*l,:],(t,l,L))
        USVDv=np.reshape(SVDtot[0][L*l:2*L*l,:],(t,l,L))
        USVDw=np.reshape(SVDtot[0][2*L*l:3*L*l,:],(t,l,L))
        #S
        SSVD=SVDtot[1]
        #VT
        VTSVD=SVDtot[2]
        
        velo.modes2(SVDtot[0],x_mm/D,y_mm/D,l,L,SSVD,"tot",exp_case,folder_fig)
        
        velo.energy(SSVD)
        plt.savefig(f"{folder_fig}/{exp_case}/singular_values.pdf")
        plt.show()
        
        im=velo.modes1(omegaZ, x_mm/D,y_mm/D,"wz")
        plt.savefig(f"{folder_fig}/{exp_case}/vorticity_modes.pdf")
        plt.show()
    
        velo.modes1(h,x_mm/D,y_mm/D,'profilométrie SVD')
        plt.savefig(f"{folder_fig}/{exp_case}/profilometry_SVD.pdf")
        plt.show()
    
    zmin=np.nanmin(h)
    zmax=np.nanmax(h)
    gridx,gridy=np.meshgrid(x_mm, y_mm)
    _1_def_GIF_3D.gif(h, gridx,gridy, "surface elevation",zmin, zmax,exp_case,folder_fig)

    ##GRADIENT
    dx=np.nanmean(np.diff(x_mm))
    dhx=np.gradient(h,dx,axis=0)
    dy=np.diff(y_mm)[0]
    dhy=np.gradient(h,dy,axis=1)
    np.savez_compressed(f"{folder_data}/{exp_case}/gradient_elevation.npz",dhx=dhx, dhy=dhy)


    # Paramètres de l’ondelette
    wavelet_size = 13
    sigma =1.0
    wavelet_T=velo.W_transform(h,wavelet_size,sigma)
    np.savez_compressed(f"{folder_data}/{exp_case}/wavelet_transform.npz",wavelet_T=wavelet_T)
