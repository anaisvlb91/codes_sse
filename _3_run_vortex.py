# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:18:10 2025

@author: alrouaud
"""
import _3_def_vortex as vor
import numpy as np
import matplotlib.pyplot as plt

folder_fig="C:/Users/alrouaud/OneDrive - NTNU/Documents/NTNU experiment/comparaison/figures"
folder_data="F:/CYLINDER/data_comparaison"
case_name=["SS_VFD10p6Hz_Cylinder_TimeResolved_PIV"]


D=12 #cylinder diameter
R=6 #radius of search


for exp_case in case_name:

    frame=np.load(f"{folder_data}/{exp_case}/frame_cropped_rotated.npz",allow_pickle=True)
    x_mm=frame["x_mm"]
    y_mm=frame["y_mm"]
    
    dx=x_mm[1]-x_mm[0]
    dy=y_mm[1]-y_mm[0]
    data=np.load(f"{folder_data}/{exp_case}/fluctuating_velocities.npz",allow_pickle=True)
    u=data["u"]*1000  #everything in mm
    v=data["v"]*1000
    
    U=np.load(f"{folder_data}/{exp_case}/rotated_cleaned_data.npz",allow_pickle=True)["U"]*1000
    Uvelocity=np.nanmean(U)
    V=np.load(f"{folder_data}/{exp_case}/rotated_cleaned_data.npz",allow_pickle=True)["V"]*1000

    Lb=vor.lambda2(vor.blurred(u,1,1),vor.blurred(v,1,1),dx,dy)
    Lb_=(Lb*(D**2))/(Uvelocity**2)
    Tb,yb=vor.area_pourcentage2(Lb_,500,0.45,r"$\frac{\lambda_{thr} D^2}{U_0^2}$ , blurred")
    THRb=((Tb*(Uvelocity**2))/(D**2))
    np.savez_compressed(f"{folder_data}/{exp_case}/lambda.npz",Lb=Lb,THRb=THRb) 
    
    du=np.nanmean(U,axis=0)*(1/15)
    dv=np.nanmean(V,axis=0)*(1/15)  
    vortexb=vor.detection(Lb,THRb,R,du,dv,x_mm, y_mm)
    vortex_PIV = {k: v for k, v in vortexb.items() if len(v) > 3} #only keeping those which last at least 3 frames
    np.savez_compressed(f"{folder_data}/{exp_case}/vortex_PIV_loin.npz",vortex_PIV=vortex_PIV) 
    
    h=np.load(f"{folder_data}/{exp_case}/rotated_cleaned_data.npz",allow_pickle=True)["h"]
    W=vor.W_transform(h,13,1)
    T2,y=vor.area_pourcentage2(W,500,1,"wavelet of the surface elevation")
    
    du=np.nanmean(U,axis=0)*(1/45)
    dv=np.nanmean(V,axis=0)*(1/45) 
    vortex=vor.detection2(W,T2,R,du,dv,x_mm,y_mm)
    vortex_PROF = {k: v for k, v in vortex.items() if len(v) > 9} #only keeping those which last at least 3 frames
    np.savez_compressed(f"{folder_data}/{exp_case}/dico_vortex_PROF.npz",vortex_PROF=vortex_PROF)
    
    vor.PDF2([len(v) for k, v in vortex_PIV.items()],20,1/15,"durée de vie des vortex blurred PIV")
    plt.savefig(f"{folder_fig}/{exp_case}/pdf_vortex_blurred_timelength_PIV.pdf")
    plt.show()
    vor.PDF2([len(v) for k, v in vortex_PROF.items()],60,1/45,"durée de vie des vortex PROF")
    plt.savefig(f"{folder_fig}/{exp_case}/pdf_vortex_timelength_PROF.pdf")
    plt.show()


    print("starting the GIF")
    vor.gif_vortex(Lb,THRb,R,vortex_PROF,vortex_PIV,x_mm,y_mm,folder_fig,"gif comparaison")
    print("GIF is finished")