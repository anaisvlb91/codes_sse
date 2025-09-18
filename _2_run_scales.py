# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 13:34:00 2025

@author: alrouaud
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 09:33:08 2025

@author: alrouaud
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:51:36 2025

@author: alrouaud
"""
import _2_def_function as fct
import _2_def_scales as sc


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
import math
import lvpyio as lv

folder_data="F:/CYLINDER/data_comparaison"
folder_fig="C:/Users/alrouaud/OneDrive - NTNU/Documents/NTNU experiment/comparaison/figures"

path=r"F:\CYLINDER\Plane10mm_Cylinder_PIV"
path = Path(path)
paths = [p for p in path.glob(r'S*1Hz*') if p.is_dir()]  ## only statistic cases
case_names = [p.name for p in paths]

for exp_case in case_names:
    print(exp_case)
    path_save_fig = Path(f"{folder_fig}/{exp_case}")  # the folder for the figures has the name of the experimental case
    path_save_fig.mkdir(parents=True, exist_ok=True)  # this creates the folder if it doesnt exist
    path_save_fig = Path(f"{folder_data}/{exp_case}")  # the folder for the figures has the name of the experimental case
    path_save_fig.mkdir(parents=True, exist_ok=True)  # this creates the folder if it doesnt exist
    
    
    X0=np.load(f"{folder_data}/{exp_case}/frame_cropped_rotated.npz")["x_mm"]*0.001
    Y0=np.load(f"{folder_data}/{exp_case}/frame_cropped_rotated.npz")["y_mm"]*0.001

    
    data=np.load(f"{folder_data}/{exp_case}/fluctuating_velocities.npz")
    u=data["u"]
    v=data["v"]
    U=np.load(f"{folder_data}/{exp_case}/rotated_cleaned_data.npz")["U"]
    Uvelocity=np.nanmean(U,axis=0)
    
    mid=int(len(Y0)/2)
    mu=1.002e-6
    A=[(mid,10),(mid,50),(mid,100),(mid,130),(mid-5,25),(mid+5,25),(mid-10,75),(mid+10,75),(mid-5,125),(mid+5,125)] ## selected points in the middle of the wake 

    Rux=fct.autocorr_fct_rx(u,1)
    print("done")
    np.savez_compressed(f"{folder_data}/{exp_case}/autocorr_fct_completed_ux_r_rotated.npz",Rux=Rux)
    print("enregistré")
        
    sfux=fct.structure_fct_rx(u,1, order=2)
    print("done")
    np.savez_compressed(f"{folder_data}/{exp_case}/structure_fct_ux_r05_rotated.npz",sfux=sfux)
    print("enregistré")
    
    dx=(X0[1]-X0[0])
    r=[]
    for j in range (len(Rux)):
        r=r+[dx*(j+1)]
    r = np.array(r)
    
    sc.drawing(A,r,X0,Y0,Rux,"Rux")
    plt.savefig(f"{folder_fig}/{exp_case}/Rux.pdf")
    plt.show()
    plt.close("all")
        
    points_x=[]
    points_y=[]
    for a in A:
        points_y=points_y+[Y0[a[0]]]
        points_x=points_x+[X0[a[1]]]
    sc.show_frame(np.nanmean(U,axis=0),X0,Y0)
    plt.scatter(points_x, points_y, color='black', marker='x', s=100, label="Points")
    plt.savefig(f"{folder_fig}/{exp_case}/crosses.pdf")
    plt.show()
    plt.close("all")
    
    dx=(X0[1]-X0[0])
    r=[]
    for j in range (len(sfux)):
        r=r+[dx*(j+1)]
    r = np.array(r)
    
    sfuxcenter=sfux
    f=0
    MIN=np.nanmin(Uvelocity)
    for k in sfuxcenter:
        for i in range (len(Uvelocity)):
            for j in range (len(Uvelocity[0])):
                if Uvelocity[i,j]>MIN+0.025:
                    k[i,j]=np.nan
    M=sc.inertial(sfuxcenter,r,X0,Y0)
    print("M=",M)
    
    E=np.empty((len(Y0)+2,len(X0)))
    for i in range (len(Y0)+2):
        for j in range (len(X0)):
            E[i,j]=sc.regression(r, sfux, i,j,M,2)[0]
    np.savez_compressed(f"{folder_data}//{exp_case}/epsilon_dissipation_rate.npz",E=E)
            
    sc.drawing_log(A,r,sfux,M,f"sfux,{exp_case}",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/sfux.pdf")
    plt.show()
    plt.close("all")

    sc.show_frame(E, X0, Y0, label=rf'${{\varepsilon}}$,{exp_case}')
    plt.savefig(f"{folder_fig}/{exp_case}/epsilon_in_space.pdf")
    plt.show()
    plt.close("all")

    sc.curves(5,E,r'${\varepsilon}$',X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/epsilon_curves.pdf")
    plt.show()
    plt.close("all")
    
    Liux=sc.Li(Rux,X0)
    Liux_fit=sc.Li_fit(Rux,X0)
    np.savez_compressed(f"{folder_data}/{exp_case}/Liux_integrale_scale.npz",Liux=Liux,Liux_fit=Liux_fit)
    sc.curves(5, Liux, f"Liux,{exp_case}",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/Liux.pdf")
    plt.show()
    sc.curves(5, Liux_fit,f" Liux fit,{exp_case}",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/liux_fit.pdf")
    plt.show()
    plt.close("all")
    
    rms_u=np.nanstd(u,axis=0)
    
    Cepux=sc.Cep(E,Liux,rms_u)
    Cepux_fit=sc.Cep(E,Liux_fit,rms_u)
    np.savez_compressed(f"{folder_data}/{exp_case}/Cepux_proportionality_scale_coeff.npz",Cepux=Cepux,Cepux_fit=Cepux_fit)
    sc.curves(5,Cepux, r"$C_{\varepsilon}$ ux",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux.pdf")
    plt.show()  
    plt.close("all")
    sc.curves(5,Cepux_fit,r"$C_{\varepsilon}$ ux fit",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_fit.pdf")
    plt.show()
    plt.close("all")
    
    sc.wake_lines(np.nanmean(U,axis=0),5,X0,Y0)
    
    sc.Cep_mean(Cepux,X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_mean.pdf")
    plt.show()
    plt.close("all")

    sc.Cep_mean(Cepux_fit,X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_fit_mean.pdf")
    plt.show()
    plt.close("all")
    
    L=sc.lambdas(rms_u,mu,E)
    np.savez_compressed(f"{folder_data}/{exp_case}/Lambda_taylor_scale.npz",L=L)
    ReynoldsL=sc.f_ReynoldsL(rms_u,L,mu)
    np.savez_compressed(f"{folder_data}/{exp_case}/Reynolds_number_lambda.npz",ReynoldsL=ReynoldsL)
    eta, tau_eta, u_eta=sc.Kolmogorov_Scale(E, mu)
    np.savez_compressed(f"{folder_data}/{exp_case}/Kolmogorov_scales.npz",eta=eta,tau_eta=tau_eta,u_eta=u_eta)
    sc.curves(5,eta,rf"${{\eta}}$,{exp_case}",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/eta.pdf")
    plt.show()
    plt.close("all")
    
    H=1
    X0=X0[:-H]
    ReynoldsL_cut=ReynoldsL[:,:-H]
    Cepux_cut=Cepux[:,:-H]
    Cepux_fit_cut=Cepux_fit[:,:-H]
    Liux_cut=Liux[:,:-H]
    L_cut=L[:,:-H]
    sc.curves(5,ReynoldsL, "${{Re_\lambda}}$",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/ReynoldsL.pdf")
    plt.show()
    plt.close("all")
    sc.curves(5,L,r"${\lambda}$ ",X0,Y0)
    plt.savefig(f"{folder_fig}/{exp_case}/Lambda.pdf")
    plt.show()
    plt.close("all")
    sc.curves(5,np.nanstd(u,0), "urms",X0,Y0)
    plt.show()
    plt.close("all")
    
    
    sc.curves2((ReynoldsL),(Liux/L),rf"Liux/${{\lambda}}$  depending on ${{Re_\lambda}}$,{exp_case}")
    plt.savefig(f"{folder_fig}/{exp_case}/Liux_dvd_lambda_depending_ReynoldsL.pdf")
    plt.show()
    plt.close("all")
    sc.curves2((ReynoldsL),(Cepux_fit),rf"$C_{{\varepsilon}}$  depending on ${{Re_\lambda}}$ ,{exp_case}")
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_depending_on_ReynoldsL.pdf")
    plt.show()
    plt.close("all")
    sc.curves2((1/ReynoldsL),(Cepux_fit),rf"$C_{{\varepsilon}}$  depending on the inverse of ${{Re_\lambda}}$,{exp_case}")
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_depending_on_1_ReynoldsL.pdf")
    plt.show()
    plt.close("all")
    
    x=np.linspace(int(np.nanmin(ReynoldsL_cut[mid])),int(np.nanmax(ReynoldsL_cut[mid]))+1,int(np.nanmax(ReynoldsL_cut[mid]))+1-int(np.nanmin(ReynoldsL_cut[mid]))+1)
    sc.curves2(x,sc.moyenne(Liux_cut/L_cut,ReynoldsL_cut,Y0),rf"Liux/${{\lambda}}$  depending on ${{Re_\lambda}}$,{exp_case}")
    plt.savefig(f"{folder_fig}/{exp_case}/Liux_dvd_lambda_depending_ReynoldsL_mean.pdf")
    plt.show()
    plt.close("all")
    sc.curves2(x,sc.moyenne(Cepux_fit_cut,ReynoldsL_cut,Y0),rf"$C_{{\varepsilon}}$  depending on ${{Re_\lambda}}$,{exp_case}")
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_depending_on_ReynoldsL_mean.pdf")
    plt.show()
    plt.close("all")
    sc.curves2(1/x,sc.moyenne(Cepux_fit_cut,ReynoldsL_cut,Y0),rf"$C_{{\varepsilon}}$  depending on the inverse of ${{Re_\lambda}}$,{exp_case}")
    plt.savefig(f"{folder_fig}/{exp_case}/Cepux_depending_on_1_ReynoldsL_mean.pdf")
    plt.show()         
    plt.close("all")
