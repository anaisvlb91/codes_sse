# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:03:07 2025

@author: alrouaud
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

sf_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\structure_fct_ux_r05_rotated.npz",allow_pickle=True)["sfux"]
sf_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\structure_fct_ux_r05_rotated.npz",allow_pickle=True)["sfux"]
sf_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\structure_fct_ux_r05_rotated.npz",allow_pickle=True)["sfux"]
sf_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\structure_fct_ux_r05_rotated.npz",allow_pickle=True)["sfux"]

eta_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\Kolmogorov_scales.npz",allow_pickle=True)["eta"]
eta_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\Kolmogorov_scales.npz",allow_pickle=True)["eta"]
eta_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\Kolmogorov_scales.npz",allow_pickle=True)["eta"]
eta_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\Kolmogorov_scales.npz",allow_pickle=True)["eta"]

u_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\fluctuating_velocities.npz",allow_pickle=True)["u"]
u_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\fluctuating_velocities.npz",allow_pickle=True)["u"]
u_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\fluctuating_velocities.npz",allow_pickle=True)["u"]
u_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\fluctuating_velocities.npz",allow_pickle=True)["u"]

U_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\rotated_cleaned_data.npz",allow_pickle=True)["U"]
U_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\rotated_cleaned_data.npz",allow_pickle=True)["U"]
U_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\rotated_cleaned_data.npz",allow_pickle=True)["U"]
U_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\rotated_cleaned_data.npz",allow_pickle=True)["U"]

x_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["x_mm"]
x_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["x_mm"]
x_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["x_mm"]
x_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["x_mm"]

y_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["y_mm"]
y_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["y_mm"]
y_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["y_mm"]
y_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\frame_cropped_rotated.npz",allow_pickle=True)["y_mm"]

grid=["STATIQUE","ACTIVE","STATIQUE","ACTIVE"]
speed=["0.25","0.25","0.38","0.38"]
color=["blue","darkorange","green","red"]

D=12
S=[sf_ss_23,sf_sz_23,sf_ss_38,sf_sz_38]
u=[u_ss_23,u_sz_23,u_ss_38,u_sz_38]
u_rms=[np.nanstd(u_ss_23,axis=0),np.nanstd(u_sz_23,axis=0),np.nanstd(u_ss_38,axis=0),np.nanstd(u_sz_38,axis=0)]
eta=[eta_ss_23,eta_sz_23,eta_ss_38,eta_sz_38]
FRAME=[(x_ss_23,y_ss_23),(x_sz_23,y_sz_23),(x_ss_38,y_ss_38),(x_sz_38,y_sz_38)]*D
name=["sf_ss_23","sf_sz_23","sf_ss_38","sf_sz_38"]
Uvelocity=[np.mean(U_ss_23,axis=0),np.nanmean(U_sz_23,axis=0),np.nanmean(U_ss_38,axis=0),np.nanmean(U_sz_38,axis=0)]

## EPSILON ux
def inertial_regression(r,sf,a,b,C):
    xMin = 2
    xMax = len(sf)-8
    sf=sf[:,a,b]
    err=-1
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
            if max[i, j]<2:
                max[i,j]=np.nan
    xMin=int(np.nanmean(min))
    xMax=int(np.nanmean(max))+1
    return xMin-2,xMax


def regression(r,sf,a,b,M,C):
    xMin=M[0]
    xMax=M[1]
    sf=sf[:,a,b]
    slope = np.nanmean(sf[xMin:xMax] / (r[xMin:xMax]**(2/3)))
    epsilon = (slope / C)**(3/2)
    return epsilon, xMin,xMax


COLOUR=["deepskyblue","sienna","lime","hotpink"]
vitesse=[0.25,0.25,0.38,0.38]
for i in range (len(S)):
        dx=FRAME[i][0][1]-FRAME[i][0][0]
        r=[]
        for j in range (len(S[i])):
            r=r+[dx*(j+1)]
        r = np.array(r)
        mid=int(len(FRAME[i][1])/2)
        p,q=(mid,50)
        sfuxcenter=S[i]
        f=0
        for k in sfuxcenter:
            for a in range (len(Uvelocity[i])):
                for b in range (len(Uvelocity[i][0])):
                    if Uvelocity[i][a,b]>vitesse[i]:
                        k[a,b]=np.nan
        M=inertial(sfuxcenter/u_rms[i],r/eta[i][p,q],FRAME[i][0],FRAME[i][1])
        print("M=",M)
        plt.loglog(r/eta[i][p,q],S[i][:,p,q]/u_rms[i][p,q],label=f"cas {speed[i]} m/s, grille {grid[i]}",color=color[i],linewidth=2)
        e=regression(r/eta[i][p,q],S[i]/u_rms[i],p,q,M,2)
        plt.loglog(r[e[1]:e[2]]/eta[i][p,q],2*(((e[0]*r[e[1]:e[2]])/eta[i][p,q])**(2/3)),color=COLOUR[i])
        print("M=",e[1],e[2])
plt.xlabel(r'$r/\eta$')
plt.ylabel(r"$S^2/u'$")
plt.legend(loc="lower right", fontsize=8)

plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def ensemble(ax, image, X, Y, norm, cmap, label=''):
    cmap.set_bad(color='white') 
    im = ax.pcolormesh(X, Y, image, cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/D$')
    ax.set_ylabel(r'$y/D$')
    ax.set_title(label)
    ax.set_aspect('equal', adjustable='box')
    return im

# Échelle couleur commune
all_data = [u_rms[i][:-2, :] for i in range(len(u_rms))]
vmin = np.nanmin([np.nanmin(arr) for arr in all_data])
vmax = np.nanmax([np.nanmax(arr) for arr in all_data])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.jet

# Création figure + axes
fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

# Position des sous-figures
P = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Tracer les 4 images
for i in range(4):
    a, b = P[i]
    im = ensemble(axs[a, b], u_rms[i][:-2, :],FRAME[i][0],FRAME[i][1],norm, cmap, label=f"{speed[i]} m/s, grille {grid[i]}")

# Ajouter colorbar à droite
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), location='right', pad=0.06)
cbar.set_label("u' en [m/s]")

# Sauvegarder
plt.savefig("C:/Users/alrouaud/OneDrive - NTNU/Documents/NTNU experiment/figures rapports de stage/u_rms_cases.pdf")
plt.show()


    


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import lvpyio as lv 

import def_velocities2 as velo
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
CALIB=lv.read_set(r"F:/CYLINDER/ImageCorrection")
pathsPIV = [p for p in pathPIV.glob(r'S*Cylinder*') if p.is_dir()]
case_names = [p.name for p in pathsPIV][1:]

##CALIB
frame_CALIB=CALIB[0][0]
x_CALIB,y_CALIB=get_CALIB_coordinates(frame_CALIB)
calibration=CALIB[0].as_masked_array().astype(np.float32)

D=12 #diameter of the cylinder in mm

x0_m =  -29.8850828566568 
y0_m = 14.35493338571112 
print("point of calibration in mm : (",x0_m,",",y0_m,")")

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

Tot=velo.BIG(s)
rawU=Tot[:,0,:,:].astype(np.float32)
rawU=np.where(rawU==0,np.nan,rawU)
lim=np.nanmin(np.nanmean(rawU,axis=0)),np.nanmax(np.nanmean(rawU,axis=0))
show_frame(np.nanmean(rawU,axis=0),X0/D,Y0/D,lim,r"$\langle{U}\rangle$ in $[m/s]$ ")
plt.show()


R_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\autocorr_fct_completed_ux_r_rotated.npz",allow_pickle=True)["Rux"]
R_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\autocorr_fct_completed_ux_r_rotated.npz",allow_pickle=True)["Rux"]
R_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\autocorr_fct_completed_ux_r_rotated.npz",allow_pickle=True)["Rux"]
R_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\autocorr_fct_completed_ux_r_rotated.npz",allow_pickle=True)["Rux"]
R=[R_ss_23,R_sz_23,R_ss_38,R_sz_38]
for i in range (len(R)):
        dx=FRAME[i][0][1]-FRAME[i][0][0]
        r=[]
        for j in range (len(R[i])):
            r=r+[dx*(j+1)]
        r = np.array(r)
        mid=int(len(FRAME[i][1])/2)
        p,q=(mid,50)
        plt.plot(r/eta[i][p,q],R[i][:,p,q],label=f"cas {speed[i]} m/s, grille {grid[i]}",color=color[i],linewidth=2)
        
plt.xlabel(r'$r/\eta$')
plt.ylabel(r"$R/u'^2$")
plt.legend(loc="upper right", fontsize=8)

plt.show()


Cep_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\Cepux_proportionality_scale_coeff.npz",allow_pickle=True)["Cepux"]
Cep_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\Cepux_proportionality_scale_coeff.npz",allow_pickle=True)["Cepux"]
Cep_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\Cepux_proportionality_scale_coeff.npz",allow_pickle=True)["Cepux"]
Cep_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\Cepux_proportionality_scale_coeff.npz",allow_pickle=True)["Cepux"]

ReynoldsL_ss_23=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD10p6Hz_Cylinder_FS1Hz_PIV\Reynolds_number_lambda.npz",allow_pickle=True)["ReynoldsL"]
ReynoldsL_sz_23=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD10p5Hz_Cylinder_FS1Hz_PIV\Reynolds_number_lambda.npz",allow_pickle=True)["ReynoldsL"]
ReynoldsL_ss_38=np.load(r"F:\CYLINDER\data_comparaison\SS_VFD15p8Hz_Cylinder_FS1Hz_PIV\Reynolds_number_lambda.npz",allow_pickle=True)["ReynoldsL"]
ReynoldsL_sz_38=np.load(r"F:\CYLINDER\data_comparaison\SZ_VFD15p7Hz_Cylinder_FS1Hz_PIV\Reynolds_number_lambda.npz",allow_pickle=True)["ReynoldsL"]

Cep=[Cep_ss_23,Cep_sz_23,Cep_ss_38,Cep_sz_38]
ReynoldsL=[ReynoldsL_ss_23,ReynoldsL_sz_23,ReynoldsL_ss_38,ReynoldsL_sz_38]

for i in range(len(Cep)):
    plt.plot(ReynoldsL[i],Cep[i],"x",color=color[i],label="cas {speed[i]}m/s et grille {grid[i]}")
plt.xlabel(r'${{Re_\lambda}}$')
plt.ylabel(r"$C_{{\varepsilon}}$")
plt.title(r"$C_{{\varepsilon}}$  depending on ${{Re_\lambda}}$ ")
plt.legend(loc="upper right", fontsize=8)
plt.show()



for i in range(len(Cep)):
    # Supposons que Cep[i] et ReynoldsL[i] sont de forme (n_lignes, n_colonnes)
    for j in range(Cep[i].shape[0]):
        if j == 0:
            plt.plot(ReynoldsL[i][j], Cep[i][j], "x", color=color[i],
                     label=f"cas {speed[i]} m/s, grille {grid[i]}")
        else:
            plt.plot(ReynoldsL[i][j], Cep[i][j], "x", color=color[i])  # sans label

plt.xlabel(r"${Re_\lambda}$")
plt.ylabel(r"$C_{\varepsilon}$")
plt.title(r"$C_{\varepsilon}$ en fonction de $Re_\lambda$")
plt.legend(loc="upper right", fontsize=8)
plt.show()

