# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:18:12 2025

@author: alrouaud
"""

from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from matplotlib.patches import Circle
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, datasets, ndimage
from tqdm import tqdm

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


def blurred(v_clean,sigma,radius):
    v_blurred=gaussian_filter(v_clean, sigma,radius=radius)
    return v_blurred

def grad5(u,dx,axis):
    du=np.zeros(u.shape)
    if axis==1:
        N=u.shape[1]
        du[:,0,:] = (-25 * u[:,0,:] + 48 * u[:,1,:] - 36 * u[:,2,:] + 16 * u[:,3,:] - 3 * u[:,4,:]) / (12 * dx)
        du[:,1,:] = (-3 * u[:,0,:] - 10 * u[:,1] + 18 * u[:,2,:] - 6 * u[:,3,:] + u[:,4,:]) / (12 * dx)
        du[:,2:N - 2,:] = (-u[:,4:N,:] + 8 * u[:,3:N - 1,:] - 8 * u[:,1:N - 3,:] + u[:,0:N - 4,:]) / (12 * dx)
        du[:,N - 2,:] = (-u[:,N - 5,:] + 6 * u[:,N - 4,:] - 18 * u[:,N - 3,:] + 10 * u[:,N - 2,:] + 3 * u[:,N - 1,:]) / (12 * dx)
        du[:,N - 1,:] = (3 * u[:,N - 5,:] - 16 * u[:,N - 4,:] + 36 * u[:,N - 3,:] - 48 * u[:,N - 2,:] + 25 * u[:,N - 1,:]) / (12 * dx)
    elif axis==2:
        N=u.shape[2]
        du[:,:,0] = (-25 * u[:,:,0] + 48 * u[:,:,1] - 36 * u[:,:,2] + 16 * u[:,:,3] - 3 * u[:,:,4]) / (12 * dx)
        du[:,:,1] = (-3 * u[:,:,0] - 10 * u[:,:,1] + 18 * u[:,:,2] - 6 * u[:,:,3] + u[:,:,4]) / (12 * dx)
        du[:,:,2:N - 2] = (-u[:,:,4:N] + 8 * u[:,:,3:N - 1] - 8 * u[:,:,1:N - 3] + u[:,:,0:N - 4]) / (12 * dx)
        du[:,:,N - 2] = (-u[:,:,N - 5] + 6 * u[:,:,N - 4] - 18 * u[:,:,N - 3] + 10 * u[:,:,N - 2] + 3 * u[:,:,N - 1]) / (12 * dx)
        du[:,:,N - 1] = (3 * u[:,:,N - 5] - 16 * u[:,:,N - 4] + 36 * u[:,:,N - 3] - 48 * u[:,:,N - 2] + 25 * u[:,:,N - 1]) / (12 * dx)
    return du

#methode approximative
def lambda2(u,v,dx,dy):
    L=(grad5(u,dx,axis=2)**2)+(grad5(v,dx,axis=2)*grad5(u,dy,axis=1))
    return L

def area_pourcentage2(L,m,p,label):
    thr=np.linspace(-0.4,0,m)
    area=L[0].shape[0]*L[0].shape[1]
    y=[]
    for j in thr:
        l=[]
        for k in L:
            l=l+[(np.sum(np.where(k <j, 1,0))/area)*100]
        y=y+[np.mean(l)]
    y=np.array(y)

    plt.plot(thr,y,label=label)
    T=thr[np.nanargmax(y>p)]
    T=round(T,3)
    print('thr= ', T)
    plt.scatter(np.array([T]*10),np.linspace(0,30, 10), color='blue', marker='o', s=10, label=f"{p} %")
    plt.xlabel("threshold")
    plt.ylabel("pourcentage of visible area %")
    plt.legend(loc="upper left", fontsize=8)
    plt.show()
    return T,y


def vortex_center(mini,x_mm,y_mm):
    vortex_center=[]
    for y in range (1,len(mini)-2):
        for x in range (1,len(mini[0])-2):
            if mini[y,x]==mini[y+1,x] and mini[y,x]==mini[y-1,x] and mini[y,x]==mini[y,x+1] and mini[y,x]==mini[y,x-1] and mini[y,x]!=0 :
                vortex_center=vortex_center + [(y_mm[y],x_mm[x])]
    return vortex_center

def detection(L,T,R,du,dv,x_mm,y_mm):
    vortex={}
    index_vortex = defaultdict(list)
    mini0=ndi.minimum_filter(np.where(L[0]<T,L[0],0),size=3)
    VC=vortex_center(mini0,x_mm,y_mm)
    n=0
    t=0
    for c0 in VC:
        n=n+1
        vortex[f"{n}"]=[[t,c0]]
        index_vortex[f"{0}"].append(f"{n}")
    VCt1=VC
    for t in range (1,len(L)-1):
        print(t)
        VCt0=VCt1
        VCt_prime=[]
        for ct0 in VCt0:
            y,x=ct0
            x_idx = np.argmin(np.abs(x_mm - x))
            y_idx = np.argmin(np.abs(y_mm - y))
            x_prime = x + du[y_idx, x_idx]
            y_prime = y + dv[y_idx, x_idx]
            ct=(y_prime,x_prime)
            VCt_prime.append(ct)
            
        VCt_prime_prime=[]
        for ct0 in VCt0:
            y,x=ct0
            x_idx = np.argmin(np.abs(x_mm - x))
            y_idx = np.argmin(np.abs(y_mm - y))
            x_prime = x + du[y_idx, x_idx]*2
            y_prime = y + dv[y_idx, x_idx]*2
            ct=(y_prime,x_prime)
            VCt_prime_prime.append(ct)
            
        minit1=ndi.minimum_filter(np.where(L[t]<T,L[t],0),size=3)
        VCt1=vortex_center(minit1,x_mm,y_mm)
        
        minit2=ndi.minimum_filter(np.where(L[t+1]<T,L[t+1],0),size=3)  # attention dernier t qui depasse
        VCt2=vortex_center(minit2,x_mm,y_mm)
        
        condition=False
        for ct1 in VCt1:
            distance=[]
            if ct1[1]<2*du[int(len(y_mm)/2),1]+x_mm[0] :
                n=n+1
                vortex[f"{n}"]=[[t,ct1]]
                index_vortex[f"{t}"].append(f"{n}")
            y1,x1=ct1
            for y2,x2 in VCt_prime:
                distance = distance+[(x2 - x1)**2 + (y2 - y1)**2]
            dist=np.argmin(distance)
            if distance[dist]<=(2*R)**2 :
                    y0,x0 = VCt0[dist]
                    t0=t
                    exist=False
                    while t0>=0 and t0>t-20 and exist==False:
                        t0=t0-1
                        for cle in index_vortex.get(f"{t0}", []):
                            liste = vortex[cle]
                            if  liste[-1] == [t-1,(y0, x0)]:
                                vortex[cle].append([t,ct1])
                                exist=True
                    condition=True
            if condition==False:
                    for ct2 in VCt2:
                        distance=[]
                        y1,x1=ct2
                        for y2,x2 in VCt_prime_prime:
                            distance = distance+[(x2 - x1)**2 + (y2 - y1)**2]
                        dist=np.argmin(distance)
                        if distance[dist]<=(2*R)**2:
                                y0,x0 = VCt0[dist]
                                t0=t
                                exist=False
                                while t0>=0 and t0>t-20 and exist==False:
                                    t0=t0-1
                                    for cle in index_vortex.get(f"{t0}", []):
                                        liste = vortex[cle]
                                        if  liste[-1] == [t-1,(y0, x0)]:
                                            vortex[cle].append([t+1,ct2])
                                            exist=True
                                condition=True
                        if condition==False:
                             n=n+1
                             vortex[f"{n}"]=[[t,ct1]]
                             index_vortex[f"{t}"].append(f"{n}")
    return vortex


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

def detection2(L,T,R,du,dv,x_mm,y_mm):
    vortex={}
    index_vortex = defaultdict(list)
    mini0=ndi.minimum_filter(np.where(L[0]<T,L[0],0),size=3)
    VC=vortex_center(mini0,x_mm,y_mm)
    n=0
    t=0
    for c0 in VC:
        n=n+1
        vortex[f"{n}"]=[[t,c0]]
        index_vortex[f"{0}"].append(f"{n}")
    VCt1=VC
    for t in range (1,len(L)):
        print(t)
        VCt0=VCt1
        i=0
        VCt_prime=[]
        for ct0 in VCt0:
            i=i+1
            y,x=ct0
            x_idx = np.argmin(np.abs(x_mm - x))
            y_idx = np.argmin(np.abs(y_mm - y))
            x_prime = x + du[y_idx, x_idx]
            y_prime = y + dv[y_idx, x_idx]
            ct=(y_prime,x_prime)
            VCt_prime.append(ct)
            
        minit1=ndi.minimum_filter(np.where(L[t]<T,L[t],0),size=3)
        VCt1=vortex_center(minit1,x_mm,y_mm)
        
        for ct1 in VCt1:
            distance=[]
            if ct1[1]<2*du[int(len(y_mm)/2),1] :
                n=n+1
                vortex[f"{n}"]=["new",[t,ct1]]
                index_vortex[f"{t}"].append(f"{n}")
            y1,x1=ct1
            if len(VCt_prime)>=1:
                for y2,x2 in VCt_prime:
                    distance = distance+[(x2 - x1)**2 + (y2 - y1)**2]
                dist=np.argmin(distance)
                if distance[dist]<=(2*R)**2 :
                        y0,x0 = VCt0[dist]
                        t0=t
                        exist=False
                        while t0>=0 and t0>t-55 and exist==False:
                            t0=t0-1
                            for cle in index_vortex.get(f"{t0}", []):
                                liste = vortex[cle]
                                if  liste[-1] == [t-1,(y0, x0)]:
                                    vortex[cle].append([t,ct1])
                                    exist=True
                        condition=True
                else:
                        condition=False
            else:
                    condition=False
            if condition==False:
                n=n+1
                vortex[f"{n}"]=[[t,ct1]]
                index_vortex[f"{t}"].append(f"{n}")
    return vortex


def PDF2(grandeurs, lim,freq, label):
    bins = np.linspace(1, lim, lim)
    frame_duration = freq  # durée d'une frame en secondes
    # Calcul de l'histogramme SANS densité
    count, bin_edges = np.histogram(grandeurs, bins=bins, density=False)
    # Convertir les comptes en durée (temps passé dans chaque bin)
    time_count = count * frame_duration
    plt.bar(bin_edges[:-1]*freq, time_count, width=(bin_edges[1]-bin_edges[0])/15)
    plt.title('PDF de la ' + label)
    plt.xlabel(label)
    plt.show()


from matplotlib.animation import FuncAnimation
from functools import partial
import scipy.ndimage as ndi
from matplotlib.patches import Circle

def update(snapshot,im,field,ax,R,T,dico_prof,dico_piv,x_mm,y_mm):
        frame = np.where(field[snapshot] < T, field[snapshot], 0)
        im.set_array(frame)

        # Supprimer les anciens cercles
        for patch in reversed(ax.patches):
            patch.remove()

        # Détecter et dessiner les centres de vortex de la profilometry
        VCprof=[t for k, v in dico_prof.items() for t in v if t[0]==snapshot*3]
        for V in VCprof:
            (y,x)=V[1]
            try:
                cercle = Circle((x, y), R, color='green', fill=False,lw=4)
                ax.add_patch(cercle)
            except IndexError:
                continue
        
        # Détecter et dessiner les centres de vortex de la profilometry
        VCpiv=[t for k, v in dico_piv.items() for t in v if t[0]==snapshot]
        for V in VCpiv:
            (y,x)=V[1]
            try:
                cercle = Circle((x, y), R, color='white', fill=False,lw=2)
                ax.add_patch(cercle)
            except IndexError:
                continue
        return [im] + ax.patches


def gif_vortex(W,T,R,dico_prof,dico_piv,x_mm,y_mm,folder_fig,label):
        ## MAKE A GIF OF THE PIV FIELD
        ## ALONG VELOCITY
        # Set up the intial figure and axis
        fig, ax, im = show_frame(np.where(W[0] < T, W[0], 0), x_mm, y_mm, r"vortex evolution")
            
        # Create the animation
        ani = FuncAnimation(fig, partial(update,im=im,field=W,ax=ax,R=R,T=T,dico_prof=dico_prof,dico_piv=dico_piv,x_mm=x_mm,y_mm=y_mm,), frames=len(W), blit=False)
        # Save the animation as a GIF
        ani.save(f"{folder_fig}/{label}.gif", writer='pillow',fps=15)
        return("done")