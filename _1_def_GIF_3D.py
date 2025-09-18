# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:06:22 2025

@author: alrouaud
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import math
from scipy import signal, datasets, ndimage
from mpl_toolkits.mplot3d import Axes3D  # active les outils 3D
from matplotlib.animation import FuncAnimation
from functools import partial

def picture_3D(X,Y,Z,label,zmin, zmax):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.plot_surface(X, Y, Z, cmap='jet') 
    fig.colorbar(im)  # légende des couleurs
    ax.set_xlabel(r'$x ~ [mm]$')  # set label x axis
    ax.set_ylabel(r'$y ~ [mm]$')
    ax.set_zlabel(r'$z ~ [cm]$')
    ax.set_zlim(zmin, zmax)
    ax.set_title(label)
    return fig, ax, im

def update_surface(snapshot, ax, field, X, Y,zmin,zmax):
    # Remove old surface by iterating over each item in the collections
    for coll in ax.collections:
        coll.remove()
    
    # Now, plot the new surface
    im = ax.plot_surface(X, Y, field[snapshot], cmap='jet', vmin=zmin, vmax=zmax)
    return im,

def gif(L,x,y,label,zmin, zmax,exp_case,folder):
    zmin = np.nanmin(L)
    zmax = np.nanmax(L)
    fig, ax, im = picture_3D(x,y,L[0],label,zmin, zmax)
    # Create the animation
    ani = FuncAnimation(fig, partial(update_surface,ax=ax,field=L,X=x,Y=y,zmin=zmin, zmax=zmax), frames=len(L), blit=False)
    # Save the animation as a GIF
    ani.save(f"{folder}/{exp_case}/Gif_{exp_case}.gif", writer='pillow')
    return("done")

