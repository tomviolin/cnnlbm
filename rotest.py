#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import  matplotlib.pyplot as plt
import numpy as np
import lbm

F=None
ux=None
uy=None
rho = None



def plotTaylorGreene(Nx,Ny):
    c = np.pi/Nx
    IX,IY = np.meshgrid(np.arange(Nx)*c,np.arange(Ny)*c)

    U = np.sin(IX)*np.cos(IY)
    V = -np.cos(IX)*np.sin(IY)
    plt.imshow(U*U+V*V,cmap='hot',alpha=0.5, extent=[0,np.pi,0,np.pi*Ny/Nx])
    plt.quiver(IX+0.25/Nx/c,IY+0.25/Nx/c,U,V)
    plt.show()
    F = lbm.calc_feq(1.0, U, V,lbm.idxs, lbm.cxs, lbm.cys, lbm.weights, lbm.NL)
    rho, ux, uy = lbm.calc_macro(F, lbm.cxs, lbm.cys)

    while True:
        F = lbm.calc_feq(1.0, ux, uy,lbm.idxs, lbm.cxs, lbm.cys, lbm.weights, lbm.NL)
        rho, ux, uy = lbm.calc_macro(F, lbm.cxs, lbm.cys)

        plt.imshow(ux*ux+uy*uy,cmap='hot',alpha=0.5, extent=[0,np.pi,0,np.pi*Ny/Nx])

        for i in range(lbm.NL):
            plt.quiver(IX+0.25/Nx/c,IY+0.25/Nx/c,F[:,:,i]*lbm.cxs[i]*0.01,F[:,:,i]*lbm.cys[i]*0.00)

        #plt.quiver(IX+0.25/Nx/c,IY+0.25/Nx/c,ux,uy)
        plt.show()

if __name__ == '__main__':
    plotTaylorGreene(lbm.Nx,lbm.Ny)
    
