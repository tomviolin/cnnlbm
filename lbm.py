#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import time

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past cylinder
for an isothermal fluid
"""

""" Lattice Boltzmann Simulation """
visc = 0.025
# Simulation parameters
Nx                     = 128    # resolution x-dir
Ny                     = 64     # resolution y-dir
rho0                   = 100    # average density
tau                    = visc + 0.5    # collision timescale
Nt                     = 4000   # number of timesteps
plotRealTime = True # switch on for plotting as the simulation goes along
FLOW_VELOCITY = 0.0662
STRESS_LIMIT = 0.4
stillrun = True
ITER_PER_FRAME = 20
MOVIE = False
CV_PREVIEW = True
if CV_PREVIEW:
    mpl.use('Agg')
else:
    mpl.use('Qt5Agg')

def on_press(event):
    global stillrun
    print (f"KEY PRESSED: '{event.key}'")
    if event.key in ['q','escape','ctrl+c']:
        stillrun = False
        plt.close('all')
        plt.pause(0.1)

def noneq_stress(feq, f):
    return np.sum((feq - f)**2 / feq, axis=2)

def main():
    global stillrun, ITER_PER_FRAME
    
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
    
    """
    # Initial Conditions
    F = np.ones((Ny,Nx,NL)) #* rho0 / NL
    np.random.seed(42)
    F += 0.01*np.random.randn(Ny,Nx,NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    rho = np.sum(F,2)
    for i in idxs:
        F[:,:,i] *= rho0 / rho
    """
    F = np.zeros((Ny,Nx,NL))
    rho = np.ones((Ny,Nx)) * rho0
    ux = np.ones((Ny,Nx)) * FLOW_VELOCITY
    uy = np.zeros((Ny,Nx))

    # Cylinder boundary
    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = (X - Nx/4)**2 + (Y - Ny/2)**2 < (Ny/8)**2

    ux[cylinder] = 0
    uy[cylinder] = 0

    # Prep figure
    fig = plt.figure(figsize=(4,2), dpi=80)
    if ~CV_PREVIEW:
        #fig.canvas.manager.window.move(100,100)
        fig.canvas.mpl_connect('key_press_event', on_press)
    
    # Simulation Main Loop
    it = -1
    while stillrun:
        if not stillrun:
            break
        it += 1
        print(f"it: {it} stillrun: {stillrun} Re: {FLOW_VELOCITY * Ny/4 / visc}")
        
        if it > 0:
            # don't stream on the first iteration
            # streaming step:
            # actually advect the particles
            for i, cx, cy in zip(idxs, cxs, cys):
                F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
                F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
            # then perform bounceback on the obstacles
            # Set reflective boundaries
            bndryF = F[cylinder,:]
            bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

            # then finally set the fluid to be the post-streaming, pre-collision values
            rho = np.sum(F,2)
            ux  = np.sum(F*cxs,2) / rho
            uy  = np.sum(F*cys,2) / rho
        
        if it == 0:
            # on the first iteration, set the initial conditions
            ux = np.ones((Ny,Nx)) * FLOW_VELOCITY
            uy = np.zeros((Ny,Nx))
            rho = np.ones((Ny,Nx)) * rho0

        # the rest of the iteration is performed for all iterations
        # Apply boundary conditions
        ux[:,0] = FLOW_VELOCITY
        uy[:,0] = 0
        ux[:,-1] = FLOW_VELOCITY
        uy[:,-1] = 0
        ux[0,:] = FLOW_VELOCITY
        uy[0,:] = 0
        ux[-1,:] = FLOW_VELOCITY
        uy[-1,:] = 0
        if it==0:
            thistau = 1.0
        else:
            thistau = tau
        # Apply forcing
        ux[cylinder] = 0
        uy[cylinder] = 0

        # Calculate Feq
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )

        # check for stability
        #if np.any(Feq < 0):
        #    print("WARNING: stability condition violated")

        stress = noneq_stress(Feq, F)
        stressed = np.logical_or(
            stress > STRESS_LIMIT,
            cylinder)
        for i in range(NL):
            F[:,:,i][stressed] = Feq[:,:,i][stressed]

        F += -(1.0/thistau) * (F - Feq)

        #if it > 0: 
            # Apply boundary 
            #F[cylinder,:] = bndryF

        
        yield np.stack([ux,uy,rho,cylinder],axis=2)
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % ITER_PER_FRAME) == 0) or (it == Nt-1):
            plt.cla()
            #ux[cylinder] = 0
            #uy[cylinder] = 0
            vel = np.sqrt(ux**2+uy**2)
            
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[cylinder] = np.nan
            vorticity = np.ma.array(vorticity, mask=cylinder)
            ncyl = np.ma.array(~cylinder, mask=~cylinder)
            plt.imshow(vorticity, cmap='RdBu', 
                interpolation='bilinear',
                alpha=1.0)
            plt.clim(-.1, .1)
            
            plt.imshow(vel, cmap='jet', 
                interpolation='bilinear',
                alpha=.1)
            plt.clim(0,0.15)
            #plt.imshow(ncyl, cmap='gray', alpha=1.0)
            #streamplot = plt.streamplot(X, Y, ux, uy, color=vel, cmap='viridis', density=1.5, linewidth=1, arrowsize=1, arrowstyle='->')
            #stressedge = np.ma.array(stressed, mask=~stressed)
            #plt.imshow(stressedge, cmap='gray', alpha=1.0)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)   
            ax.set_aspect('equal')  
            figfile = f'rlb{it:06d}.png'
            if ~MOVIE:
                figfile = f'/dev/shm/rlb.png'
            plt.savefig(figfile,dpi=240)
            if CV_PREVIEW:
                cv2.imshow('frame', cv2.imread(figfile))
                k = cv2.waitKey(1)
                if k > 0:
                    print(f"KEY PRESSED: {k}")
                if k == 81: # left
                    ITER_PER_FRAME -= 1
                elif k == 83: # right
                    ITER_PER_FRAME += 1
                elif k == 82: # up
                    ITER_PER_FRAME += 10
                elif k == 84: # down
                    ITER_PER_FRAME -= 10
                elif k == 27 or k == ord('q'): # escape
                    stillrun = False
                    plt.close('all')
                    plt.pause(0.1)
            else:
                plt.pause(0.01)
                if not stillrun:
                    break
    
    # Save figure
    plt.savefig('rlbfinal.png',dpi=240)
    plt.show()
        
    return 0



if __name__== "__main__":
    gen = main()
    while stillrun:
        next(gen)

