#!/usr/bin/env python3
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
import numba as nb
import cv2
cv2.ocl.setUseOpenCL(True)
import time
import os
"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past cylinder
for an isothermal fluid
"""
NP_FLOAT_TYPE = np.float64
NB_FLOAT_TYPE = nb.float64

BOLD='\033[33;1m'
END='\033[0m'
""" Lattice Boltzmann Simulation """
visc = 0.0025
# Simulation parameters
Nx                     = 130   # resolution x-dir
Ny                     = 50   # resolution y-dir
rho0                   = 1.0  # average density
tau                    = visc + 0.5    # collision timescale
Nt                     = 4000   # number of timesteps
plotRealTime = True # switch on for plotting as the simulation goes along
PLOT_SAVE_PLOTS = False
PLOT_DISPLAY_PLOTS = True
FLOW_VELOCITY = 0.0662
NUM_CYLINDERS = 8
STRESS_LIMIT = 0.59
stillrun = True
ITER_PER_FRAME = 7
#if plotRealTime:
#    mpl.use('Agg')
print(f"{BOLD}loading lbm.py{END}")
X, Y = np.meshgrid(range(Nx), range(Ny))
def on_press(event):
    global stillrun
    print (f"{BOLD}KEY PRESSED: '{event.key}'{END}")
    if event.key in ['q','escape','ctrl+c']:
        stillrun = False
        #plt.close('all')
        #plt.pause(0.1)

@nb.njit
def noneq_stress(feq, f):
    return np.sum((feq - f)**2 / feq, axis=2)

# Calculate Feq
@nb.njit
def calc_feq(rho, ux, uy,idxs, cxs, cys, weights, NL):
    Feq = np.zeros((Ny,Nx,NL), dtype=NP_FLOAT_TYPE)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        dotProd = (cx*ux+cy*uy)
        dotuu = (ux**2+uy**2)
        #Feq[:,:,i] = rho * w * ( 1 + 3*dotProd  + 9*dotProd*dotProd/2 - 3*(dotuu)/2 )
        Feq[:,:,i] = rho * w * (np.exp(3/2*(2*dotProd - dotuu)) )
                #scD * rho*(exp((3. / 2.)*(2 * dotProd - dot(u, u))))
    return Feq

@nb.njit
def calc_cylinder(X,Y,x,y,r):
    cylinder = (X - x)**2 + (Y - y)**2 < (r)**2
    return cylinder

@nb.njit
def stream(F, cx, cy,idxs,cxs,cys):
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:,:,i] = np.roll(F[:,:,i], cx)
        F[:,:,i] = np.roll(F[:,:,i].T, cy).T
    return F
# calc macroscopic variables
@nb.njit(nb.numba.types.Tuple((NB_FLOAT_TYPE[:,:],NB_FLOAT_TYPE[:,:],NB_FLOAT_TYPE[:,:]))(NB_FLOAT_TYPE[:,:,:],nb.int64[:],nb.int64[:]))
def calc_macro(F, cxs, cys):
    rho = np.sum(F,2)
    ux  = np.sum(F*cxs,2) / rho
    uy  = np.sum(F*cys,2) / rho
    return rho, ux, uy

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL, dtype=np.int64)
cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1], dtype=np.int64)
cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1], dtype=np.int64)
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
#                     0  1  2  3  4  5  6  7  8
rot90idxs = np.array([0, 3, 4, 5, 6, 7, 8, 1, 2], dtype=np.int64)
rot90from = np.array([0, 7, 8, 1, 2, 3, 4, 5, 6], dtype=np.int64)

'''
# rot90idxs
6   5   4       8   7   6
7   0   3   =>  1   0   5
8   1   2       2   3   4
'''

def main():
    global stillrun, ITER_PER_FRAME
    
    F = np.zeros((Ny,Nx,NL), dtype=NP_FLOAT_TYPE)
    rho = np.ones((Ny,Nx), dtype=NP_FLOAT_TYPE) * rho0
    ux = np.ones((Ny,Nx), dtype=NP_FLOAT_TYPE) * FLOW_VELOCITY
    uy = np.zeros((Ny,Nx), dtype=NP_FLOAT_TYPE)

    # Cylinder boundary
    cylinder = np.zeros((Ny,Nx), dtype=bool)

    cyls = np.random.random((NUM_CYLINDERS, 3)).astype(NP_FLOAT_TYPE)
    cyls[:,0] *= Nx/2  # x coordinate
    cyls[:,1] *= Ny/20# size
    cyls[:,2] *= np.pi*2 # phase

    ux[cylinder] = 0
    uy[cylinder] = 0

    F = calc_feq(rho, ux, uy,idxs, cxs, cys, weights, NL)

    # Simulation Main Loop
    it = -1
    lasttime=None
    while stillrun:
        if not stillrun:
            break
        it += 1
        thistime = time.time()
        if lasttime is None:
            lasttime = thistime
        else:
            # print(f"it: {it}, 1/dt: {1/(thistime-lasttime):.2f}")
            lasttime = thistime
        
        # update obstacles (cylinder)
        # do original cylinder
        cylinder = np.zeros((Ny,Nx), dtype=bool)
        cylinder = calc_cylinder(X,Y,Nx/5,Ny/2,Ny/12)

        #cylinder = np.logical_or(cylinder, calc_cylinder(X,Y,Nx/2,Ny/2,1))


        for i in range(NUM_CYLINDERS):
            xpos = Nx/3 + cyls[i,0] # x coordinate
            ypos = Ny/2 + Ny/4.5 * np.sin(it/250 + cyls[i,2],dtype=NP_FLOAT_TYPE) # y coordinate
            size = cyls[i,1] # size
            cylinder = np.logical_or(cylinder, (X - xpos)**2 + (Y - ypos)**2 < size**2)

        stream(F, cxs, cys,idxs,cxs,cys) # stream the particles
        # then perform bounceback on the obstacles
        # Set reflective boundaries
        bndryF = F[cylinder,:]
        bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

        # then finally set the fluid to be the post-streaming, pre-collision values
        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho
        
        # the rest of the iteration is performed for all iterations
        # Apply boundary conditions
        ux[:,:2] = FLOW_VELOCITY
        uy[:,:2] = 0
        ux[:,-1] = FLOW_VELOCITY
        uy[:,-1] = 0
        ux[0,:] = FLOW_VELOCITY
        uy[0,:] = 0
        ux[-1,:] = FLOW_VELOCITY
        uy[-1,:] = 0
        ux[:,-30:] = FLOW_VELOCITY*0.001 + ux[:,-30:]*0.999
        uy[:,-30:] = uy[:,-30:]*0.999
        thistau = tau
        # Apply forcing
        ux[cylinder] = ux[cylinder]*0.
        uy[cylinder] = uy[cylinder]*0.
        # Apply collision
        Feq = calc_feq(rho, ux, uy,idxs, cxs, cys, weights, NL)
        F += -(1.0/thistau) * (F - Feq)
        # Apply stress limit
        stress = noneq_stress(Feq, F)
        stressed = np.logical_or(
            stress > STRESS_LIMIT,
            cylinder)
        for i in range(NL):
            F[:,:,i][stressed] = Feq[:,:,i][stressed]*0.9 + F[:,:,i][stressed]*0.1

        rho, ux, uy = calc_macro(F, cxs, cys)

        ux[cylinder] = 0
        uy[cylinder] = 0

        if it > 10:
            yieldthing = np.array(np.stack([ux/0.1,uy/0.1,rho/rho0]+[F[:,:,k] for k in range(NL)]+[cylinder,],axis=2)).copy()
            print(f"{BOLD}yieldthing: {yieldthing.shape}{END}")
            yield yieldthing
            print(f"{BOLD}yielded: {it}{END}")
        else:
            continue
        
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % ITER_PER_FRAME) == 0) or (it == Nt-1):
            print(f"{BOLD}plotting: {it}{END}")
            uxp=cv2.GaussianBlur(cv2.UMat(ux), (5,5),2.7)
            uyp=cv2.GaussianBlur(cv2.UMat(uy), (5,5),2.7)

            print(f"{BOLD}blurred: {it}{END}")
            vel = cv2.sqrt(cv2.add(cv2.pow(uxp,2),cv2.pow(uyp,2)))

            print(f"{BOLD}vencalc'd: {it}{END}")
            
            print(f"{BOLD}makng kennels for vorticity/curl 1/2: {it}{END}")
            kernel1 = cv2.UMat(np.array([
                [ 0, 1, 0],
                [ 0, 0, 0],
                [ 0,-1, 0]],dtype=NP_FLOAT_TYPE))

            print(f"{BOLD}makng kennels for vorticity/curl 2/2: {it}{END}")
            kernel2 = cv2.UMat(np.array([
                [ 0, 0, 0],
                [-1, 0, 1],
                [ 0, 0, 0]],dtype=NP_FLOAT_TYPE))


            print(f"{BOLD}vorticity calc: kernels for vorticity/curl: {it}{END}")
            vorticity = cv2.add(cv2.filter2D(uxp,6,kernel1),cv2.filter2D(uyp,6,kernel2))
            print(f"{BOLD}vorticity calc done: {it}{END}")
            #print(f"vorticity: {vorticity.get().max()}")
            #rolluxp1 = np.roll(uxp, 1, axis=0)
            #rolluxm1 = np.roll(uxp, -1, axis=0)
            #rolluyp1 = np.roll(uyp, 1, axis=1)
            #rolluym1 = np.roll(uyp, -1, axis=1)

            #vorticity = (np.roll(uxp, -1, axis=0) - np.roll(uxp, 1, axis=0)) - (np.roll(uyp, -1, axis=1) - np.roll(uyp, 1, axis=1))
            vmaskpos = cv2.multiply(vorticity,5.0)
            vmaskneg = cv2.multiply(vorticity,-5.0)
            #vmaskpos = cv2.inRange(cv2.multiply(vorticity,255.0),0,255)
            #vmaskneg = cv2.inRange(cv2.multiply(vorticity,-255.0),0,255)
            cylnum = cylinder + 0.0
            cylmask = cv2.multiply(cv2.subtract(cv2.UMat(cylnum),1),-1)
            #r_chan = cv2.multiply(vmaskpos,cylmask) #cv2.inRange(cv2.multiply(vmaskpos,1),0,1)
            #g_chan = cv2.multiply(vmaskneg, cylmask)#cv2.inRange(cv2.multiply(vmaskneg,1),0,1)
            r_chan = cv2.multiply(vmaskpos, cylmask)
            g_chan = cv2.multiply(vmaskneg, cylmask)

            b_chan = cv2.multiply(g_chan,0)
            #print(f"r_chan: {r_chan.get().max()}")
            #print(f"g_chan: {g_chan.get().max()}")
            #print(f"b_chan: {b_chan.get().max()}")

            #vortrgb[:,:,3] = (~cylinder).astype(NP_FLOAT_TYPE)

            vortrgb = cv2.merge([r_chan,g_chan,r_chan])
            #cv2.imshow('frame', cv2.resize((ux**4+uy**4)*10, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            #cv2.imshow('frame', vortrgb);
            if __name__ == "__main__":
                print(f"{BOLD}showing: {it}{END}")
                cv2.imshow('frame', cv2.resize(vortrgb, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC))
                print(f"{BOLD}showed: {it}{END}")
                k=cv2.waitKey(100)
                print(f"{BOLD}showed: {it}-{k}{END}")
                #    cv2.UMat(vortrgb),(0,0), fx=1/2.5, fy=1/2.5, interpolation=cv2.INTER_AREA), (0,0),fx=10,fy=10,interpolation=cv2.INTER_CUBIC))
                if k != -1:
                    print(f"{BOLD}KEY PRESSED: {k}{END}")
                if k in [ord('q'), 27]:
                    stillrun = False
                elif k == 81: # left
                    ITER_PER_FRAME -= 1 ; print(f"{BOLD}ITER_PER_FRAME: {ITER_PER_FRAME}{END}")
                elif k == 83: # right
                    ITER_PER_FRAME += 1 ; print(f"{BOLD}ITER_PER_FRAME: {ITER_PER_FRAME}{END}")

            #plt.cla()
            #ux[cylinder] = 0
            #uy[cylinder] = 0
            #vel = np.sqrt(ux**2+uy**2)
            
            #vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            #vorticity[cylinder] = np.nan
            #vorticity = np.ma.array(vorticity, mask=cylinder)
            #ncyl = np.ma.array(~cylinder, mask=~cylinder)
            #plt.imshow(vorticity, cmap='RdBu', 
            #    interpolation='bilinear',
            #    alpha=1.0)
            #plt.clim(-.1, .1)
            
            #plt.imshow(vel, cmap='jet', 
            #    interpolation='bilinear',
            #    alpha=.1)
            #plt.clim(0,0.15)
            #plt.imshow(ncyl, cmap='gray', alpha=1.0)
            #streamplot = plt.streamplot(X, Y, ux, uy, color=vel, cmap='viridis', density=1.5, linewidth=1, arrowsize=1, arrowstyle='->')
            #stressedge = np.ma.array(stressed, mask=~stressed)
            #plt.imshow(stressedge, cmap='gray', alpha=1.0)
            #ax = plt.gca()
            #ax.invert_yaxis()
            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)   
            #ax.set_aspect('equal')  
            #figfile = f'/dev/shm/.rlb.png'
            #if PLOT_SAVE_PLOTS:
            #    figfile = f'images/rlb{it:06d}.png'
            #plt.savefig(figfile,dpi=240)
            #if not PLOT_SAVE_PLOTS:
            #    os.rename(figfile, f'/dev/shm/rlb.png')
            #    figfile = f'/dev/shm/rlb.png'
            #if PLOT_DISPLAY_PLOTS:
            #    #cv2.imshow('frame', cv2.imread(figfile))
            #    #k = cv2.waitKey(10)
            #    k=0
            #    if k > 0:
            #        print(f"KEY PRESSED: {k}")
            #    if k == 81: # left
            #        ITER_PER_FRAME -= 1
            #    elif k == 83: # right
            #        ITER_PER_FRAME += 1
            #    elif k == 82: # up
            #        ITER_PER_FRAME += 10
            #    elif k == 84: # down
            #        ITER_PER_FRAME -= 10
            #    elif k == 27 or k == ord('q'): # escape
            #        stillrun = False
            #        plt.close('all')
            #        plt.pause(0.1)
    
    # Save figure
    #if plotRealTime:
    #    plt.savefig('rlbfinal.png',dpi=240)
    #    plt.show()
        
    return 0



if __name__== "__main__":
    gen = main()
    while stillrun:
        next(gen)

