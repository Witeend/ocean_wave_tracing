import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.animation as animation
import cmocean
import xarray as xa
import cartopy.crs as ccrs
from IPython.display import HTML

import sys
sys.path.insert(1, '/Users/tpvan/Waves/ocean_wave_tracing')

from ocean_wave_tracing.ocean_wave_tracing import Wave_tracing

def smooth(depth, loops, maximum):
    Y, X = depth.shape
    for k in range(loops):
        for y in range(Y):
            for x in range(X - 1):
                diff = depth[y, x + 1] - depth[y, x]
                if diff > maximum:
                    depth[y, x + 1] -= 0.5 * diff
                    depth[y, x] += 0.5 * diff
                if -diff > maximum:
                    depth[y, x + 1] -= 0.5 * diff
                    depth[y, x] += 0.5 * diff
        for x in range(X):
            for y in range(Y - 1):
                diff = depth[y + 1, x] - depth[y, x]
                if diff > maximum:
                    depth[y + 1, x] -= 0.5 * diff
                    depth[y, x] += 0.5 * diff
                if -diff > maximum:
                    depth[y + 1, x] -= 0.5 * diff
                    depth[y, x] += 0.5 * diff
    return depth

def bar(d_off=5, d_bw=2, T_wave=10):
    nx = 300; ny = 300 # number of grid points in x- and y-direction
    x = np.linspace(0,2000,nx) # size x-domain [m]
    y = np.linspace(0,1000,ny) # size y-domain [m]
    T = 400 # simulation time [s]
    U=np.zeros((ny,nx))
    V=np.zeros((ny,nx))
    # U[ny//2:,:]=1
    
    nt=100
    rays=60
    
    d = np.ones((ny, nx)) * d_off
    
    for i in range(ny):
        d[i, :] -= np.ones(nx) * d_off * (1 - i / (ny - 1))
    
    d[int(ny//2-ny/20):int(ny//2+ny/20), int(nx//2-nx/4):int(nx//2+nx/4)] = np.ones((int(ny/10), int(nx/2))) * d_bw
    
    d = smooth(d, 50, 0.2)
    
    # xv, yv = np.meshgrid(x, y)
    # fig, ax = plt.subplots(figsize=(12, 5));
    # pc=ax.pcolormesh(xv,yv,d,shading='auto', cmap=cmocean.cm.deep)
    # dc=fig.colorbar(pc)
    # dc.set_label('depth [m]')
    
    # plt.xlabel('[m]')
    # plt.ylabel('[m]')
    
    # Define a wave tracing object
    wt = Wave_tracing(U=U,V=V,
                           nx=nx, ny=ny, nt=nt,T=T,
                           dx=x[1]-x[0],dy=y[1]-y[0],
                           nb_wave_rays=rays,
                           domain_X0=x[0], domain_XN=x[-1],
                           domain_Y0=y[0], domain_YN=y[-1],
                           d=d)
    
    # Set initial conditions
    wt.set_initial_condition(wave_period=T_wave,
                              theta0=-0.3*np.pi, ipx=np.linspace(-2000, 2000, rays), ipy=np.ones(rays)*1000)
    
    # Solve
    wt.solve()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5));
    # pc=ax.pcolormesh(wt.x,wt.y,wt.U.isel(time=0),shading='auto', cmap='Wistia')
    pc=ax.pcolormesh(wt.x,wt.y,wt.d,shading='auto', cmap=cmocean.cm.deep, vmin=0, vmax=d_off, zorder=-1)
    dc=fig.colorbar(pc)
    dc.set_label('depth [m]')
    
    for ray_id in range(wt.nb_wave_rays):
        ax.plot(wt.ray_x[ray_id,:],wt.ray_y[ray_id,:],'-r', zorder=0)
    
    for ray_id in range(wt.nb_wave_rays):
        for time_id in range(0, nt, 3):
            ax.scatter(wt.ray_x[ray_id,time_id],wt.ray_y[ray_id,time_id], edgecolors='black', c='white', s=20, zorder=1)
    
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    
    plt.xlabel('[m]')
    plt.ylabel('[m]')
    
    # ax.quiver(x, y, U ,V)
    plt.show()
    
    def update(idt):
        ax.clear()
        pc=ax.pcolormesh(wt.x,wt.y,wt.d,shading='auto', cmap=cmocean.cm.deep)
        ax.set_xlim([x.min(),x.max()])
        ax.set_ylim([y.min(),y.max()])
        ax.set_xlabel('[m]')
        ax.set_ylabel('[m]')
        for ray_id in range(0, wt.nb_wave_rays, 1):
            ax.plot(wt.ray_x[ray_id,:idt],wt.ray_y[ray_id,:idt],'-r')
            
    ani = animation.FuncAnimation(fig, update, interval=100, repeat=True, repeat_delay=10000, frames=nt)
    ani.save("bar.gif")

    return ani