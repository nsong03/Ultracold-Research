from .dependencies import *
from .settings import *


from .dependencies import *
from .settings import *
from .general import *
def montecarlo_3D(force_func, globalvariables, initialdistribution3D, atommass):
    ''' Monte Carlo simulation of a distribution of particles in a potential landscape.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
        # Assuming that force is in units of kg*m/s^2, and mass is in kg, then acceleration is in m/s^2
    # Need to convert force from units of meters to units of pixels to make things computationally efficient
    # ddx = forces / atommass # m/s^2
    x0 = tocupy(initialdistribution3D[0,0]) # Right now its in pixels
    dx0 = tocupy(initialdistribution3D[1,0]) # Right now its in m/s
    y0 = tocupy(initialdistribution3D[0,1]) # Right now its in pixels
    dy0 = tocupy(initialdistribution3D[1,1]) # Right now its in m/s
    z0 = tocupy(initialdistribution3D[0,2]) # Right now its in pixels
    dz0 = tocupy(initialdistribution3D[1,2]) # Right now its in m/s
    x_t1 = x0
    dx_t1 = dx0
    y_t1 = y0
    dy_t1 = dy0
    z_t1 = z0
    dz_t1 = dz0
    # x_t2 = x0
    # dx_t2 = dx0
    for iteration in range(len(ddx)):
        # ddx_frame = ddx[iteration]
        ideal_depth = 10
        trap_x=0
        trap_rayleigh=1
        trap_rad=0.5
        trap_xfocalshift=2
        ddx_t1,ddy_t1,ddz_t1 = (1/atommass) * heuristic_3D_force([x_t1,y_t1,z_t1],\
        ideal_depth,trap_x,trap_rayleigh,trap_rad,trap_xfocalshift)    # force_func as a function of t and (x_t1,y_t1,z_t1)/atommass
        dx_t2 = dx_t1 + ddx_t1 * timestep
        x_t2 = x_t1 + dx_t1 * timestep / pixelsize_fourier
        dy_t2 = dy_t1 + ddy_t1 * timestep
        y_t2 = y_t1 + dy_t1 * timestep / pixelsize_fourier
        dz_t2 = dz_t1 + ddz_t1 * timestep
        z_t2 = z_t1 + dz_t1 * timestep / pixelsize_fourier
        dx_t1 = dx_t2
        x_t1 = x_t2
        dy_t1 = dy_t2
        y_t1 = y_t2
        dz_t1 = dz_t2
        z_t1 = z_t2
    return np.array([x_t1,y_t1,z_t1]), np.array([dx_t1,dy_t1,dz_t1]), np.array([ddx_t1,ddy_t1,ddz_t1])


def heuristic_3D_potential(atom_loc,ideal_depth,trap_x,trap_rayleigh,trap_rad,trap_xfocalshift):
    [x,y,z] = atom_loc
    rad_y = trap_rad * np.sqrt(1+(z/trap_rayleigh)**2)
    rad_x = trap_rad * np.sqrt(1+((z-trap_xfocalshift)/trap_rayleigh)**2)
    normalized_intensity = np.exp(-2*(x-trap_x)**2/rad_x**2)*np.exp(-2*y**2/rad_y**2)*trap_rad**2/(rad_x*rad_y)
    return -ideal_depth * normalized_intensity
def heuristic_3D_force(atom_loc,ideal_depth,trap_x,trap_rayleigh,trap_rad,trap_xfocalshift):
    [x,y,z] = atom_loc
    rad_y = trap_rad * np.sqrt(1+(z/trap_rayleigh)**2)
    rad_x = trap_rad * np.sqrt(1+((z-trap_xfocalshift)/trap_rayleigh)**2)
    normalized_intensity = np.exp(-2*(x-trap_x)**2/rad_x**2)*np.exp(-2*y**2/rad_y**2)*trap_rad**2/(rad_x*rad_y)
    normalized_force_x = normalized_intensity * (4*(x-trap_x)/rad_x**2)
    normalized_force_y = normalized_intensity * (4*y/rad_y**2)
    temp_A = z/(trap_rayleigh**2+z**2)
    temp_B = (z-trap_xfocalshift)/(trap_rayleigh**2+(z-trap_xfocalshift)**2)
    temp_C = temp_A*(1-4*y**2/rad_y**2) + temp_B*(1-4*(x-trap_x)**2/rad_x**2)
    normalized_force_z = normalized_intensity * temp_C
    return -ideal_depth * np.array([normalized_force_x,normalized_force_y,normalized_force_z])