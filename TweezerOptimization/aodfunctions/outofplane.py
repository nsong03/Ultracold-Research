from .dependencies import *
from .settings import *
from .general import *

import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, Lens, nm, mm, cm, RectangularSlit


def interpolate_potential_cp(potential_slices, xspacing, zspacing):
    """
    Interpolate a 2D potential profile from slices using CuPy.

    Parameters:
    potential_slices (cupy.ndarray): Array of 1D arrays, each representing a slice of the potential along the x-axis.
    xspacing (float): Spatial spacing of each pixel in each frame.
    zspacing (float): Spatial spacing in the z-direction between each frame.

    Returns:
    cupy.ndarray: Interpolated 2D potential profile.
    """
    potential_slices = cp.asarray(potential_slices)
    num_slices, num_points = potential_slices.shape
    z_coords = cp.arange(num_slices) * zspacing
    x_coords = cp.arange(num_points) * xspacing

    zz, xx = cp.meshgrid(z_coords, x_coords, indexing='ij')
    return potential_slices, zz, xx

def compute_gradient_cp(potential_2d, xspacing, zspacing):
    """
    Compute the gradient of a 2D potential profile using CuPy with second-order central differences.

    Parameters:
    potential_2d (cupy.ndarray): 2D potential profile with shape (numframes, lenframe).
    xspacing (float): Spatial spacing of each pixel in each frame.
    zspacing (float): Spatial spacing in the z-direction between each frame.

    Returns:
    tuple: Gradients in the x and z directions.
    """
    numframes, lenframe = potential_2d.shape
    
    # Initialize gradients
    grad_x = cp.zeros_like(potential_2d)
    grad_z = cp.zeros_like(potential_2d)
    
    # Compute second-order central differences for interior points
    grad_x[:, 2:-2] = (-potential_2d[:, 4:] + 8 * potential_2d[:, 3:-1] - 8 * potential_2d[:, 1:-3] + potential_2d[:, :-4]) / (12 * xspacing)
    grad_z[2:-2, :] = (-potential_2d[4:, :] + 8 * potential_2d[3:-1, :] - 8 * potential_2d[1:-3, :] + potential_2d[:-4, :]) / (12 * zspacing)
    
    # Handle boundaries with first-order differences
    grad_x[:, 0] = (potential_2d[:, 1] - potential_2d[:, 0]) / xspacing
    grad_x[:, 1] = (potential_2d[:, 2] - potential_2d[:, 0]) / (2 * xspacing)
    grad_x[:, -2] = (potential_2d[:, -1] - potential_2d[:, -3]) / (2 * xspacing)
    grad_x[:, -1] = (potential_2d[:, -1] - potential_2d[:, -2]) / xspacing
    
    grad_z[0, :] = (potential_2d[1, :] - potential_2d[0, :]) / zspacing
    grad_z[1, :] = (potential_2d[2, :] - potential_2d[0, :]) / (2 * zspacing)
    grad_z[-2, :] = (potential_2d[-1, :] - potential_2d[-3, :]) / (2 * zspacing)
    grad_z[-1, :] = (potential_2d[-1, :] - potential_2d[-2, :]) / zspacing
    
    return grad_x, grad_z

def interpolate_gradient_cp(grad_x, grad_z, x_coords, y_coords):
    """
    Interpolates the gradient at specific coordinates using bilinear interpolation with CuPy.

    Parameters:
    grad_x (cupy.ndarray): Gradient in the x direction.
    grad_z (cupy.ndarray): Gradient in the z direction.
    x_coords (list or array): x coordinates to retrieve the gradient at.
    y_coords (list or array): y coordinates to retrieve the gradient at.
    xspacing (float): Spatial spacing of each pixel in each frame.
    zspacing (float): Spatial spacing in the z-direction between each frame.

    Returns:
    tuple: Interpolated gradients at the specified coordinates.
    """
    # Convert coordinates to grid space
    x_coords = cp.array(x_coords) 
    y_coords = cp.array(y_coords) 

    x0 = cp.floor(x_coords).astype(cp.int32)
    x1 = x0 + 1
    y0 = cp.floor(y_coords).astype(cp.int32)
    y1 = y0 + 1

    x0 = cp.clip(x0, 0, grad_x.shape[1] - 1)
    x1 = cp.clip(x1, 0, grad_x.shape[1] - 1)
    y0 = cp.clip(y0, 0, grad_x.shape[0] - 1)
    y1 = cp.clip(y1, 0, grad_x.shape[0] - 1)

    Ia_x = grad_x[y0, x0]
    Ib_x = grad_x[y1, x0]
    Ic_x = grad_x[y0, x1]
    Id_x = grad_x[y1, x1]

    Ia_z = grad_z[y0, x0]
    Ib_z = grad_z[y1, x0]
    Ic_z = grad_z[y0, x1]
    Id_z = grad_z[y1, x1]

    wa = (x1 - x_coords) * (y1 - y_coords)
    wb = (x1 - x_coords) * (y_coords - y0)
    wc = (x_coords - x0) * (y1 - y_coords)
    wd = (x_coords - x0) * (y_coords - y0)

    interpolated_grad_x = wa * Ia_x + wb * Ib_x + wc * Ic_x + wd * Id_x
    interpolated_grad_z = wa * Ia_z + wb * Ib_z + wc * Ic_z + wd * Id_z

    return interpolated_grad_x, interpolated_grad_z

def snapshots_oop_potential(AWGframe_E, netpower, znumoffsets, zstart, zspacing, frameheight_real, framesizes, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    ''' Get znumoffsets snapshots of the fourier transform, starting at zstart and propagating through znumoffsets frames at zspacing. 
    Frameheight_real is the number of y-frames to use (pick an odd number to avoid interpolating), while framsizes determines the
    x and y padding beyond the start and end locations. Netpower is for power conservation, but only of ONE
    1D frame at the FOCAL point. So be sure to get only one slice from the bluestein propagation.'''
    xextent = framesizes[0]
    yextent = framesizes[1]

    AWGframe_E = array_1dto2d(AWGframe_E, frameheight_real)
    
    snapshotsout = cp.zeros((znumoffsets, numpix_frame))
    
    F = MonochromaticField(
        wavelength = wavelength, extent_x=numpix_frame * pixelsize_real, extent_y=numpix_frame * pixelsize_real, 
        Nx=numpix_frame, Ny=frameheight_real, intensity =1.
    )
    F.E = AWGframe_E
    F.propagate(focallength)
    F.add(Lens(focallength, radius = numpix_frame * pixelsize_real / 2))
    F.zoom_propagate(zstart,
             x_interval = [startlocation - xextent, endlocation+xextent], y_interval = [- yextent, +yextent])
    I = F.get_intensity()
    I = I / cp.sum(cp.abs(I)) * netpower
    Icut = I[(len(I)-1)//2]
    snapshotsout[0, :] = Icut.astype(cp.float32)
    
    for i in range(1,znumoffsets):
        F.propagate(zspacing)
        I = F.get_intensity()
        I = I / cp.sum(cp.abs(I)) * netpower
        Icut = I[(len(I)-1)//2]
        snapshotsout[i, :] = Icut.astype(cp.float)
        
    return snapshotsout

def retrieve_oop_potentials(AWGwaveform, znumoffsets, zstart, zspacing, frameheight_real, framesizes, globalvariables,
                            timeperframe = 1):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = snapshots_oop_potential(AWGinitguessexponential[0:500], 1, 1, focallength, 0,
                                  frameheight_real, framesizes, globalvariables)
    calibrationshot_max = cp.max(calibrationshot)
    normalizedenergy = tweezerenergy_max / calibrationshot_max

    num_snapshots = (len(AWGwaveform) - numpix_frame) // timeperframe + 1
    
    # Create a view of the input array with overlapping windows
    strides = (AWGwaveform.strides[0] * timeperframe, AWGwaveform.strides[0])
    shape = (num_snapshots, numpix_frame)
    snapshots = as_strided(AWGwaveform, shape=shape, strides=strides)


    def wrapper(waveformshot):
        return snapshots_oop_potential(waveformshot, normalizedenergy, znumoffsets, zstart, zspacing,
                                       frameheight_real, framesizes, globalvariables)
    
    frames_2d = cp.array([wrapper(snap) for snap in snapshots])

    # snapshots = cp.array([realtofourier_norm(zeropadframe(snap, globalvariables),calibrationshot_energy) for snap in snapshots]).astype(float)
    if timeperframe > 1:
        interpolated_snapshots = cp.zeros((num_snapshots + (num_snapshots - 1) * (timeperframe - 1), znumoffsets,numpix_frame), dtype=cp.float)
        interpolated_snapshots[::timeperframe] = frames_2d
        
        for i in range(1, timeperframe):
            interpolated_snapshots[i::timeperframe] = (frames_2d[:-1] * (timeperframe - i) + frames_2d[1:] * i) / timeperframe

        return interpolated_snapshots

    return frames_2d

def retrieve_oop_forces(AWGwaveform, znumoffsets, zstart, zspacing, frameheight_real, framesizes, globalvariables,
                            timeperframe = 1):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = snapshots_oop_potential(AWGwaveform[0:500], 1, 1, focallength, 0,
                                  frameheight_real, framesizes, globalvariables)
    calibrationshot_max = cp.max(calibrationshot)
    normalizedenergy = tweezerenergy_max / calibrationshot_max

    num_snapshots = (len(AWGwaveform) - numpix_frame) // timeperframe + 1
    
    # Create a view of the input array with overlapping windows
    strides = (AWGwaveform.strides[0] * timeperframe, AWGwaveform.strides[0])
    shape = (num_snapshots, numpix_frame)
    snapshots = as_strided(AWGwaveform, shape=shape, strides=strides)


    def wrapper(waveformshot):
        return snapshots_oop_potential(waveformshot, normalizedenergy, znumoffsets, zstart, zspacing,
                                       frameheight_real, framesizes, globalvariables)
    
    frames_2d = cp.array([wrapper(snap) for snap in snapshots])

    frame_xspacing = (framesizes[0] * 2 + cp.abs(startlocation - endlocation)) / numpix_frame
    def wrapper_getforces(potential):
        # Interpolate potential
        potential_slices, zz, xx = interpolate_potential_cp(potential, frame_xspacing, zspacing)
        # Compute gradient
        grad_x, grad_z = compute_gradient_cp(potential_slices, frame_xspacing, zspacing)
        return cp.array([grad_x, grad_z])

    # snapshots = cp.array([realtofourier_norm(zeropadframe(snap, globalvariables),calibrationshot_energy) for snap in snapshots]).astype(float)
    if timeperframe > 1:
        interpolated_snapshots = cp.zeros((num_snapshots + (num_snapshots - 1) * (timeperframe - 1), znumoffsets,numpix_frame), dtype=cp.float)
        interpolated_snapshots[::timeperframe] = frames_2d
        
        for i in range(1, timeperframe):
            interpolated_snapshots[i::timeperframe] = (frames_2d[:-1] * (timeperframe - i) + frames_2d[1:] * i) / timeperframe

        interpolated_forces = cp.array([wrapper_getforces(potential) for potential in interpolated_snapshots])

        return interpolated_forces

    interpolated_forces = cp.array([wrapper_getforces(potential) for potential in frames_2d])

    return interpolated_forces

def retrieve_oop_forces_ideal(AWGwaveform, znumoffsets, zstart, zspacing, frameheight_real, framesizes, globalvariables,
                            timeperframe = 1):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = snapshots_oop_potential(AWGwaveform[0:500], 1, 1, focallength, 0,
                                  frameheight_real, framesizes, globalvariables)
    calibrationshot_max = cp.max(calibrationshot)
    normalizedenergy = tweezerenergy_max / calibrationshot_max

    num_snapshots = (len(AWGwaveform) - numpix_frame) // timeperframe + 1
    
    # Create a view of the input array with overlapping windows
    strides = (AWGwaveform.strides[0] * timeperframe, AWGwaveform.strides[0])
    shape = (num_snapshots, numpix_frame)
    snapshots = as_strided(AWGwaveform, shape=shape, strides=strides)


    idealsnapshot =snapshots_oop_potential(AWGwaveform[0:500], normalizedenergy, znumoffsets, zstart, zspacing,
                                       frameheight_real, framesizes, globalvariables)
    
    def wrapper(waveformshot):
        return idealsnapshot
    
    frames_2d = cp.array([wrapper(snap) for snap in snapshots])

    frame_xspacing = (framesizes[0] * 2 + cp.abs(startlocation - endlocation)) / numpix_frame
    def wrapper_getforces(potential):
        # Interpolate potential
        potential_slices, zz, xx = interpolate_potential_cp(potential, frame_xspacing, zspacing)
        # Compute gradient
        grad_x, grad_z = compute_gradient_cp(potential_slices, frame_xspacing, zspacing)
        return cp.array([grad_x, grad_z])

    # snapshots = cp.array([realtofourier_norm(zeropadframe(snap, globalvariables),calibrationshot_energy) for snap in snapshots]).astype(float)
    if timeperframe > 1:
        interpolated_snapshots = cp.zeros((num_snapshots + (num_snapshots - 1) * (timeperframe - 1), znumoffsets,numpix_frame), dtype=cp.float)
        interpolated_snapshots[::timeperframe] = frames_2d
        
        for i in range(1, timeperframe):
            interpolated_snapshots[i::timeperframe] = (frames_2d[:-1] * (timeperframe - i) + frames_2d[1:] * i) / timeperframe

        interpolated_forces = cp.array([wrapper_getforces(potential) for potential in interpolated_snapshots])

        return interpolated_forces

    interpolated_forces = cp.array([wrapper_getforces(potential) for potential in frames_2d])

    return interpolated_forces

def initdistribution_MaxwellBoltzmann3D(num_particles, temperature, positionstd, zstart, zspacing, frame_sizes, globalvariables):
    """
    Generates a Maxwell-Boltzmann distribution of particles' positions and velocities. In units of fourier pixels / timestep

    Parameters:
    - num_particles (int): Number of particles.
    - mass (float): Mass of each particle.
    - temperature (float): Temperature in Kelvin.
    - kb (float, optional): Boltzmann constant. Default is 1.38e-23 J/K.

    Returns:
    - positions (np.ndarray): Array of positions of particles. x is direction of motion, z is tweezer propagation direction.
    - velocities (np.ndarray): Array of velocities of particles.
    """
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    
    frame_xspacing = (frame_sizes[0] * 2 + cp.abs(startlocation - endlocation)) / numpix_frame # in units of meters / FRAME pixel
    x0 = tonumpy(frame_sizes[0] / frame_xspacing )# In units of FRAME pixels
        
    # Standard deviation for velocity from Maxwell-Boltzmann distribution
    kb = 1.38*10**(-23)
    energy = 1/2 * kb * temperature
    std_velocity = np.sqrt(2 * energy / atommass)
    std_velocity = (std_velocity) # velocity in terms of pixels / timestep
    std_position = tonumpy(positionstd / frame_xspacing) # pixel position
    # Generating velocities
    velocitiesx = np.random.normal(0, std_velocity, (num_particles,1)) # in units of m/s
    velocitiesx[velocitiesx > 2* np.std(velocitiesx)] *= 0.5
    velocitiesy = np.random.normal(0, std_velocity, (num_particles,1)) # in units of m/s
    velocitiesy[velocitiesy > 2* np.std(velocitiesy)] *= 0.5
    velocitiesz = np.random.normal(0, std_velocity, (num_particles,1)) # in units of m/s
    velocitiesz[velocitiesz > 2* np.std(velocitiesz)] *= 0.5
    # Generating positions (assuming normal distribution centered at 0 with some spread) # in units of FRAME pixels
    positionsx = np.random.normal(0, std_position, (num_particles,1)) + x0
    positionsy = np.random.normal(0, std_position, (num_particles,1)) 
    positionsz = np.random.normal(0, std_position, (num_particles,1)) + (focallength - zstart) / zspacing

    positions = np.array([positionsx, positionsy, positionsz])
    velocities = np.array([velocitiesx, velocitiesy, velocitiesz])
    return positions, velocities

def montecarlo_oop_2D(forces, initdistribution3D, atommass, frame_sizes, zspacing, globalvariables, numframes=10):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    framespacing = len(forces) // numframes
    
    frame_xspacing = (frame_sizes[0] * 2 + cp.abs(startlocation - endlocation)) / numpix_frame

    x0 = tocupy(initdistribution3D[0][0]) # Right now its in FRAME pixels
    dx0 = tocupy(initdistribution3D[1][0]) # Right now its in m/s
    z0 = tocupy(initdistribution3D[0][2]) # Right now its in FRAME pixels
    dz0 = tocupy(initdistribution3D[1][2]) # Right now its in m/s
    x_t1 = x0
    dx_t1 = dx0
    z_t1 = z0
    dz_t1 = dz0

    atommoveframes = []
    atommoveframes.append(cp.array([x_t1,z_t1]))

    for iteration in range(len(forces)):
        ddx_frame = forces[iteration][0] / atommass # gradient in terms of m/s^2 now, with coordinates 1 pixel -> frame_xspacing 
        ddz_frame = forces[iteration][1] / atommass # gradient in terms of m/s^2 now, with coordinates 1 pixel -> zspacing

        ddx_t1,ddz_t1 = interpolate_gradient_cp(ddx_frame, ddz_frame, x_t1, z_t1) # in units of m/s^2 
        dx_t2 = dx_t1 + ddx_t1 * timestep # in units of m/s
        x_t2 = x_t1 + dx_t1 * timestep / frame_xspacing  # in units of FRAME pixels
        dz_t2 = dz_t1 + ddz_t1 * timestep
        z_t2 = z_t1 + dz_t1 * timestep / frame_xspacing
        dx_t1 = dx_t2
        x_t1 = x_t2
        dz_t1 = dz_t2
        z_t1 = z_t2
        if (iteration % framespacing == 0) and ((len(forces) - iteration) > framespacing):
            atommoveframes.append(cp.array([x_t2,z_t2]))

    atommoveframes.append(cp.array([x_t1,z_t1]))

    return cp.array([x_t1,z_t1]), cp.array([dx_t1,dz_t1]), cp.array([ddx_t1,ddz_t1]), cp.array(atommoveframes)

def sum_and_plot_intensity_arrays(intensity_arrays, num_to_sum):
    """
    Sum a specified number of evenly spaced 2D CuPy intensity arrays and plot the result with adjustable figure size.

    Parameters:
    intensity_arrays (list of cupy.ndarray): List of 2D CuPy intensity arrays.
    num_to_sum (int): Number of arrays to sum.
    fig_size (tuple): Size of the plot figure.
    """
    # Ensure num_to_sum does not exceed the length of intensity_arrays
    num_to_sum = min(num_to_sum, len(intensity_arrays))
    
    # Calculate indices of evenly spaced arrays
    indices = cp.linspace(0, len(intensity_arrays) - 1, num_to_sum, dtype=cp.int32)
    
    # Sum the specified number of evenly spaced intensity arrays
    selected_arrays = [intensity_arrays[idx] for idx in indices]
    summed_array = cp.sum(cp.stack(selected_arrays), axis=0)
    
    # Transfer the result to the host for plotting
    summed_array_host = tonumpy(summed_array)
    
    # Plot the summed intensity array
    plt.imshow(summed_array_host, cmap='viridis')
    plt.colorbar(label='Intensity', shrink = 0.1)
    plt.title(f'Sum of {num_to_sum} Evenly Spaced Intensity Arrays')
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
    plt.show()

def analyze_survivalprobability_oop_2D(pout, finalposition, tweezerwidths, globalvariables):
    """
    Calculate the percentage of values in xout that are within 1 Gaussian width of the final position.
    
    Parameters:
    xout (cp.ndarray): Array of values to analyze.
    gaussianwidth (float): The width of the Gaussian.
    globalvariables (dict): A dictionary of global variables.
    
    Returns:
    float: The percentage of values within 1 Gaussian width of the final position.
    """
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    # finalposition, _ = positionstofourier(endlocation, 0, globalvariables)
    
    pout_x = pout[0] # Everything in terms of Kung-Fu fighting (FRAME pixels)
    pout_z = pout[1]
    finalposition_x = finalposition[0]
    finalposition_z = finalposition[1]
    tweezerwidth_x = tweezerwidths[0]
    tweezerwidth_z = tweezerwidths[1]
    
    # Calculate the lower and upper bounds
    lower_bound_x = finalposition_x - tweezerwidth_x
    upper_bound_x = finalposition_x + tweezerwidth_x
    
    lower_bound_z = finalposition_z - tweezerwidth_z
    upper_bound_z = finalposition_z + tweezerwidth_z
    
    # Count the number of values within the bounds
    count_within_bounds = np.sum((pout_x >= lower_bound_x) & (pout_x <= upper_bound_x) & (pout_z <= upper_bound_z) & (pout_z >= lower_bound_z))
    # Calculate the percentage
    percentage_within_bounds = count_within_bounds / len(pout_x) * 100.0
    
    return percentage_within_bounds

def analyze_fixeddistance_nonoptimized_oop_2D(movementtimes, initialtemperatures, oop_variables, responsetype="Exponential",calctype="Not Ideal", guesstype = "SinSq", timeperframe=1, globalvariables=globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    numzframes, zstart, frame_zspacing, frame_size, frameheight_real = oop_variables
    
    # Calculate the number of movement times and initial temperatures
    num_movementtimes = len(movementtimes)
    num_initialtemperatures = len(initialtemperatures)
    
    # Initialize arrays to store the results
    results = np.empty((num_movementtimes, num_initialtemperatures), dtype=object)
    movementframes = np.empty((num_movementtimes, num_initialtemperatures), dtype=object)

    # Iterate over each movement time and initial temperature
    for i in range(num_movementtimes):
        for j in range(num_initialtemperatures):
            movementtime = movementtimes[i]
            numpix_waveform = int(movementtime / cycletime * numpix_frame) + 2* numpix_frame # Why is there a 2* cycletime here? To add on the initial and final stages of the AOD. We will only change the portion in the movement time and fix the ends.
            AWGwaveform = cp.zeros(numpix_waveform)
            globalvariables = [aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients]
            
            
            # Perform the analysis for each combination of movement time and initial temperature
            if calctype =="Not Ideal":
                if guesstype == "Linear":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_linearramp(globalvariables)
                elif guesstype == "MinJerk":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_minimizejerk(globalvariables)
                elif guesstype == "SinSq":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_sinsqramp_general(globalvariables)
                    
                fourierpixels, time = positionstofourier(optimized_position, time, globalvariables)
                expanded_position, expanded_time = expand_position_array(time, fourierpixels, globalvariables)
                
                if responsetype == "Cosine":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = cosinephaseresponse(AWGinput)                
                elif responsetype =="Exponential":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = exponentialphaseresponse(AWGinput)     
                
                waveform_forces = retrieve_oop_forces(AWGphase, numzframes, zstart, frame_zspacing, frameheight_real, frame_size, globalvariables, timeperframe)
                initdistribution = initdistribution_MaxwellBoltzmann3D(num_particles, initialtemperatures[j], 0, zstart, frame_zspacing, frame_size, globalvariables)
                p_out, dp_out, ddp_out, atommoveframes = montecarlo_oop_2D(waveform_forces, initdistribution, atommass, frame_size, frame_zspacing, globalvariables)

                frame_xspacing = (frame_size[0] * 2 + cp.abs(startlocation - endlocation)) / numpix_frame # in units of meters / FRAME pixel
                finalposition_x = (frame_size[0]  + cp.abs(startlocation - endlocation)) / frame_xspacing # In units of FRAME pixels
                finalposition_z = (focallength - zstart) / frame_zspacing
                finalposition = [finalposition_x, finalposition_z]
                
                
                calibration_potential = snapshots_oop_potential(AWGphase[-numpix_frame:], 1, numzframes, zstart, frame_zspacing, 
                                                                frameheight_real, frame_size, globalvariables)
                tweezerwidths = fit_gaussian_2d(tonumpy(calibration_potential))
                percentagelive = analyze_survivalprobability_oop_2D(p_out, finalposition, tweezerwidths, globalvariables)
                                
                
                # Store the result in the results array
                results[i, j] = [tonumpy(percentagelive),tonumpy(p_out),tonumpy(dp_out)]
                movementframes[i, j] = [atommoveframes]

        
            if calctype =="Ideal":
                if guesstype == "Linear":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_linearramp(globalvariables)
                elif guesstype == "MinJerk":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_minimizejerk(globalvariables)
                elif guesstype == "SinSq":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_sinsqramp_general(globalvariables)
                    
                fourierpixels, time = positionstofourier(optimized_position, time, globalvariables)
                expanded_position, expanded_time = expand_position_array(time, fourierpixels, globalvariables)
                
                if responsetype == "Cosine":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = cosinephaseresponse(AWGinput)                
                elif responsetype =="Exponential":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = exponentialphaseresponse(AWGinput)     
                
                waveform_forces = retrieve_oop_forces_ideal(AWGphase, numzframes, zstart, frame_zspacing, frameheight_real, frame_size, globalvariables, timeperframe)
                initdistribution = initdistribution_MaxwellBoltzmann3D(num_particles, initialtemperatures[j], 0, zstart, frame_zspacing, frame_size, globalvariables)
                p_out, dp_out, ddp_out, atommoveframes = montecarlo_oop_2D(waveform_forces, initdistribution, atommass, frame_size, frame_zspacing, globalvariables)

                frame_xspacing = (frame_size[0] * 2 + cp.abs(startlocation - endlocation)) / numpix_frame # in units of meters / FRAME pixel
                finalposition_x = (frame_size[0]  + cp.abs(startlocation - endlocation)) / frame_xspacing # In units of FRAME pixels
                finalposition_z = (focallength - zstart) / frame_zspacing
                finalposition = [finalposition_x, finalposition_z]
                
                
                calibration_potential = snapshots_oop_potential(AWGphase[-numpix_frame:], 1, numzframes, zstart, frame_zspacing, 
                                                                frameheight_real, frame_size, globalvariables)
                tweezerwidths = fit_gaussian_2d(tonumpy(calibration_potential))
                percentagelive = analyze_survivalprobability_oop_2D(p_out, finalposition, tweezerwidths, globalvariables)
                                
                
                # Store the result in the results array
                results[i, j] = [tonumpy(percentagelive),tonumpy(p_out),tonumpy(dp_out)]
                movementframes[i, j] = [atommoveframes]


    return results, movementframes

























 

# def get_longitudinal_profile(self, start_distance, end_distance, steps, scale_factor = 1):
#     """
#     Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
#     the colors and the field over the xz plane
#     """

#     z = cp.linspace(start_distance, end_distance, steps)

#     self.E0 = self.E.copy()

#     longitudinal_profile_rgb = cp.zeros((steps,self.Nx, 3))
#     longitudinal_profile_E = cp.zeros((steps,self.Nx), dtype = complex)
#     z0 = self.z 
#     t0 = time.time()


#     bar = progressbar.ProgressBar()
#     for i in bar(range(steps)):

#         if scale_factor == 1:     
#             self.propagate(z[i])
#         else:
#             self.scale_propagate(z[i], scale_factor)

#             self.extent_x/=scale_factor
#             self.extent_y/=scale_factor

#             self.dx/=scale_factor
#             self.dy/=scale_factor
#             self.x/=scale_factor
#             self.y/=scale_factor

#             self.xx/=scale_factor
#             self.yy/=scale_factor

#         rgb = self.get_colors()
#         longitudinal_profile_rgb[i,:,:]  = rgb[self.Ny//2,:,:]
#         longitudinal_profile_E[i,:] = self.E[self.Ny//2,:]
#         self.E = np.copy(self.E0)


#     # restore intial values
#     self.z = z0
#     self.I = cp.real(self.E * cp.conjugate(self.E))  

#     print ("Took", time.time() - t0)

#     extent = [self.x[0]*scale_factor, self.x[-1]*scale_factor, start_distance, end_distance]
#     return longitudinal_profile_rgb, longitudinal_profile_E, extent



# # Most of computational code is taken from 
# """

# MPL 2.0 License 

# Copyright (c) 2022, Rafael de la Fuente
# All rights reserved.

# This license prohibits others from using the project to promote derived products without written consent. Redistributions, with or without
# modification, requires giving appropriate attribution to the author for the original work. Redistributions must:

# 1. Keep the original copyright on the software
# 2. Include full text of license inside the software
# 3. You must put an attribution in all advertising materials

# Under the terms of the MPL, it also allows the integration of MPL-licensed code into proprietary codebases, but only on condition those components remain accessible.
# It grants liberal copyright and patent licenses allowing for free use, modification, distribution of the work, but does not grant the licensee any rights to a contributor's trademarks.

# """
# # with some modifications for our specific use case. Using this as a reference

# # Define fourier transform methods
# ## General utilities (FT)


# def scaled_fourier_transform(x, y, U, λ = 1,z =1, scale_factor = 1, mesh = False):
#     """ 

#     Computes de following integral:

#     Uf(x,y) = ∫∫  U(u, v) * exp(-1j*pi/ (z*λ) *(u*x + v*y)) * du*dv

#     Given the extent of the input coordinates of (u, v) of U(u, v): extent_u and extent_v respectively,
#     Uf(x,y) is evaluated in a scaled coordinate system (x, y) with:

#     extent_x = scale_factor*extent_u
#     extent_y = scale_factor*extent_v

#     """

#     Ny,Nx = U.shape    
    
#     if mesh == False:
#         dx = x[1]-x[0]
#         dy = y[1]-y[0]
#         xx, yy = cp.meshgrid(x, y)
#     else:
#         dx = x[0,1]-x[0,0]
#         dy = y[1,0]-y[0,0]
#         xx, yy = x,y

#     extent_x = dx*Nx
#     extent_y = dy*Ny

#     L1 = extent_x
#     L2 = extent_x*scale_factor

#     f_factor = 1/(λ*z)
#     fft_U = cp.fft.fftshift(cp.fft.fft2(U * cp.exp(-1j*cp.pi* f_factor*(xx**2 + yy**2) ) * cp.exp(1j*cp.pi*(L1- L2)/L1 * f_factor*(xx**2 + yy**2 ))))
    
    
#     fx = cp.fft.fftshift(cp.fft.fftfreq(Nx, d = dx))
#     fy = cp.fft.fftshift(cp.fft.fftfreq(Ny, d = dy))
#     fxx, fyy = cp.meshgrid(fx, fy)

#     Uf = cp.fft.ifft2(cp.fft.ifftshift( cp.exp(- 1j * cp.pi / f_factor * L1/L2 * (fxx**2 + fyy**2))  *  fft_U) )
    
#     extent_x = extent_x*scale_factor
#     extent_y = extent_y*scale_factor

#     dx = dx*scale_factor
#     dy = dy*scale_factor

#     x = x*scale_factor
#     y = y*scale_factor

#     xx = xx*scale_factor
#     yy = yy*scale_factor  

#     Uf = L1/L2 * cp.exp(-1j *cp.pi*f_factor* (xx**2 + yy**2)   - 1j * cp.pi*f_factor* (L1-L2)/L2 * (xx**2 + yy**2)) * Uf *1j * (λ*z)

#     if mesh == False:
#         return x, y, Uf
#     else:
#         return xx, yy, Uf

# def angular_spectrum_method(simulation, E, z, λ, scale_factor = 1):
#     """
#     Compute the field in distance equal to z with the angular spectrum method. 
#     By default (scale_factor = 1), the ouplut plane coordinates is the same than the input.
#     Otherwise, it's recommended to use the two_steps_fresnel_method as it's computationally cheaper.
#     To arbitrarily choose and zoom in a region of interest, use bluestein method instead.

#     Reference: https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html
#     """

#     # compute angular spectrum
#     fft_c = cp.fft.fft2(E)
#     c = cp.fft.fftshift(fft_c)

#     fx = cp.fft.fftshift(cp.fft.fftfreq(simulation.Nx, d = simulation.dx))
#     fy = cp.fft.fftshift(cp.fft.fftfreq(simulation.Ny, d = simulation.dy))
#     fxx, fyy = cp.meshgrid(fx, fy)

#     argument = (2 * cp.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)

#     #Calculate the propagating and the evanescent (complex) modes
#     tmp = cp.sqrt(cp.abs(argument))
#     kz = cp.where(argument >= 0, tmp, 1j*tmp)


#     if scale_factor == 1:

#         # propagate the angular spectrum a distance z
#         E = cp.fft.ifft2(cp.fft.ifftshift(c * cp.exp(1j * kz * z)))

#     else:
#         nn_, mm_ = cp.meshgrid(cp.arange(simulation.Nx)-simulation.Nx//2, cp.arange(simulation.Ny)-simulation.Ny//2)
#         factor = ((simulation.dx *simulation.dy)* cp.exp(cp.pi*1j * (nn_ + mm_)))


#         simulation.x = simulation.x*scale_factor
#         simulation.y = simulation.y*scale_factor

#         simulation.dx = simulation.dx*scale_factor
#         simulation.dy = simulation.dy*scale_factor

#         extent_fx = (fx[1]-fx[0])*simulation.Nx
#         simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, factor*c * cp.exp(1j * kz * z),  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
#         simulation.extent_x = simulation.extent_x*scale_factor
#         simulation.extent_y = simulation.extent_y*scale_factor

#     return E

# def bluestein_method(simulation, E, z, λ, x_interval, y_interval):
#     """
#     Compute the field in distance equal to z with the Bluestein method. 
#     Bluestein method is the more versatile one as the dimensions of the output plane can be arbitrarily chosen by using 
#     the arguments x_interval and y_interval

#     Parameters
#     ----------

#     x_interval: A length-2 sequence [x1, x2] giving the x outplut plane range
#     y_interval: A length-2 sequence [y1, y2] giving the y outplut plane range

#     Reference: 
#     Hu, Y., Wang, Z., Wang, X. et al. Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. 
#     Light Sci Appl 9, 119 (2020).
#     """

#     E = bluestein_fft2(E * cp.exp(1j * 2*cp.pi/λ /(2*z) *(simulation.xx**2 + simulation.yy**2)), 
#                         x_interval[0] / (z*λ), x_interval[1] / (z*λ), 1/simulation.dx, 
#                         y_interval[0] / (z*λ), y_interval[1] / (z*λ), 1/simulation.dy)

#     dfx = 1/(simulation.Nx*simulation.dx)
#     dfy = 1/(simulation.Ny*simulation.dy)

#     fx_zfft = bluestein_fftfreq(x_interval[0]/ (z*λ),x_interval[1]/ (z*λ), simulation.Nx)
#     fy_zfft = bluestein_fftfreq(y_interval[0]/ (z*λ),y_interval[1]/ (z*λ), simulation.Ny)
#     dfx_zfft = fx_zfft[1]-fx_zfft[0]
#     dfy_zfft = fy_zfft[1]-fy_zfft[0]


#     nn, mm = cp.meshgrid((cp.linspace(0,(simulation.Nx-1),simulation.Nx)*dfx_zfft/dfx ), (cp.linspace(0,(simulation.Ny-1),simulation.Ny)*dfy_zfft/dfy ))
#     factor = (simulation.dx*simulation.dy* cp.exp(cp.pi*1j * (nn + mm)))


#     simulation.x = fx_zfft*(z*λ)
#     simulation.y = fy_zfft*(z*λ)

#     simulation.xx, simulation.yy = cp.meshgrid(simulation.x, simulation.y)

#     simulation.dx = simulation.x[1] - simulation.x[0]
#     simulation.dy = simulation.y[1] - simulation.y[0]

#     simulation.extent_x = simulation.x[1] - simulation.x[0] + simulation.dx
#     simulation.extent_y = simulation.y[1] - simulation.y[0] + simulation.dy

#     return E*factor * cp.exp(1j*cp.pi/(λ*z)  * (simulation.xx**2 + simulation.yy**2)  +   1j*2*cp.pi/λ * z ) / (1j*z*λ)

# def bluestein_fft(x, axis, f0, f1, fs, M):
#     """
#     bluestein FFT function to evaluate the DFT
#     coefficients for the rows of an array in the frequency range [f0, f1]
#     using N points.
    
#     Parameters
#     ----------

#     x: array to evaluate DFT (along last dimension of array)
#     f0: lower bound of frequency bandwidth
#     f1: upper bound of frequency bandwidth
#     fs: sampling frequency
#     M: number of points used when evaluating the 1DFT (N <= signal length)
#     axis: axis along which the fft's are computed (defaults to last axis)


#     Reference: 
    
#     Leo I. Bluestein, “A linear filtering approach to the computation of the discrete Fourier transform,” 
#     Northeast Electronics Research and Engineering Meeting Record 10, 218-219 (1968).
#     """

#     # Swap axes
#     x = cp.swapaxes(a=x, axis1=axis, axis2=-1)

#     # Normalize frequency range
#     phi0 = 2.0 * cp.pi * f0 / fs
#     phi1 = 2.0 * cp.pi * f1 / fs
#     d_phi = (phi1 - phi0) / (M - 1)

#     # Determine shape of signal
#     A = cp.exp(1j * phi0)
#     W = cp.exp(-1j * d_phi)
#     X = chirpz(x=x, A=A, W=W, M=M)

#     return cp.swapaxes(a=X, axis1=axis, axis2=-1)

# def bluestein_fft2(U, fx0, fx1, fxs,   fy0, fy1, fys):
#     """
#     bluestein FFT function to evaluate the 2DFT

#     Parameters
#     ----------

#     U: array to evaluate 2DFT

#     fx0: lower bound of x frequency bandwidth
#     fx1: upper bound of x frequency bandwidth
#     fxs: sampling x frequency

#     fy0: lower bound of y frequency bandwidth
#     fy1: upper bound of y frequency bandwidth
#     fys: sampling y frequency


#     """
#     Ny, Nx = U.shape
#     return bluestein_fft( bluestein_fft(U, f0=fy0, f1=fy1, fs=fys, M=Ny, axis=0), f0=fx0, f1=fx1, fs=fxs, M=Nx, axis=1)

# def bluestein_ifft(X, axis, f0, f1, fs, M):
#     """
#     bluestein iFFT function to evaluate the iDFT
#     coefficients for the rows of an array in the frequency range [f0, f1]
#     using N points.
    
#     Parameters
#     ----------

#     x: array to evaluate iDFT (along last dimension of array)
#     f0: lower bound of frequency bandwidth
#     f1: upper bound of frequency bandwidth
#     fs: sampling frequency
#     M: number of points used when evaluating the iDFT (N <= signal length)
#     axis: axis along which the ifft's are computed (defaults to last axis)

#     """
#     # Swap axes
#     X = cp.swapaxes(a=X, axis1=axis, axis2=-1)

#     N = X.shape[-1]

#     phi0 = f0 / fs * 2.0 * cp.pi / N
#     phi1 = f1 / fs * 2.0 * cp.pi / N
#     d_phi = (phi1 - phi0) / (M - 1)

#     A = cp.exp(-1j * phi0)
#     W = cp.exp(1j * d_phi)
#     x = chirpz(x=X, A=A, W=W, M=M) / N

#     return cp.swapaxes(a=x, axis1=axis, axis2=-1)

# def bluestein_ifft2(U, fx0, fx1, fxs,   fy0, fy1, fys):
#     """
#     bluestein iFFT function to evaluate the i2DFT

#     Parameters
#     ----------

#     U: array to evaluate 2DFT

#     fx0: lower bound of x frequency bandwidth
#     fx1: upper bound of x frequency bandwidth
#     fxs: sampling x frequency

#     fy0: lower bound of y frequency bandwidth
#     fy1: upper bound of y frequency bandwidth
#     fys: sampling y frequency


#     """
#     Ny, Nx = U.shape
#     return bluestein_ifft( bluestein_ifft(U, f0=fy0, f1=fy1, fs=fys, M=Ny, axis=0), f0=fx0, f1=fx1, fs=fxs, M=Nx, axis=1)

# def bluestein_fftfreq(f0, f1, M):
#     """
#     Return frequency values of the bluestein FFT
#     coefficients returned by bluestein_fft().
    
#     Parameters
#     ----------

#     f0: lower bound of frequency bandwidth
#     f1: upper bound of frequency bandwidth
#     fs: sampling rate
    
#     """

#     df = (f1 - f0) / (M - 1)
#     return cp.arange(M) * df + f0

# def chirpz(x, A, W, M):
#     """
    
#     Parameters
#     ----------

#     x: array to evaluate chirp-z transform (along last dimension of array)
#     A: starting point of chirp-z contour
#     W: controls frequency sample spacing and shape of the contour
#     M: number of frequency sample points

#     Reference:
#     Rabiner, L.R., R.W. Schafer and C.M. Rader. The Chirp z-Transform
#     Algorithm. IEEE Transactions on Audio and Electroacoustics,
#     AU-17(2):86--92, 1969

#     Originally Written by Stefan van der Walt: 
#     http://www.mail-archive.com/numpy-discussion@scipy.org/msg01812.html
    
#     The discrete z-transform,
#     X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
#     is calculated at M points,
#     z_k = AW^-k, k = 0,1,...,M-1
#     for A and W complex, which gives
#     X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}
#     """

#     x = cp.asarray(x, dtype=complex)
#     P = x.shape

#     N = P[-1]
#     L = int(2 ** cp.ceil(cp.log2(M + N - 1)))

#     n = cp.arange(N, dtype=float)
#     y = cp.power(A, -n) * cp.power(W, n ** 2 / 2.)
#     y = cp.tile(y, (P[0], 1)) * x
#     Y = cp.fft.fft(y, L)

#     n = cp.arange(L, dtype=float)
#     v = cp.zeros(L, dtype=complex)
#     v[:M] = cp.power(W, -n[:M] ** 2 / 2.)
#     v[L-N+1:] = cp.power(W, -(L - n[L-N+1:]) ** 2 / 2.)

#     V = cp.fft.fft(v)

#     g = cp.fft.ifft(cp.tile(V, (P[0], 1)) * Y)[:,:M]
#     k = cp.arange(M)
#     g = g * cp.tile(cp.power(W, k ** 2 / 2.), (P[0],1))

#     # Return result
#     return g

# def PSF_convolution(simulation, E, λ, PSF, scale_factor = 1):
#     """
#     Convolve the field with a the given coherent point spread function (PSF) sampled in spatial simulation coordinates.

#     Note: the angular spectrum propagation can be exactly reproduced with this method by using as PSF the Rayleigh-Sommerfeld kernel:
#     PSF = 1 / (λ) * (1/(k * r) - 1j)  * (exp(j * k * r)* z/r ) where k = 2 * pi / λ  and r = sqrt(x**2 + y**2 + z**2)
#     (Also called free space propagation impulse response function)
#     """


#     global cp

#     nn_, mm_ = cp.meshgrid(cp.arange(simulation.Nx)-simulation.Nx//2, cp.arange(simulation.Ny)-simulation.Ny//2)
#     factor = ((simulation.dx *simulation.dy)* cp.exp(cp.pi*1j * (nn_ + mm_)))


#     E_f = factor*cp.fft.fftshift(cp.fft.fft2(E))
    
#     #Definte the ATF function, representing the Fourier transform of the PSF.
#     H = factor*cp.fft.fftshift(cp.fft.fft2(PSF))


#     if scale_factor == 1:
#         return cp.fft.ifft2(cp.fft.ifftshift(E_f*H /factor ))

#     else:
#         fx = cp.fft.fftshift(cp.fft.fftfreq(simulation.Nx, d = simulation.x[1]-simulation.x[0]))
#         fy = cp.fft.fftshift(cp.fft.fftfreq(simulation.Ny, d = simulation.y[1]-simulation.y[0]))
#         fxx, fyy = cp.meshgrid(fx, fy)
#         extent_fx = (fx[1]-fx[0])*simulation.Nx
#         simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, E_f*H,  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
#         simulation.x = simulation.x*scale_factor
#         simulation.y = simulation.y*scale_factor
#         simulation.dx = simulation.dx*scale_factor
#         simulation.dy = simulation.dy*scale_factor
#         simulation.extent_x = simulation.extent_x*scale_factor
#         simulation.extent_y = simulation.extent_y*scale_factor
#         return E

# def apply_transfer_function(simulation, E, λ, H, scale_factor = 1):
#     """
#     Apply amplitude transfer function ATF (H) to the field in the frequency domain sampled in FFT simulation coordinates

#     Note: the angular spectrum method amplitude transfer function equivalent is: H = exp(1j * kz * z)
#     """

#     import matplotlib.pyplot as plt

#     if scale_factor == 1:
#         E_f = cp.fft.fftshift(cp.fft.fft2(E))
#         return cp.fft.ifft2(cp.fft.ifftshift(E_f*H ))

#     else:
#         fx = cp.fft.fftshift(cp.fft.fftfreq(simulation.Nx, d = simulation.x[1]-simulation.x[0]))
#         fy = cp.fft.fftshift(cp.fft.fftfreq(simulation.Ny, d = simulation.y[1]-simulation.y[0]))
#         fxx, fyy = cp.meshgrid(fx, fy)

#         nn_, mm_ = cp.meshgrid(cp.arange(simulation.Nx)-simulation.Nx//2, cp.arange(simulation.Ny)-simulation.Ny//2)
#         factor = ((simulation.dx *simulation.dy)* cp.exp(cp.pi*1j * (nn_ + mm_)))

#         E_f = factor*cp.fft.fftshift(cp.fft.fft2(E))

#         extent_fx = (fx[1]-fx[0])*simulation.Nx
#         simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, E_f*H,  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
#         simulation.x = simulation.x*scale_factor
#         simulation.y = simulation.y*scale_factor
#         simulation.dx = simulation.dx*scale_factor
#         simulation.

# def get_longitudinal_profile(self, start_distance, end_distance, steps, scale_factor = 1):
#     """
#     Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
#     the colors and the field over the xz plane
#     """

#     z = cp.linspace(start_distance, end_distance, steps)

#     self.E0 = self.E.copy()

#     longitudinal_profile_rgb = cp.zeros((steps,self.Nx, 3))
#     longitudinal_profile_E = cp.zeros((steps,self.Nx), dtype = complex)
#     z0 = self.z 
#     t0 = time.time()


#     bar = progressbar.ProgressBar()
#     for i in bar(range(steps)):

#         if scale_factor == 1:     
#             self.propagate(z[i])
#         else:
#             self.scale_propagate(z[i], scale_factor)

#             self.extent_x/=scale_factor
#             self.extent_y/=scale_factor

#             self.dx/=scale_factor
#             self.dy/=scale_factor
#             self.x/=scale_factor
#             self.y/=scale_factor

#             self.xx/=scale_factor
#             self.yy/=scale_factor

#         rgb = self.get_colors()
#         longitudinal_profile_rgb[i,:,:]  = rgb[self.Ny//2,:,:]
#         longitudinal_profile_E[i,:] = self.E[self.Ny//2,:]
#         self.E = np.copy(self.E0)


#     # restore intial values
#     self.z = z0
#     self.I = cp.real(self.E * cp.conjugate(self.E))  

#     print ("Took", time.time() - t0)

#     extent = [self.x[0]*scale_factor, self.x[-1]*scale_factor, start_distance, end_distance]
#     return longitudinal_profile_rgb, longitudinal_profile_E, extent


















