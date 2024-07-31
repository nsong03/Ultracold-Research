from .dependencies import *
from .settings import *

def initpath_linearramp(globalvariables):
    '''Initializes positions throughout the movementtime with an acceleration profile that is a linear ramp up for half the time then down for half the time that moves
    the atom from startlocation to endlocation.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    
    # Define the number of time steps
    num_steps = 1000
    total_time = movementtime
    time = np.linspace(0, total_time, num_steps)
    
    # Initial and final positions
    initial_position = startlocation * 10**6  # Convert to micrometers
    final_position = endlocation * 10**6  # Convert to micrometers
    D = final_position - initial_position
    
    # Calculate the constant acceleration needed
    a_max = (4 * D) / (total_time**2)  # Maximum acceleration
    
    # Initialize arrays for acceleration, velocity, position, and jerk
    accelerations = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    positions = np.zeros(num_steps)
    jerks = np.zeros(num_steps)
    
    # Calculate acceleration, velocity, and position for each time step
    half_time = total_time / 2
    half_steps = num_steps // 2
    time_step = total_time / num_steps
    
    for i in range(num_steps):
        if i <= half_steps:
            # Ramp up phase
            current_time = i * time_step
            accelerations[i] = a_max * (current_time / half_time)
        else:
            # Ramp down phase
            current_time = (i - half_steps) * time_step
            accelerations[i] = a_max * (1 - (current_time / half_time))

    def integrate_acceleration(acceleration, dt):
        velocity = np.cumsum(acceleration) * dt
        position = np.cumsum(velocity) * dt
        return velocity, position

    # Function to calculate jerk from acceleration
    def calculate_jerk(acceleration, dt):
        jerk = np.gradient(acceleration, dt, edge_order = 2)
        return jerk

    velocities, positions = integrate_acceleration(accelerations, time_step)
    jerks = calculate_jerk(accelerations, time_step)

    positions = positions / 10**6 + startlocation
    velocities = velocities / 10**6
    accelerations = accelerations / 10**6
    jerks = jerks / 10**6

    return positions,velocities,accelerations, jerks, time


def initpath_sinsqramp_general(globalvariables):
    '''Initializes positions throughout the movementtime with an acceleration profile that is a linear ramp up for half the time then down for half the time that moves
    the atom from startlocation to endlocation.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    
    # Define the number of time steps
    num_steps = 1000
    total_time = movementtime
    time = np.linspace(0, total_time, num_steps)
    
    # Initial and final positions
    initial_position = startlocation * 10**6  # Convert to micrometers
    final_position = endlocation * 10**6  # Convert to micrometers
    D = final_position - initial_position
    
    positions = np.sin(np.linspace(0, np.pi/2, num_steps))**2 * D + initial_position
    velocities = np.gradient(positions, time, edge_order = 2)
    accelerations = np.gradient(velocities, time, edge_order = 2)
    jerks = np.gradient(accelerations, time, edge_order = 2)
    
    positions = positions / 10**6
    velocities = velocities / 10**6
    accelerations = accelerations / 10**6
    jerks = jerks / 10**6
    
    return positions,velocities,accelerations, jerks, time

def initpath_general(globalvariables):
    '''Initializes positions throughout the movementtime with an acceleration profile that is a linear ramp up for half the time then down for half the time that moves
    the atom from startlocation to endlocation.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    
    # Define the number of time steps
    num_steps = 1000
    total_time = movementtime
    time = np.linspace(0, total_time, num_steps)
    
    # Initial and final positions
    initial_position = startlocation * 10**6  # Convert to micrometers
    final_position = endlocation * 10**6  # Convert to micrometers
    D = final_position - initial_position
    
    positions = np.sin(np.linspace(0, np.pi/2, num_steps))**2 * D + initial_position
    velocities = np.gradient(positions, time, edge_order = 2)
    accelerations = np.gradient(velocities, time, edge_order = 2)
    jerks = np.gradient(accelerations, time, edge_order = 2)
    
    positions = positions / 10**6
    velocities = velocities / 10**6
    accelerations = accelerations / 10**6
    jerks = jerks / 10**6
    
    return positions,velocities,accelerations, jerks, time

def initpath_sinsqramp(startposition, endposition, globalvariables):
    '''Initializes positions throughout the movementtime with an acceleration profile that is a linear ramp up for half the time then down for half the time that moves
    the atom from startlocation to endlocation.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    
    # Define the number of time steps
    num_steps = 1000
    total_time = movementtime
    time = np.linspace(0, total_time, num_steps)
    
    # Initial and final positions
    initial_position = startposition * 10**6  # Convert to micrometers
    final_position = endposition * 10**6  # Convert to micrometers
    D = final_position - initial_position
    
    positions = np.sin(np.linspace(0, np.pi/2, num_steps))**2 * D + initial_position
    velocities = np.gradient(positions, time)
    accelerations = np.gradient(velocities, time)
    jerks = np.gradient(accelerations, time)
    
    positions = positions / 10**6
    velocities = velocities / 10**6
    accelerations = accelerations / 10**6
    jerks = jerks / 10**6
    
    return positions,velocities,accelerations, jerks, time


def initpath_minimizejerk(globalvariables):
    '''Initializes positions throughout the movementtime that minimizes jerk.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    # Define the number of time steps and total time
    num_steps = 1000
    total_time = movementtime  # You may need to adjust this based on the problem's requirements
    time = np.linspace(0, 10, num_steps)
    time_out = np.linspace(0, total_time, num_steps)
    # Initial and final conditions
    initial_position = startlocation*(10**6)
    final_position = endlocation*(10**6)
    D = final_position - initial_position

    # Function to calculate velocity and position from acceleration
    def integrate_acceleration(acceleration, dt):
        velocity = np.cumsum(acceleration) * dt
        position = np.cumsum(velocity) * dt
        return velocity, position

    # Function to calculate jerk from acceleration
    def calculate_jerk(acceleration, dt):
        jerk = np.gradient(acceleration, dt)
        return jerk

    # Objective function to minimize jerk and achieve target position
    def objective(acceleration, dt):
        velocity, position = integrate_acceleration(acceleration, dt)
        jerk = calculate_jerk(acceleration, dt)
        jerk_cost = np.sum(jerk**2)  # Minimize the squared jerk
        position_error = (position[-1] - D)**2  # Ensure final position is D
        velocity_error = velocity[-1]**2  # Ensure final velocity is 0
        return jerk_cost + position_error + velocity_error

    # Initial guess for the acceleration profile
    initial_guess = np.zeros(num_steps)

    # Optimization constraints
    constraints = [
        {'type': 'eq', 'fun': lambda a: integrate_acceleration(a, time[1] - time[0])[1][-1] - D},  # Final position constraint
        {'type': 'eq', 'fun': lambda a: integrate_acceleration(a, time[1] - time[0])[0][-1]},     # Final velocity constraint
    ]

    # Perform the optimization
    result = minimize(objective, initial_guess, args=(time[1] - time[0]), constraints=constraints, method='SLSQP')

    # Extract the optimized acceleration profile
    optimized_acceleration = result.x
    optimized_velocity, optimized_position = integrate_acceleration(optimized_acceleration, time[1] - time[0])
    optimized_jerk = calculate_jerk(optimized_acceleration, time[1] - time[0])
    
    optimized_acceleration = optimized_acceleration / (10**6)
    optimized_velocity = optimized_velocity / (10**6)
    optimized_position = (optimized_position)/ (10**6) + startlocation
    optimized_jerk = optimized_jerk / (10**6)
    
    return optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time_out

def positionstofourier(positions, time, globalvariables):
    '''Converts positions from initialized paths onto pixels in the fourier plane, preserving accuracy by storing as floats. These will later be converted to desired
    spots on the AWG waveform.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    framecenter = numpix_real / 2
    fourierpixels = (positions / pixelsize_fourier) + framecenter

    return fourierpixels, time

def fouriertopositions(fourierpixels, time, globalvariables):
    '''Converts positions from initialized paths onto pixels in the fourier plane, preserving accuracy by storing as floats. These will later be converted to desired
    spots on the AWG waveform.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    framecenter = numpix_real / 2
    positions = (fourierpixels - framecenter)*pixelsize_fourier
    
    return positions, time   

def expand_position_array(time, position, globalvariables):
    '''
    Expands the position array to match the resolution of the AWGwaveform time array.
    
    Parameters:
        time (np.array): The array of time values corresponding to the position array.
        position (np.array): The array of position values.
        AWGwaveform (np.array): The higher resolution time array for the AWG waveform.
        
    Returns:
        np.array: The expanded position array.
    '''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    AWGtime = np.linspace(0,movementtime, int(movementtime / cycletime * numpix_frame))
    # Create an interpolation function based on the original time and position arrays
    interpolation_function = interp1d(tonumpy(time), tonumpy(position), kind='linear', fill_value='extrapolate')
    
    # Use the interpolation function to generate the expanded position array
    expanded_position = interpolation_function(tonumpy(AWGtime))
    
    return expanded_position, AWGtime

def realtofourier(inputarray):
    """
    Simulate the Fourier transform of a waveform.
    
    Parameters:
    inputarray (cp.ndarray): The input waveform.
    
    Returns:
    cp.ndarray: The Fourier transform of the input waveform.
    """
    # Perform the Fourier transform
    outputarray = cpfft.fftshift(cpfft.fft(cpfft.fftshift(inputarray), norm="ortho"))
    fourierintensity = cp.square (cp.abs(outputarray))
    return fourierintensity

def realtofourier_norm(inputarray, normvalue):
    """
    Simulate the Fourier transform of a waveform.
    
    Parameters:
    inputarray (cp.ndarray): The input waveform.
    
    Returns:
    cp.ndarray: The Fourier transform of the input waveform.
    """
    # Perform the Fourier transform
    outputarray = cpfft.fftshift(cpfft.fft(cpfft.fftshift(inputarray), norm="ortho"))
    outputarray_energy = cp.sum(cp.abs(outputarray)**2)
    outputarray = (normvalue / outputarray_energy) * outputarray
    fourierintensity = cp.square(cp.abs(outputarray))
    return fourierintensity

def fouriertoreal(inputarray):
    """
    Simulate the Fourier transform of a waveform.
    
    Parameters:
    inputarray (cp.ndarray): The input waveform.
    
    Returns:
    cp.ndarray: The Fourier transform of the input waveform.
    """
    # Perform the Fourier transform
    outputarray = cpfft.ifftshift(cpfft.fft(cpfft.ifftshift(inputarray), norm="ortho"))
    fourierintensity = cp.square (cp.abs(outputarray))
    return fourierintensity

def initguess_waveform(AWGwaveform, positions, time, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    AWG_fourierspace = cp.zeros(len(AWGwaveform))
    fourierpixels, time = positionstofourier(positions, time, globalvariables)
    expanded_fourierpixels, expanded_time = expand_position_array(time, fourierpixels, globalvariables)
    AWG_fourierspace[numpix_frame: -numpix_frame] = tocupy(expanded_fourierpixels)
    frequency_t0 = fourierpixels[0]
    frequency_tF = fourierpixels[-1]
    AWG_fourierspace[0:numpix_frame] = frequency_t0
    AWG_fourierspace[-numpix_frame:] = frequency_tF
    AWG_time = cp.linspace(0, 1*len(AWGwaveform) / numpix_real, len(AWGwaveform))
    
    AWG_fourierspace = AWG_fourierspace - numpix_real // 2

    AWGwaveform_out = 2*cp.pi*cp.cumsum(AWG_fourierspace) * (AWG_time[1] - AWG_time[0])
    

    return AWGwaveform_out

def cosinephaseresponse(AWGwaveform):
    return cp.cos(AWGwaveform )      # DO NOTHING because the waveform is precisely cosine

def exponentialphaseresponse(AWGwaveform):
    # return AWGwaveform + cp.sqrt(1-AWGwaveform**2) * 1j # Eulers identity  + Trig trick
    return cp.exp(AWGwaveform*1j)

 
def snapshot(cycle, globalvariables):
    '''Cycle here is non-zero padded section of length numpix_frame of the AWGwaveform.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    cycle = zeropadframe(cycle, globalvariables)
    return realtofourier(cycle)

def retrieveforces_idealconditions(AWGwaveform, positions, globalvariable):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    numframes = len(AWGwaveform) - numpix_frame + 1
    
    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = cpfft.fftshift(cpfft.fft(cpfft.fftshift(zeropadframe(AWGwaveform[0:numpix_frame], globalvariables)), norm="ortho"))
    calibrationshot_energy = cp.sum(cp.abs(calibrationshot)**2)
    rescalingfactor = tweezerenergy_max / cp.max(cp.square(cp.abs(calibrationshot))) 
    
    calibrationshot = realtofourier_norm(zeropadframe(AWGwaveform[0:numpix_frame], globalvariables),calibrationshot_energy) 
    tweezerprofile = removeleftside(calibrationshot)
    tweezerforce = tweezerprofile* rescalingfactor / pixelsize_fourier

    
    dummyindices1 = np.linspace(0,1,len(positions))
    expanded_position = np.linspace(0,1,numframes)
    interpolation_function = interp1d(dummyindices1, positions, kind='linear')
    expanded_position = interpolation_function(expanded_position)
    expanded_position_pixels = positionstofourier(expanded_position, 0, globalvariables)[0]


    # max_peak_location = np.interp(np.argmax(tweezerprofile), np.arange(len(tweezerprofile)), expanded_position)

    shifted_profiles = shift_tweezer_profile(tonumpy(tweezerforce), tonumpy(expanded_position_pixels))
    
    return calculateforces(tocupy(shifted_profiles))


def retrievepotentials_idealconditions(AWGwaveform, positions, globalvariable):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    numframes = len(AWGwaveform) - numpix_frame + 1
    
    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = cpfft.fftshift(cpfft.fft(cpfft.fftshift(zeropadframe(AWGwaveform[0:numpix_frame], globalvariables)), norm="ortho"))
    calibrationshot_energy = cp.sum(cp.abs(calibrationshot)**2)
    rescalingfactor = tweezerenergy_max / cp.max(cp.square(cp.abs(calibrationshot))) 
    
    calibrationshot = realtofourier_norm(zeropadframe(AWGwaveform[0:numpix_frame], globalvariables),calibrationshot_energy) 
    tweezerprofile = removeleftside(calibrationshot)
    tweezerforce = tweezerprofile* rescalingfactor / pixelsize_fourier

    
    dummyindices1 = np.linspace(0,1,len(positions))
    expanded_position = np.linspace(0,1,numframes)
    interpolation_function = interp1d(dummyindices1, positions, kind='linear')
    expanded_position = interpolation_function(expanded_position)
    expanded_position_pixels = positionstofourier(expanded_position, 0, globalvariables)[0]


    # max_peak_location = np.interp(np.argmax(tweezerprofile), np.arange(len(tweezerprofile)), expanded_position)

    shifted_profiles = shift_tweezer_profile(tonumpy(tweezerforce), tonumpy(expanded_position_pixels))
    
    return shifted_profiles


def retrieveforces(AWGwaveform, globalvariables, timeperframe = 1, filterOn=True):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    
    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = cpfft.fftshift(cpfft.fft(cpfft.fftshift(zeropadframe(AWGwaveform[0:numpix_frame], globalvariables)), norm="ortho"))
    calibrationshot_energy = cp.sum(cp.abs(calibrationshot)**2)
    rescalingfactor = tweezerenergy_max / cp.max(cp.square(cp.abs(calibrationshot)))   

    num_snapshots = (len(AWGwaveform) - numpix_frame) // timeperframe + 1
    
    # Create a view of the input array with overlapping windows
    strides = (AWGwaveform.strides[0] * timeperframe, AWGwaveform.strides[0])
    shape = (num_snapshots, numpix_frame)
    snapshots = as_strided(AWGwaveform, shape=shape, strides=strides)    
    snapshots = cp.array([realtofourier_norm(zeropadframe(snap, globalvariables),calibrationshot_energy) for snap in snapshots]).astype(float)

    snapshots = calculateforces(snapshots)
    if timeperframe > 1:
        interpolated_snapshots = cp.zeros((num_snapshots + (num_snapshots - 1) * (timeperframe - 1), numpix_real)).astype(float)
        interpolated_snapshots[::timeperframe] = snapshots
        
        for i in range(1, timeperframe):
            interpolated_snapshots[i::timeperframe] = (snapshots[:-1] * (timeperframe - i) + snapshots[1:] * i) / timeperframe
        
        if filterOn:
            interpolated_snapshots = cp.array([removeleftside(snap) for snap in interpolated_snapshots]).astype(float)
        return interpolated_snapshots* rescalingfactor/ pixelsize_fourier # Add pixelsize_fourier for derivative rescaling
    
    if filterOn: 
        snapshots = cp.array([removeleftside(snap) for snap in snapshots]).astype(float)

    return snapshots* rescalingfactor/ pixelsize_fourier

def retrievepotentials(AWGwaveform, globalvariables, timeperframe = 1, filterOn=True):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    tweezerenergy_max = hbar * tweezerdepth
    calibrationshot = cpfft.fftshift(cpfft.fft(cpfft.fftshift(zeropadframe(AWGwaveform[0:numpix_frame], globalvariables)), norm="ortho"))
    calibrationshot_energy = cp.sum(cp.abs(calibrationshot)**2)
    rescalingfactor = tweezerenergy_max / cp.max(cp.square(cp.abs(calibrationshot)))   

    num_snapshots = (len(AWGwaveform) - numpix_frame) // timeperframe + 1
    
    # Create a view of the input array with overlapping windows
    strides = (AWGwaveform.strides[0] * timeperframe, AWGwaveform.strides[0])
    shape = (num_snapshots, numpix_frame)
    snapshots = as_strided(AWGwaveform, shape=shape, strides=strides)    
    snapshots = cp.array([realtofourier_norm(zeropadframe(snap, globalvariables),calibrationshot_energy) for snap in snapshots]).astype(float)


    if timeperframe > 1:
        interpolated_snapshots = cp.zeros((num_snapshots + (num_snapshots - 1) * (timeperframe - 1), numpix_real), dtype=AWGwaveform.dtype)
        interpolated_snapshots[::timeperframe] = snapshots
        
        for i in range(1, timeperframe):
            interpolated_snapshots[i::timeperframe] = (snapshots[:-1] * (timeperframe - i) + snapshots[1:] * i) / timeperframe

        if filterOn:
            interpolated_snapshots = cp.array([removeleftside(snap) for snap in interpolated_snapshots])
        return interpolated_snapshots* rescalingfactor
    
    if filterOn: 
        snapshots = cp.array([removeleftside(snap) for snap in snapshots])

    return snapshots* rescalingfactor

def calculateforces(potentials):
    forces = cp.zeros_like(potentials)
    # forces = [gradient(potential) * potential for potential in potentials]
    forces = [gradient(potential) for potential in potentials]

    return cp.array(forces)


def initdistribution_MaxwellBoltzmann(num_particles, temperature, positionstd, atommass, globalvariables):
    """
    Generates a Maxwell-Boltzmann distribution of particles' positions and velocities. In units of fourier pixels / timestep

    Parameters:
    - num_particles (int): Number of particles.
    - mass (float): Mass of each particle.
    - temperature (float): Temperature in Kelvin.
    - kb (float, optional): Boltzmann constant. Default is 1.38e-23 J/K.

    Returns:
    - positions (np.ndarray): Array of positions of particles.
    - velocities (np.ndarray): Array of velocities of particles.
    """
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
       
    x0, time = positionstofourier(startlocation,0, globalvariables)
    
    # Standard deviation for velocity from Maxwell-Boltzmann distribution
    kb = 1.38*10**(-23)
    energy = 1/2 * kb * temperature
    std_velocity = np.sqrt(2 * energy / atommass)
    std_velocity = (std_velocity) # velocity in terms of pixels / timestep
    std_position = positionstd / pixelsize_fourier # pixel position
    # Generating velocities
    velocities = np.random.normal(0, std_velocity, (num_particles,1)) # in units of pixels/s
    velocities[velocities > 2* np.std(velocities)] *= 0.5
    # Generating positions (assuming normal distribution centered at 0 with some spread) # in units of pixels
    positions = np.random.normal(0, std_position, (num_particles,1)) +x0
    
    return positions, velocities

# update MB distrib initiatilization
 # def initdistribution_MaxwellBoltzmann(num_particles, temperature, positionstd, atommass, globalvariables):
    """
    Generates a Maxwell-Boltzmann distribution of particles' positions and velocities. In units of fourier pixels / timestep

    Parameters:
    - num_particles (int): Number of particles.
    - mass (float): Mass of each particle.
    - temperature (float): Temperature in Kelvin.
    - kb (float, optional): Boltzmann constant. Default is 1.38e-23 J/K.

    Returns:
    - positions (np.ndarray): Array of positions of particles.
    - velocities (np.ndarray): Array of velocities of particles.
    """
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
       
    x0, time = positionstofourier(startlocation,0, globalvariables)
    
    # Standard deviation for velocity from Maxwell-Boltzmann distribution
    kb = 1.38*10**(-23)
    energy = 1/2 * kb * temperature
    std_velocity = np.sqrt(2 * energy / atommass)
    std_velocity = (std_velocity) # velocity in terms of pixels / timestep
    positionstd = np.sqrt(2 * energy / atommass )/ trapfreq  # pixel position
    
    std_position = positionstd / pixelsize_fourier # pixel position
    # Generating velocities
    velocities = np.random.normal(0, std_velocity, (num_particles,1)) # in units of pixels/s
    velocities[velocities > 2* np.std(velocities)] *= 0.5
    # Generating positions (assuming normal distribution centered at 0 with some spread) # in units of pixels
    positions = np.random.normal(0, std_position, (num_particles,1)) +x0
    
    return positions, velocities


def montecarlo(forces, globalvariables, initialdistribution, atommass):
    ''' Monte Carlo simulation of a distribution of particles in a potential landscape.'''
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
        # Assuming that force is in units of kg*m/s^2, and mass is in kg, then acceleration is in m/s^2
    
    # Need to convert force from units of meters to units of pixels to make things computationally efficient
    
    ddx = forces / atommass # m/s^2
    x0 = tocupy(initialdistribution[0]) # Right now its in pixels
    dx0 = tocupy(initialdistribution[1]) # Right now its in m/s
    x_t1 = x0
    dx_t1 = dx0
    x_t2 = x0
    dx_t2 = dx0
    for iteration in range(len(ddx)):
        ddx_frame = ddx[iteration]
        ddx_t1 = tocupy(np.interp(tonumpy(x_t1), np.arange(numpix_real), tonumpy(ddx_frame)))
        dx_t2 = dx_t1 + ddx_t1 * timestep
        x_t2 = x_t1 + dx_t1 * timestep / pixelsize_fourier
        
        dx_t1 = dx_t2
        x_t1 = x_t2

    
    return x_t2, dx_t2, ddx_t1

# quality of life
def gradient(arr):
    # Calculate the spacing between points    
    return tocupy(np.gradient(tonumpy(arr)))

def tonumpy(array):
    '''Checks if the input array is a CuPy array and converts it to a NumPy array if necessary.'''
    if isinstance(array, np.ndarray):
        return array  # Convert CuPy array to NumPy array
    elif isinstance(array, cp.ndarray):
        return cp.asnumpy(array)  # Already a NumPy array, return as is
    else:
        raise TypeError("Input is neither a NumPy nor a CuPy array")

def tocupy(array):
    '''Checks if the input array is a NumPy array and converts it to a CuPy array if necessary.'''
    if isinstance(array, np.ndarray):
        return cp.asarray(array)  # Convert NumPy array to CuPy array
    elif isinstance(array, cp.ndarray):
        return array  # Already a CuPy array, return as is
    else:
        raise TypeError("Input is neither a NumPy nor a CuPy array")

def zeropadframe(frame, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    paddedframe = cp.zeros(numpix_real, dtype=complex)
    paddedframe[(numpix_real - numpix_frame) // 2: (numpix_real + numpix_frame) // 2] = frame
    return paddedframe

def micronstoMHz(startlocation_microns, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    frequencyposition_meters = startlocation_microns
    frequencyposition_pixel = positionstofourier(frequencyposition_meters,0, globalvariables)[0]
    acousticwavelength_pixel = (numpix_real)/(frequencyposition_pixel - numpix_real // 2)
    acousticwavelength_meters = acousticwavelength_pixel * pixelsize_real
    frequencyposition_MHz = soundvelocity / acousticwavelength_meters
    return frequencyposition_MHz    

def gaussian(x, amplitude, mean, sigma):
    """
    Gaussian function.
    
    Parameters:
    x (cp.ndarray): Input array.
    amplitude (float): Amplitude of the Gaussian.
    mean (float): Mean of the Gaussian.
    sigma (float): Standard deviation of the Gaussian.
    
    Returns:
    cp.ndarray: Values of the Gaussian function.
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def get_gaussianwidth_1d(array):
    """
    Fit a Gaussian to a 1D array and return its width (standard deviation).
    
    Parameters:
    array (cp.ndarray): Input 1D array.
    
    Returns:
    float: Width (standard deviation) of the fitted Gaussian.
    """
    # Ensure input is a CuPy array
    array = np.sqrt(tonumpy(array)) # Because we are using intensity, and the actual E-field is Gaussian
    
    # Generate x values
    x = np.arange(array.size)
    # Initial guess for the parameters
    initial_guess = [array.max(), x.size / 2, x.size / 4]
    
    # Fit the Gaussian
    try:
        popt, _ = curve_fit(gaussian, x, array, p0=initial_guess)
    except RuntimeError:
        return np.nan  # Return NaN if the fit fails
    
    # Extract the fitted sigma (standard deviation)
    fitted_sigma = popt[2]
    
    return fitted_sigma 

def get_gaussiancenter_1d(array):
    """
    Fit a Gaussian to a 1D array and return its width (standard deviation).
    
    Parameters:
    array (cp.ndarray): Input 1D array.
    
    Returns:
    float: Width (standard deviation) of the fitted Gaussian.
    """
    # Ensure input is a CuPy array
    array = np.sqrt(tonumpy(array)) # Because we are using intensity, and the actual E-field is Gaussian
    
    # Generate x values
    x = np.arange(array.size)
    # Initial guess for the parameters
    initial_guess = [array.max(), x.size / 2, x.size / 4]
    
    # Fit the Gaussian
    try:
        popt, _ = curve_fit(gaussian, x, array, p0=initial_guess)
    except RuntimeError:
        return np.nan  # Return NaN if the fit fails
    
    # Extract the fitted sigma (standard deviation)
    fitted_sigma = popt[1]
    
    return fitted_sigma 



def find_max_peak_location(tweezerprofile):
    """
    Find the location of the maximal peak in the tweezer profile with high accuracy.
    
    Parameters:
    tweezerprofile (np.ndarray): The 1D array representing the tweezer profile.
    
    Returns:
    float: The location of the maximal peak.
    """
    # Find the indices of peaks
    peaks, _ = find_peaks(tweezerprofile)
    
    # Find the peak with the maximum value
    max_peak_idx = peaks[np.argmax(tweezerprofile[peaks])]
    
    # Refine the peak location using quadratic interpolation for higher accuracy
    if max_peak_idx == 0 or max_peak_idx == len(tweezerprofile) - 1:
        return max_peak_idx  # Cannot refine if the peak is at the boundary
    
    x0 = max_peak_idx - 1
    x1 = max_peak_idx
    x2 = max_peak_idx + 1
    
    y0 = tweezerprofile[x0]
    y1 = tweezerprofile[x1]
    y2 = tweezerprofile[x2]
    
    # Quadratic interpolation to find the peak's x-coordinate
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return max_peak_idx  # Avoid division by zero
    
    refined_peak_idx = x1 + 0.5 * (y0 - y2) / denom
    
    return refined_peak_idx

def shift_tweezer_profile(tweezerprofile, positions):
    """
    Create an array of 1D arrays containing the tweezer profile centered at each new position.
    
    Parameters:
    tweezerprofile (np.ndarray): The 1D array representing the tweezer profile.
    positions (np.ndarray): The 1D array of positions to center the tweezer profile.
    
    Returns:
    np.ndarray: An array of 1D arrays with the shifted tweezer profiles.
    """
    # Find the location of the maximal peak
    peak_location = find_max_peak_location(tweezerprofile)
    
    # Length of the tweezer profile
    profile_length = len(tweezerprofile)
    
    # Create a 2D array to store the shifted profiles
    shifted_profiles = np.empty((len(positions), profile_length))
    
    # Interpolation function for the original tweezer profile
    x_original = np.arange(profile_length)
    interp_func = interp1d(x_original, tweezerprofile, kind='linear', fill_value="extrapolate")
    
    for i, pos in enumerate(positions):
        # Calculate the shift amount
        shift =  peak_location - pos
        
        # Create a new x-axis for the shifted profile
        x_shifted = x_original + shift
        
        # Interpolate to get the shifted profile
        shifted_profile = interp_func(x_shifted)
        
        # Store the shifted profile
        shifted_profiles[i] = shifted_profile
    
    return shifted_profiles

def array_1dto2d(arr, N):
    """
    Stacks a 1D array into a 2D array by repeating it N times along the rows.

    Parameters:
    arr (np.ndarray): The 1D array to be stacked.
    N (int): The number of times to repeat the 1D array along the rows.

    Returns:
    np.ndarray: The resulting 2D array.
    """
    # Ensure the input is a 1D array
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")

    # Use np.tile to repeat the array N times along a new axis
    result = cp.tile(arr, (N, 1))
    
    return result

def gaussian_2d(coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
    """
    2D Gaussian function.
    
    Parameters:
    coords: tuple of 2D coordinate arrays (x, y)
    x0, y0: center of the Gaussian
    sigma_x, sigma_y: standard deviations in x and y directions
    amplitude: peak amplitude
    offset: offset
    
    Returns:
    2D Gaussian value at (x, y)
    """
    x, y = coords
    return offset + amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))

def fit_gaussian_2d(intensity_array):
    """
    Fit a 2D Gaussian to a 2D numpy intensity array and return the horizontal and vertical widths.
    
    Parameters:
    intensity_array (numpy.ndarray): 2D intensity array
    
    Returns:
    tuple: (sigma_x, sigma_y) where sigma_x and sigma_y are the standard deviations (widths) in x and y directions
    """
    # Create coordinate arrays
    x = np.linspace(0, intensity_array.shape[1] - 1, intensity_array.shape[1])
    y = np.linspace(0, intensity_array.shape[0] - 1, intensity_array.shape[0])
    x, y = np.meshgrid(x, y)
    
    # Initial guess for the parameters based on maximum intensity
    amplitude = intensity_array.max()
    offset = intensity_array.min()
    y0, x0 = np.unravel_index(np.argmax(intensity_array), intensity_array.shape)
    sigma_x = intensity_array.shape[1] / 10  # Starting guess
    sigma_y = intensity_array.shape[0] / 10  # Starting guess
    
    initial_guess = (x0, y0, sigma_x, sigma_y, amplitude, offset)
    
    # Flatten the arrays for fitting
    x_data = x.ravel()
    y_data = y.ravel()
    intensity_data = intensity_array.ravel()
    
    # Fit the Gaussian
    popt, _ = curve_fit(gaussian_2d, (x_data, y_data), intensity_data, p0=initial_guess)
    
    # Extract the fitted parameters
    _, _, sigma_x_fit, sigma_y_fit, _, _ = popt
    
    return cp.abs(sigma_x_fit), cp.abs(sigma_y_fit)



# Visualization

def removeleftside(arr):
    # arr[0:len(arr)//2] = 0
    return arr

def zoomin(array_1d, N):
    """
    Crops a 1D numpy array to the relevant parts where the data is high enough in intensity.
    The threshold is defined as N times the mean of the array.
    
    Parameters:
    array_1d (np.ndarray): A 1D numpy array.
    N (float): A multiplier for the mean of the array to determine the threshold.
    
    Returns:
    np.ndarray: The cropped array containing only the relevant high-intensity parts.
    """
    # Calculate the mean of the array
    mean_value = cp.mean(array_1d)
    
    # Define the threshold as N times the mean
    threshold = N * mean_value
    
    # Find indices where the data exceeds the threshold
    high_intensity_indices = cp.where(array_1d > threshold)[0]
    
    if len(high_intensity_indices) == 0:
        # No values exceed the threshold, return an empty array
        return cp.array([])
    
    # Get the start and end indices of the relevant parts
    start_index = high_intensity_indices[0]
    end_index = high_intensity_indices[-1] + 1
    
    # Crop the array to the relevant parts
    cropped_array = array_1d[start_index:end_index]
    
    return cropped_array

def zoomin_cropped(array_1d, R):
    """
    Crops a 1D numpy array to a range centered around the maximum value with a specified width R.
    
    Parameters:
    array_1d (np.ndarray): A 1D numpy array.
    R (int): The width of the range to crop around the maximum value.
    
    Returns:
    np.ndarray: The cropped array centered around the maximum value with width R.
    """
    if R <= 0:
        raise ValueError("R must be a positive integer.")
    
    # Find the index of the maximum value in the array
    max_index = cp.argmax(array_1d)
    
    # Determine the start and end indices for the cropped range
    start_index = max(0, max_index - R // 2)
    end_index = min(len(array_1d), start_index + R)
    
    # Adjust the start index if the end index is out of bounds
    if end_index - start_index < R:
        start_index = max(0, end_index - R)
    
    # Crop the array to the specified range
    cropped_array = array_1d[start_index:end_index]
    
    return cropped_array

def plot_arrays(arrays, titles=None, subplot_figsize=(5, 5)):
    num_arrays = len(arrays)
    
    # Create a figure with subplots, one for each array
    fig, axes = plt.subplots(1, num_arrays, figsize=(subplot_figsize[0] * num_arrays, subplot_figsize[1]))
    
    # If there's only one array, axes will not be a list
    if num_arrays == 1:
        axes = [axes]
    
    # Use provided titles or default to 'Array {i+1}'
    if titles is None:
        titles = [f'Array {i+1}' for i in range(num_arrays)]
    
    for i, array in enumerate(arrays):
        axes[i].plot(array)
        axes[i].set_title(titles[i])
    
    plt.tight_layout()
    plt.show()


## Plot potentials (or forces) at multiple snapshots in time
def plot_snapshots(frames, numsnapshots, globalvariables, figuresize = (10,6),filterOn=True):
    """
    Plot the frames of an input array at multiple points in time. Final graph has all the plots overlaid.
    
    Parameters:
    frames (cp.ndarray): Array of frames, each of which contains a 1D array representing a snapshot.
    numsnapshots (int): Number of snapshots to plot, evenly distributed throughout the frames array.
    globalvariables (dict): A dictionary of global variables.
    filterOn (bool, optional): Whether to apply an xlim filter to the snapshots. Default is True.
    """
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    # Convert cupy array to numpy array for compatibility with matplotlib
    frames = tonumpy(frames)
    
    # Calculate the indices for the snapshots to plot
    indices = np.linspace(0, len(frames) - 1, numsnapshots, dtype=int)
    
    plt.figure(figsize=figuresize)
    
    for idx in indices:
        snapshot = frames[idx]
        
        # Plot each snapshot
        plt.plot(snapshot, label=f'Time {idx * timestep*10**6} us')
    
    if filterOn:
        # Apply xlim filter based on global variables
        xlim_min = positionstofourier(startlocation, 0, globalvariables)[0] - 50
        xlim_max = positionstofourier(endlocation, 0, globalvariables)[0] + 50
        plt.xlim(xlim_min, xlim_max)
    
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    plt.title('Snapshots Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.show()



def analyze_survivalprobability(xout, finalposition, gaussianwidth, globalvariables):
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
    
    xout = tonumpy(xout)
    
    # Calculate the lower and upper bounds
    lower_bound = finalposition - gaussianwidth
    upper_bound = finalposition + gaussianwidth
    # Count the number of values within the bounds
    count_within_bounds = np.sum((xout >= lower_bound) & (xout <= upper_bound))
    # Calculate the percentage
    percentage_within_bounds = count_within_bounds / len(xout) * 100
    
    return percentage_within_bounds

def analyze_fixeddistance_nonoptimized(movementtimes, initialtemperatures, responsetype="Cosine",calctype="Ideal", guesstype = "Linear", timeperframe=1, globalvariables=globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    # Calculate the number of movement times and initial temperatures
    num_movementtimes = len(movementtimes)
    num_initialtemperatures = len(initialtemperatures)
    
    # Initialize arrays to store the results
    results = np.empty((num_movementtimes, num_initialtemperatures), dtype=object)
    
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
                    
                shotlast = realtofourier(zeropadframe(AWGphase[-numpix_frame:], globalvariables))
                gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin(removeleftside(shotlast), 2)))
                endtweezerlocation = get_gaussiancenter_1d(removeleftside(shotlast))

                forces = retrieveforces(AWGphase, globalvariables, timeperframe, True)
                print(max(forces[len(forces)//2]))
                
                initial_distributions = initdistribution_MaxwellBoltzmann(num_particles, initialtemperatures[j], 1e-8, atommass, globalvariables)
                xout, vout, accel = montecarlo(forces, globalvariables, initial_distributions, atommass)
                print(endtweezerlocation)

                survivalprobability = analyze_survivalprobability(xout, endtweezerlocation, gaussianwidth, globalvariables)
                # Store the result in the results array
                results[i, j] = [np.array(survivalprobability),tonumpy(xout),tonumpy(vout)]
                
            elif calctype == "Ideal":
                if guesstype == "Linear":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_linearramp(globalvariables)
                elif guesstype == "MinJerk":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_minimizejerk(globalvariables)
                elif guesstype == "SinSq":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_sinsqramp_general(globalvariables)
                
                if responsetype == "Cosine":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = cosinephaseresponse(AWGinput)                
                elif responsetype =="Exponential":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = exponentialphaseresponse(AWGinput)     
                    
                shotlast = realtofourier(zeropadframe(AWGphase[-numpix_frame:], globalvariables))
                gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin(removeleftside(shotlast), 2)))
                endtweezerlocation = get_gaussiancenter_1d(removeleftside(shotlast))
                
                forces = retrieveforces_idealconditions(AWGphase, optimized_position, True)
                print(max(forces[len(forces)//2]))
                
                initial_distributions = initdistribution_MaxwellBoltzmann(num_particles, initialtemperatures[j], 1e-8, atommass, globalvariables)
                xout, vout, accel = montecarlo(forces, globalvariables, initial_distributions, atommass)
                print(endtweezerlocation)

                survivalprobability = analyze_survivalprobability(xout, endtweezerlocation, gaussianwidth, globalvariables)
                # Store the result in the results array
                results[i, j] = [np.array(survivalprobability),tonumpy(xout),tonumpy(vout)]

    return results

def analyze_fixeddistance_optimized(movementtimes, initialtemperatures, responsetype="Cosine",calctype="Ideal", guesstype = "Linear", timeperframe=10, globalvariables=globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    # Calculate the number of movement times and initial temperatures
    num_movementtimes = len(movementtimes)
    num_initialtemperatures = len(initialtemperatures)
    
    # Initialize arrays to store the results
    results = np.empty((num_movementtimes, num_initialtemperatures), dtype=object)
    
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
                    
                shotlast = realtofourier(zeropadframe(AWGphase[-numpix_frame:], globalvariables))
                gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin(removeleftside(shotlast), 2)))
                endtweezerlocation = get_gaussiancenter_1d(removeleftside(shotlast))
                fittedwaveform, fittedlegendre = init_opt_waveformfitLegendre(AWGinput, 1000, globalvariables)
                optimizedwaveform, optimizedLegendre, AWGwave_template = opt_atomsurvival_Legendre(fittedwaveform, fittedlegendre, initialtemperatures[j], globalvariables)
                optimizedwaveform_expresponse = exponentialphaseresponse(optimizedwaveform)
                survivalprobability, xout, vout = get_atomsurvivalfromwaveform(optimizedwaveform_expresponse, initialtemperatures[j], timeperframe, globalvariables)
                
                # Store the result in the results array
                results[i, j] = [np.array(survivalprobability),tonumpy(xout),tonumpy(vout)]
                
            elif calctype == "Ideal":
                if guesstype == "Linear":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_linearramp(globalvariables)
                elif guesstype == "MinJerk":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_minimizejerk(globalvariables)
                elif guesstype == "SinSq":
                    optimized_position, optimized_velocity, optimized_acceleration, optimized_jerk, time = initpath_sinsqramp_general(globalvariables)
                
                if responsetype == "Cosine":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = cosinephaseresponse(AWGinput)                
                elif responsetype =="Exponential":
                    AWGinput = initguess_waveform(AWGwaveform, optimized_position, time, globalvariables)
                    AWGphase = exponentialphaseresponse(AWGinput)     
                    
                shotlast = realtofourier(zeropadframe(AWGphase[-numpix_frame:], globalvariables))
                gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin(removeleftside(shotlast), 2)))
                endtweezerlocation = get_gaussiancenter_1d(removeleftside(shotlast))
                fittedwaveform, fittedlegendre = init_opt_waveformfitLegendre(AWGinput, 1000, globalvariables)
                optimizedwaveform, optimizedLegendre, AWGwave_template = opt_atomsurvival_Legendre(fittedwaveform, fittedlegendre, initialtemperatures[j], globalvariables)
                optimizedwaveform_expresponse = exponentialphaseresponse(optimizedwaveform)
                survivalprobability, xout, vout = get_atomsurvivalfromwaveform(optimizedwaveform_expresponse, initialtemperatures[j], timeperframe, globalvariables)
                
                # Store the result in the results array
                results[i, j] = [np.array(survivalprobability),tonumpy(xout),tonumpy(vout)]
    return results



def plots_fixeddistance(movementtimes, initialtemperatures, analysisout):
    """
    Plot the results of the fixed distance analysis.
    
    Parameters:
    analysisout (np.ndarray): Array of results from the fixed distance analysis.
    """
    # Get the number of movement times and initial temperatures
    num_movementtimes, num_initialtemperatures = analysisout.shape
    kb = 1.38*10**(-23)

    # Create a figure and axis
    fig, axs = plt.subplots(1,2,figsize=(15, 6))

    # Iterate over each combination of movement time and initial temperature
    for j in range(num_initialtemperatures):
        # Initialize an empty list to store the survival rates for each movement time
        survival_rates = []
        temperatures = []
        # Iterate over each movement time
        for i in range(num_movementtimes):
            # Get the results for the current combination
            survivalrate, xout, vout = analysisout[i, j]
            vout = tonumpy(vout)
            temperature = np.mean(vout**2 * atommass / (3*kb))
            temperature_errors = np.std(vout**2 * atommass / (3*kb))
            # Append the survival rate to the list
            survival_rates.append(survivalrate)
            temperatures.append(temperature* 1e6)
        # Plot the survival rates for the current initial temperature
        axs[0].plot(movementtimes * 1e6, survival_rates, label=f"T0: {initialtemperatures[j] * 1e6:.2f} μK")
        
        # Plot the temperatures for the current initial temperature
        axs[1].plot(movementtimes * 1e6, temperatures, label=f"T0: {initialtemperatures[j] * 1e6:.2f} μK")
        axs[1].errorbar(movementtimes * 1e6, temperatures, yerr=temperature_errors, label=f"T0: {initialtemperatures[j] * 1e6:.2f} μK")

    # Set labels, title, and legend for survival rate plot
    axs[0].set_xlabel('Movement Time (μs)')
    axs[0].set_ylabel('Survival Probability')
    axs[0].set_title('Survival Probability vs Movement Time')
    axs[0].set_ylim(-5, 105)
    axs[0].legend()

    # Set labels, title, and legend for temperature plot
    axs[1].set_xlabel('Movement Time (μs)')
    axs[1].set_ylabel('Temperature (μK)')
    axs[1].set_title('Temperature vs Movement Time')
    axs[1].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plots_fixeddistance_multipleanalysis(movementtimes, initialtemperatures, analysisout, titles):
    """
    Plot the results of the fixed distance analysis.
    
    Parameters:
    analysisout (np.ndarray): Array of results from the fixed distance analysis.
    """
    # Get the number of movement times and initial temperatures
    num_analysis, num_movementtimes, num_initialtemperatures = analysisout.shape
    kb = 1.38*10**(-23)

    # Create a figure and axis
    fig, axs = plt.subplots(1,2,figsize=(15, 6))
    
    # Iterate over each combination of movement time and initial temperature
    for k in range(num_analysis):
        for j in range(num_initialtemperatures):
            # Initialize an empty list to store the survival rates for each movement time
            survival_rates = []
            temperatures = []
            temperaturerrors = []
            # Iterate over each movement time
            for i in range(num_movementtimes):
                # Get the results for the current combination
                survivalrate, xout, vout = analysisout[k][i, j]
                vout = tonumpy(vout)
                temperature = np.mean(vout**2 * atommass / (3*kb))
                temperature_error = np.std(vout**2 * atommass / (3*kb))
                # Append the survival rate to the list
                survival_rates.append(survivalrate)
                temperatures.append(temperature* 1e6)
                temperaturerrors.append(temperature_error*1e6)
            # Plot the survival rates for the current initial temperature
            axs[0].plot(movementtimes * 1e6, survival_rates, label=f"T0: {initialtemperatures[j] * 1e6:.2f} μK,  {titles[k]}")
            
            # Plot the temperatures for the current initial temperature
            # axs[1].plot(movementtimes * 1e6, temperatures, label=f"T0: {initialtemperatures[j] * 1e6:.2f} μK")
            axs[1].errorbar(movementtimes * 1e6, temperatures, yerr=temperaturerrors, label=f"T0: {initialtemperatures[j] * 1e6:.2f} μK,  {titles[k]}")

    # Set labels, title, and legend for survival rate plot
    axs[0].set_xlabel('Movement Time (μs)')
    axs[0].set_ylabel('Survival Probability')
    axs[0].set_title('Survival Probability vs Movement Time')
    axs[0].set_ylim(-5, 105)
    axs[0].legend()

    # Set labels, title, and legend for temperature plot
    axs[1].set_xlabel('Movement Time (μs)')
    axs[1].set_ylabel('Temperature (μK)')
    axs[1].set_title('Temperature vs Movement Time')
    axs[1].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Optimization
def constructamplitudes(amplitudes, optimizationspace):
    
    return 

def constructphases(amplitudes, optimizationspace):
    
    return 

def init_opt_waveformfitFourierVariant(AWGinitguess,freqres, ampres, phaseres, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

    AWGwaveform = cp.zeros(len(AWGinitguess))
    AWGwaveform[0:numpix_frame] = AWGinitguess[0:numpix_frame]
    AWGwaveform[-numpix_frame:] = AWGinitguess[-numpix_frame:]
    
    optimizationsection = AWGinitguess[numpix_frame:-numpix_frame]
    optimizationspace = cp.zeros(len(AWGinitguess) - 2*numpix_frame, dtype=complex)
    
    
    frequencies = cp.empty(freqres)
    amplitudes = cp.empty((freqres, ampres))
    phaseres = cp.empty((freqres, phaseres))
    
    constructedamplitudes = constructamplitudes(amplitudes, optimizationspace)
    constructedphases = constructphases(phaseres, optimizationspace)
    
    optimizedwaveform = constructFourierVariant(frequencies, constructedamplitudes, constructedphases, globalvariables )
    


    return



# Opt: Legendre


def fit_legendre_polynomial(x, y, degree):
    """
    Fits a Legendre polynomial to the given data.
    
    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.
    degree (int): The degree of the Legendre polynomial.
    
    Returns:
    Legendre: The fitted Legendre polynomial.
    """
    # Fit the Legendre polynomial
    coeffs = Legendre.fit(x, y, degree).convert().coef
    
    # Create a Legendre object with the coefficients
    legendre_poly = Legendre(coeffs)
    
    return legendre_poly

def init_opt_waveformfitLegendre(AWGinitguess,degree, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    
    AWGwaveform = cp.zeros(len(AWGinitguess))
    AWGwaveform[0:numpix_frame] = AWGinitguess[0:numpix_frame]
    AWGwaveform[-numpix_frame:] = AWGinitguess[-numpix_frame:]
    
    optimizationsection = AWGinitguess[numpix_frame:-numpix_frame]
    optimizationspace = cp.linspace(-1,1,len(optimizationsection))
    
    startfrequency = positionstofourier(startlocation, 0, globalvariables)[0] - numpix_real // 2
    endfrequency = positionstofourier(endlocation, 0, globalvariables)[0] - numpix_real // 2

    
    fitted_legendre = fit_legendre_polynomial(tonumpy(optimizationspace), tonumpy(optimizationsection), degree)
    fitted_section = fitted_legendre(optimizationspace)
    max_error = np.max(np.abs(optimizationsection - fitted_section))
    print("Maximum error:", max_error)
    
    AWGwaveform[numpix_frame:-numpix_frame] = fitted_section
    
    return tocupy(AWGwaveform), tocupy(fitted_legendre.coef)

def opt_atomsurvival_Legendre(fittedwaveform, fittedcoefficients, inittemperature, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    
    AWGwaveform = fittedwaveform.copy()
    AWGwave_template = tonumpy(fittedwaveform.copy())
    AWGwaveform_expresponse = exponentialphaseresponse(fittedwaveform)
    optimizationsection = AWGwaveform_expresponse[numpix_frame:-numpix_frame]
    optimizationspace = cp.linspace(-1,1,len(optimizationsection))
    
    initial_distributions = initdistribution_MaxwellBoltzmann(num_particles, inittemperature, 1e-8, atommass, globalvariables)
    shotlast = realtofourier(zeropadframe(AWGwaveform_expresponse[-numpix_frame:], globalvariables))
    gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin((shotlast), 2)))
    endtweezerlocation = get_gaussiancenter_1d((shotlast))

    def objective_survivalLegendre(params):
        legendre_poly = Legendre(params)
        reconstructed_waveform = legendre_poly(tonumpy(optimizationspace))
        AWGwave_template[numpix_frame:-numpix_frame] = reconstructed_waveform
        AWGwave_test = exponentialphaseresponse(tocupy(AWGwave_template))
        forces = retrieveforces(AWGwave_test, globalvariables, 10, True)
        xout, vout, accel = montecarlo(forces, globalvariables, initial_distributions, atommass)
        survivalprobability = analyze_survivalprobability(xout, endtweezerlocation, gaussianwidth, globalvariables)
        print(survivalprobability)
        return 1 - survivalprobability

    initial_guess = fittedcoefficients
    
    # result = minimize(objective_survivalLegendre, tonumpy(initial_guess), method='Nelder-Mead', tol=1e-2,
    #     options={'maxfev ':1, 'maxiter':0, 'xtol':1e-2,'ftol':1e-2})

    result = minimize(objective_survivalLegendre,tonumpy(initial_guess),method='Powell',options={'disp': True,'maxfev':100,'maxiter':1, 'xtol': 1e-2, 'ftol': 1e-2})
    

    optimized_coefficients= result.x

    fitted_legendrepoly = Legendre(optimized_coefficients)
    AWGwaveform[numpix_frame:-numpix_frame] = tocupy(fitted_legendrepoly(tonumpy(optimizationspace)))
    
    return AWGwaveform, optimized_coefficients, AWGwave_template

def get_atomsurvivalfromwaveform(AWGwaveform, initialtemperature, timeperframe, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    shotlast = realtofourier(zeropadframe(AWGwaveform[-numpix_frame:], globalvariables))
    gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin(removeleftside(shotlast), 2)))
    endtweezerlocation = get_gaussiancenter_1d(removeleftside(shotlast))

    forces = retrieveforces(AWGwaveform, globalvariables, timeperframe, True)
    
    initial_distributions = initdistribution_MaxwellBoltzmann(num_particles, initialtemperature, 1e-8, atommass, globalvariables)
    xout, vout, accel = montecarlo(forces, globalvariables, initial_distributions, atommass)

    survivalprobability = analyze_survivalprobability(xout, endtweezerlocation, gaussianwidth, globalvariables)
    return survivalprobability, xout, vout

def opt_forces_Legendre(fittedwaveform, desiredpositions, desiredacceleration, fittedcoefficients, globalvariables):
    aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
    
    AWGwaveform = fittedwaveform.copy()
    AWGwave_template = tonumpy(fittedwaveform.copy())
    AWGwaveform_expresponse = exponentialphaseresponse(fittedwaveform)
    optimizationsection = AWGwaveform_expresponse[numpix_frame:-numpix_frame]
    optimizationspace = cp.linspace(-1,1,len(optimizationsection))

    forces = cp.zeros(len(fittedwaveform) - numpix_frame + 1)
    targetaccelerations = cp.zeros(len(forces))
    targetpositions = cp.zeros(len(forces))
    targetaccelerations[:len(desiredacceleration)] = desiredacceleration
    targetaccelerations[-numpix_frame-1:] = desiredacceleration[-1]
    targetaccelerations = targetaccelerations / cp.max(targetaccelerations)
    
    targetpositions[:len(desiredacceleration)] = desiredpositions
    targetpositions[-numpix_frame-1:] = desiredpositions[-1]

    
    shotlast = realtofourier(zeropadframe(AWGwaveform_expresponse[-numpix_frame:], globalvariables))
    gaussianwidth = get_gaussianwidth_1d(tonumpy(zoomin((shotlast), 2)))
    
    def evaluateforces(forces, targetpositions, targetaccelerations, globalvariables):
        maximumforceatpositions = [cp.max(cp.abs(forces[i][int(targetpositions[i] - gaussianwidth): int(targetpositions[i] + gaussianwidth)])) for i in range(len(forces))]
        maximumaccelerationatpositions = cp.array(maximumforceatpositions)
        maximumaccelerationatpositions = maximumaccelerationatpositions / cp.max(maximumaccelerationatpositions)
        intensityweight = len(forces) / cp.sum(maximumaccelerationatpositions)
        uniformityweight = cp.sum(cp.abs(targetaccelerations) - cp.abs(maximumaccelerationatpositions))
        print(uniformityweight)
        return intensityweight * uniformityweight
    
    def objective_survivalLegendre(params):
        legendre_poly = Legendre(params)
        reconstructed_waveform = legendre_poly(tonumpy(optimizationspace))
        AWGwave_template[numpix_frame:-numpix_frame] = reconstructed_waveform
        AWGwave_test = exponentialphaseresponse(tocupy(AWGwave_template))
        print("waveform made")
        forces = retrieveforces(AWGwave_test, globalvariables, 10, True)
        print("forces made")
        metric = evaluateforces(forces, targetpositions, targetaccelerations, globalvariables) * 10**(-20)
        return metric


    initial_guess = fittedcoefficients
    
    # result = minimize(objective_survivalLegendre, tonumpy(initial_guess), method='Nelder-Mead', tol=1e-2,
    #     options={'maxfev ':1, 'maxiter':0, 'xtol':1e-2,'ftol':1e-2})

    result = minimize(objective_survivalLegendre,tonumpy(initial_guess),method='Nelder-Mead',options={'disp': True,'maxfev':100,'maxiter':1, 'xtol': 1e-2, 'ftol': 1e-2})
    

    optimized_coefficients= result.x

    fitted_legendrepoly = Legendre(optimized_coefficients)
    AWGwaveform[numpix_frame:-numpix_frame] = tocupy(fitted_legendrepoly(tonumpy(optimizationspace)))
    
    return AWGwaveform, optimized_coefficients, AWGwave_template











# Analysis: Compare the optimized and non-optimized versions

## Case 1: Fixed distance, get best atom survival for movement times. 


## Case 2: Fixed movement time, optimized distances.







### Multiple tweezer movement:




























# class EField:
#     def __init__(self,  globalvariables):
#         """
#         Initializes the field, representing the cross-section profile of a plane wave

#         Parameters
#         ----------
#         wavelength: wavelength of the plane wave
#         extent_x: length of the rectangular grid 
#         extent_y: height of the rectangular grid 
#         Nx: horizontal dimension of the grid 
#         Ny: vertical dimension of the grid 
#         intensity: intensity of the field
#         """
#         aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables

#         self.extent_x = numpix_frame * pixelsize_fourier
#         self.dx = pixelsize_fourier

#         self.x = cp.linspace(0, numpix_frame * pixelsize_fourier, numpix_frame) -numpix_frame * pixelsize_fourier / 2
#         self.xx = self.x

#         self.Nx = numpix_frame
#         self.E = cp.zeros(numpix_frame) * cp.exp(1j * 0)
#         self.λ = wavelength
#         self.z = 0

# def lens(flength, globalvariables):
#     aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
#     xx = cp.linspace(0, numpix_frame * pixelsize_fourier, numpix_frame) -numpix_frame * pixelsize_fourier / 2
#     fresnellens = cp.exp(-1j * cp.pi / (wavelength * flength) * (xx**2))
#     return fresnellens

# def propogate_bluestein(self, E, dz, x_interval, globalvariables, scalefactor = 1):
#     aodaperture, soundvelocity, cycletime, focallength, wavelength, numpix_frame, numpix_real, pixelsize_real, aperturesize_real, aperturesize_fourier, pixelsize_fourier, movementtime, timestep, startlocation, endlocation, num_particles, atommass, tweezerdepth, hbar, optimizationbasisfunctions, numcoefficients = globalvariables
#     self.z += dz
#     λ = wavelength


#     E = bluestein_fft(E * cp.exp(1j * 2*cp.pi/λ /(2*dz) *(self.xx**2)), 
#                     f0=x_interval[0] / (dz*λ), f1=x_interval[1] / (dz*λ), fs=1/self.dx, M=numpix_frame, axis=0)

#     dfx = 1/(self.Nx*self.dx)

#     fx_zfft = bluestein_fftfreq(x_interval[0]/ (dz*λ),x_interval[1]/ (dz*λ), self.Nx)
#     dfx_zfft = fx_zfft[1]-fx_zfft[0]


#     nn = (cp.linspace(0,(self.Nx-1),self.Nx)*dfx_zfft/dfx )
#     factor = (self.dx* cp.exp(cp.pi*1j * (nn)))


#     self.x = fx_zfft*(dz*λ)

#     self.xx = self.x

#     self.dx = self.x[1] - self.x[0]

#     self.extent_x = self.x[1] - self.x[0] + self.dx
#     E = E*factor * cp.exp(1j*cp.pi/(λ*dz) * (self.xx**2)  +   1j*2*cp.pi/λ * dz ) / (1j*dz*λ)
#     self.E = E

#     return E

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