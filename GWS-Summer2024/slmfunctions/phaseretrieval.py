from .dependencies import *
from .settings import *


# virtual_slm = virtual_slmsimulation(slmabberation, phase_slm)
# virtual_slmphase, algoerrors = derivephase(virtual_slm, phase_slm)
# error = np.sum(np.abs(std_int - virtual_slm))
# correction = (virtual_slmphase - phase_slm + np.pi) % (2*np.pi) - np.pi
# updatedinputphase = (phase_slm - correction + np.pi) % (2*np.pi) - np.pi
# updatedimg = virtual_slmsimulation(slmabberation, updatedinputphase)
# virtualslmphase2, algoerrors2 = derivephase(updatedimg, updatedinputphase)
# error2 = np.sum(np.abs(std_int - updatedimg))
# correction2 = (virtualslmphase2 - updatedinputphase + np.pi) % (2*np.pi) - np.pi
# updatedinputphase2 = (updatedinputphase - correction2 + np.pi) % (2*np.pi) - np.pi
# updatedimg2 = virtual_slmsimulation(slmabberation, updatedinputphase2)
# virtualslmphase3, algoerrors3 = derivephase(updatedimg2, updatedinputphase2)
# error3 = np.sum(np.abs(std_int - updatedimg2))
# correction3 = (virtualslmphase3 - updatedinputphase2 + np.pi) % (2*np.pi) - np.pi
# updatedinputphase3 = (updatedinputphase2 - correction3 + np.pi) % (2*np.pi) - np.pi
# updatedimg3 = virtual_slmsimulation(slmabberation, updatedinputphase3)


def gen_calibrationpseudodonut(ring_spacing, ring_width, x_offcenteramount, y_offcenteramount):
    # Calculate the center coordinates
    xcenter = int(np.shape(blank)[1] / 2) + x_offcenteramount
    ycenter = int(np.shape(blank)[0] / 2) + y_offcenteramount

    # Generate a single donut ring
    radius_outer = ring_spacing / 2
    radius_inner = radius_outer - ring_width

    # Create a filled circle at the center of the blank array
    y, x = np.ogrid[:blank.shape[0], :blank.shape[1]]
    outer_mask = (x - xcenter) ** 2 + (y - ycenter) ** 2 <= radius_outer ** 2
    inner_mask = (x - xcenter) ** 2 + (y - ycenter) ** 2 <= radius_inner ** 2

    # Create the donut shape by subtracting the inner circle from the outer circle
    donut = np.zeros_like(blank)
    donut[outer_mask] = 1
    donut[inner_mask] = 0

    return donut

def gen_slmphaseabberationoval(width, height, oval_intensity, noise_intensity, x_offcenteramount, y_offcenteramount):
    # Create a blank numpy array
    noisyoval = np.zeros(np.shape(blank))
    # Calculate the center coordinates
    xcenter = int(width / 2) + x_offcenteramount
    ycenter = int(height / 2) + y_offcenteramount

    # Generate a noisy oval shape
    for y in range(height):
        for x in range(width):
            # Calculate distance from center
            distance = np.sqrt((x - xcenter)**2 + (y - ycenter)**2)

            # Oval intensity decreases with distance from the center
            oval_value = oval_intensity * (1 - distance / (width / 2))

            # Add noise to the oval
            noise = np.random.normal(0, noise_intensity)
            
            # Combine oval and noise values
            noisyoval[y, x] = max(0, oval_value + noise)

    # Normalize values to be in the range [0, 1]
    noisyoval /= np.max(noisyoval)

    return noisyoval

def gen_slmphaseabberationshapes(width, height, num_shapes, noise_intensity):
    # Create a blank numpy array
    noisyscratch = np.zeros(np.shape(blank))

    # Generate a noisy scratched pattern with random shapes using OpenCV
    for _ in range(num_shapes):
        shape_type = np.random.choice(['ellipse', 'rectangle', 'polygon'])
        if shape_type == 'ellipse':
            center = (int(np.random.rand() * width), int(np.random.rand() * height))
            axes = (int(np.random.rand() * width / 2), int(np.random.rand() * height / 2))
            angle = np.random.rand() * 360
            color = 1  # White color for ellipse
            cv2.ellipse(noisyscratch, center, axes, angle, 0, 360, color, -1)
        elif shape_type == 'rectangle':
            pt1 = (int(np.random.rand() * width), int(np.random.rand() * height))
            pt2 = (pt1[0] + int(np.random.rand() * width / 2), pt1[1] + int(np.random.rand() * height / 2))
            color = 1  # White color for rectangle
            cv2.rectangle(noisyscratch, pt1, pt2, color, -1)
        elif shape_type == 'polygon':
            num_vertices = np.random.randint(3, 6)
            vertices = np.random.rand(num_vertices, 2) * [width, height]
            vertices = vertices.astype(np.int32)
            color = 1  # White color for polygon
            cv2.fillPoly(noisyscratch, [vertices], color)

        # Add noise to the shape
        noise = np.random.normal(0, noise_intensity)
        noisyscratch += noise

    # Normalize values to be in the range [0, 1]
    noisyscratch /= np.max(noisyscratch)

    return noisyscratch

def gen_calibrationrings(numrings, ring_spacing, ring_width, x_offcenteramount, y_offcenteramount):
    blank = np.empty((1920*precision,1200*precision),dtype=float)

    # Create a blank numpy array
    # Calculate the center coordinates
    xcenter = int(np.shape(blank)[1] / 2) + x_offcenteramount
    ycenter = int(np.shape(blank)[0] / 2) + y_offcenteramount

    # Generate rings
    for i in range(numrings):
        radius = i * ring_spacing
        ring = np.zeros_like(blank)

        # Create a filled circle at the center of the blank array
        y, x = np.ogrid[:blank.shape[0], :blank.shape[1]]
        mask = (x - xcenter) ** 2 + (y - ycenter) ** 2 <= radius ** 2

        # Create a ring by subtracting the inner circle from the outer circle
        ring[mask] = 1
        if i > 0:
            inner_radius = (i - 1) * ring_spacing + (ring_width / 2)
            inner_mask = (x - xcenter) ** 2 + (y - ycenter) ** 2 <= inner_radius ** 2
            ring[inner_mask] = 0

        # Add the ring to the simplegrating
        blank += ring

    return blank


def gen_calibrationvortex(x_offset, y_offset, p, l, beamwaist, wavelength, fourierlen):
    # x_offset = 200
    # y_offset = 200
    # p = 0;                  # Degree of LG mode
    # l = 1;                  # Order of LG mode
    # w0 = 40.0;               # Beam waist
    w0 = beamwaist
    # k = 2*np.pi/650.0e-9;   # Wavenumber of light
    k = 2*np.pi/(wavelength*e-9);   # Wavenumber of light

    zR = k*w0**2.0/2;       # Calculate the Rayleigh range

    # Setup the cartesian grid for the plot at plane z
    z = 0.0
    xsize, ysize = np.shape(blank)
    xx, yy = np.meshgrid(np.linspace(0, xsize-1, xsize)-xsize // 2 + x_offset, np.linspace(0, ysize-1, ysize)-ysize // 2 + y_offset)
    rho = np.sqrt(xx**2 + yy**2)


    # Calculate the cylindrical coordinates
    r = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    testfourier =  np.pi / (fourierlen) *((xx-xsize//2)**2 + (yy-ysize//2)**2)
    phi = (phi + testfourier + np.pi) %(2*np.pi) - np.pi

    U00 = 1.0/(1 + 1j*z/zR) * np.exp(-r**2.0/w0**2/(1 + 1j*z/zR))
    w = w0 * np.sqrt(1.0 + z**2/zR**2)
    R = np.sqrt(2.0)*r/w

    # Lpl from OT toolbox (Nieminen et al., 2004)
    Lpl = comb(p+l,p) * np.ones(np.shape(R));   # x = R(r, z).^2
    for m in range(1, p+1):
        Lpl = Lpl + (-1.0)**m/factorial(m) * comb(p+l,p-m) * R**(2.0*m)

    U = U00*R**l*Lpl*np.exp(1j*l*phi)*np.exp(-1j*(2*p + l + 1)*np.arctan(z/zR))

    phase = np.angle(U)
    intensity = abs(U)**2
    
    return phase, intensity, U

def create_spiral_phase_plate(turns):
    # Create a blank array
    phase = np.zeros(np.shape(blank))
    width = np.shape(blank)[1]
    height = np.shape(blank)[0]
    # Define the coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = width // 2, height // 2

    # Calculate the angle of each pixel
    theta = np.arctan2(y - center_y, x - center_x)

    # Calculate the spiral phase plate
    k = 2 * np.pi 
    phase = k * turns * theta
    # Apply the phase to the blank array
    phase = phase % (np.max(phase)*2 / turns)
    phase = (phase / np.max(phase)) * 2*np.pi - np.pi
    return phase

# blazed_grating_phase = create_blazed_diffraction_grating_phase(wavelength=650, groove_spacing=650/2)

def gen_concentriccircles(powerratio, w, r):
    array = np.zeros(np.shape(blank))
    width = np.shape(blank)[1]
    height = np.shape(blank)[0]
    centerwidth = int(width / 2)
    centerheight = int(height / 2)
    
    geometric_series = w * np.power(powerratio, range(0,100))
    distribution = np.cumsum(geometric_series) - w + r
    distribution = np.insert(distribution,0,0)
    
    for i in range(height):
        for j in range(width):
            distance = np.sqrt((i - centerheight)**2 + (j - centerwidth)**2)
            index = np.argmax(distribution > distance)
            if distribution[index] > distance:
                outer = distribution[index]
                inner = distribution[index-1]
            else:
                outer = 0
                inner = 0
            if outer - inner > 0.2:
                value = abs((distance - inner) / (outer-inner))
                array[i,j] = 1 - value
    
    array = array * 2 * np.pi - np.pi
    return array

def opticalvortex(xshift,yshift, topo_charge, focal, k, wavelength):
    phase = np.zeros(np.shape(blank))
    width = np.shape(blank)[1]
    height = np.shape(blank)[0]
    # Define the coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = width // 2+xshift, height // 2+yshift
    
    diffractiongrating = k*(x)
    phasespiral = topo_charge*np.arctan2(y-center_y,x-center_x)
    fourier = np.pi / (wavelength*focal) *((x-center_x)**2 + (y-center_y)**2)
    phase = (phasespiral  + fourier + np.pi) % (2*np.pi) - np.pi
    return phase


def create_smooth_random_topology(shape, scale):
    rows, cols = shape

    # Generate random values
    random_values = np.random.randn(rows, cols)

    # Apply smoothing using Gaussian filter
    smooth_random_topology = gaussian_filter(random_values, sigma=scale)

    # Normalize the values to fit within the desired range
    smooth_random_topology = (smooth_random_topology - np.min(smooth_random_topology)) / (
            np.max(smooth_random_topology) - np.min(smooth_random_topology))

    return smooth_random_topology
# vortexphase, vortexintensity, vortexU = gen_calibrationvortex(0,0,0,1,30,650)
# testing = ((vortexphase+np.pi)/np.max(vortexphase)*np.pi*2 + spiral_phase_plate/2+np.pi) % (np.pi*2) - np.pi

# joinedvortex = join_phase_ampl(joinedvortexangle,Beam_shape(1200,1920,255,0))
# # virtual_trial1 = np.angle(sfft.ifftshift(sfft.ifft2(vortexphase)))
# joinedvortex2 = sfft.fftshift(sfft.fft2(joinedvortex))
# joinedvortexangle = np.angle(joinedvortex2)
# joinedvortexint=np.square(np.abs(joinedvortex2))
# joinedvortexint = joinedvortexint / np.max(joinedvortexint)

def add_phasefourier(originalphase, fourierscaling):
    phase = np.zeros(np.shape(originalphase))
    width = np.shape(originalphase)[1]
    height = np.shape(originalphase)[0]
    # Define the coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = width // 2, height // 2

    fourier = np.pi / (fourierscaling) *((x-center_x)**2 + (y-center_y)**2)
    phase = (originalphase  + fourier + np.pi) % (2*np.pi) - np.pi
    return phase


def remove_phasefourier(originalphase, fourierscaling):
    phase = np.zeros(np.shape(originalphase))
    width = np.shape(originalphase)[1]
    height = np.shape(originalphase)[0]
    # Define the coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    center_x, center_y = width // 2, height // 2

    fourier = np.pi / (fourierscaling) *((x-center_x)**2 + (y-center_y)**2)
    phase = (originalphase  - fourier + np.pi) % (2*np.pi) - np.pi
    return phase

def add_phasediffractiongrating(originalphase, spacing):
    phase = np.zeros(np.shape(originalphase))
    width = np.shape(originalphase)[1]
    height = np.shape(originalphase)[0]
    # Define the coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    diffractiongrating = spacing*x
    phase = (originalphase  + diffractiongrating + np.pi) % (2*np.pi) - np.pi
    return phase

def create_smooth_random_topology(shape, scale):
    rows, cols = shape

    # Generate random values
    random_values = np.random.randn(rows, cols)

    # Apply smoothing using Gaussian filter
    smooth_random_topology = gaussian_filter(random_values, sigma=scale)

    # Normalize the values to fit within the desired range
    smooth_random_topology = (smooth_random_topology - np.min(smooth_random_topology)) / (
            np.max(smooth_random_topology) - np.min(smooth_random_topology))

    return smooth_random_topology