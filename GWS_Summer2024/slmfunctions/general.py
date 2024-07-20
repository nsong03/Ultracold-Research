from .dependencies import *
from .settings import *

def Ger_Sax_algo(InputImg, width, height, max_iter):
    TwoDImg = np.reshape(InputImg, (-1, width))

    pm_s = np.random.rand(height, width)
    pm_f = np.ones((height, width))
    am_s = np.sqrt(TwoDImg) / 2
    am_f = np.ones((height, width))

    signal_s = am_s*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(signal_s)))
        pm_f = np.angle(signal_f)
        signal_f = am_f*np.exp(pm_f * 1j)
        signal_s = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(signal_f)))
        pm_s = np.angle(signal_s)
        signal_s = am_s*np.exp(pm_s * 1j)

    phase_mask = np.uint8(((pm_f/(2*np.pi))*256)+128)
    ### flat_phase = phase_mask.flatten()

    return phase_mask

### SLM Direct Images
# Create an SLM grating with sinusoidal pixel values

def gen_tweezers(xrad, yrad, xspacing, yspacing, xmin, xmax, ymin,ymax):
    simplegrating = np.zeros(np.shape(blank))

    for i in range(xmin,xmax):
        for j in range(ymin,ymax):
            pixcoord = 0
            distj1 = j  % yspacing
            disti1 = i  % xspacing
            distj2 = (yspacing - j) % yspacing
            disti2 = (xspacing - i) % xspacing
            distj = min((distj1, distj2))
            disti = min((disti1, disti2))
            if (distj < xrad) and (disti < yrad):
                pixcoord = 255
            simplegrating[j,i] = pixcoord
    return simplegrating


def gaussian_distribution(x, radius):
    return np.exp(-0.5 * (x / radius)**2)

def gen_tweezers_precise(xrad, yrad, xspacing, yspacing, xstart, xend, ystart,yend):
    simplegrating = np.zeros(np.shape(blank))
    
    for i in range(xstart, xend):
        for j in range(ystart,yend):
            pixcoord = 0
            distj1 = (j - ystart)  % yspacing
            disti1 = (i - xstart) % xspacing
            distj2 = (yspacing - (j-ystart)) % yspacing
            disti2 = (xspacing - (i-xstart)) % xspacing
            distj = min((distj1, distj2))
            disti = min((disti1, disti2))
            if (distj < xrad) and (disti < yrad):
                pixcoord = 255
            simplegrating[j,i] = pixcoord
    return simplegrating

def gaussian_tweezers_precise(xrad, yrad, xspacing, yspacing, xstart, xend, ystart,yend, radius):
    tweezers = gen_tweezers_precise(xrad, yrad, xspacing, yspacing, xstart, xend, ystart,yend)
    gaussiantweezers = tweezers
    binary_camera_image = (tweezers > 2).astype(np.uint8)
    labeled_camera_image, num_labels_camera = label(binary_camera_image)
    centers_camera = np.array(center_of_mass(binary_camera_image, labeled_camera_image, range(1, num_labels_camera + 1)))
    ## Capture 99.7% of the distribution
    factor = 3
    for i in range(len(centers_camera)):
        for x in range(-radius, radius,1):
            for y in range(-radius, radius,1):
                
                radvalue = (x**2+y**2)**0.5
                if radvalue < radius:
                    gaussiantweezers[centers_camera[i][0].astype(np.int) + x, centers_camera[i][1].astype(np.int) + y] += 255*gaussian_distribution(radvalue * 3, radius)
    
    return gaussiantweezers

def gen_circlemask(radius):
    simplegrating = np.zeros(np.shape(blank))
    for i in range(1920 * precision):
        for j in range(1200 * precision):
            pixcoord = 0
            if (((1920 * precision)/2 - i)**2 + ((1200 * precision)/2 - j)**2)**0.5 < radius:
                pixcoord = 255
            simplegrating[i,j] = pixcoord
    return simplegrating


### Image writing!

def f_grating(pattern):
    return Ger_Sax_algo(pattern, numcols, numrows, max_iter)

def writeimg(array, name):
    data = Image.fromarray(array).convert('RGB')
    data.save(name+".bmp")
    return print(name+" saved succesfully.")

def print_1Dimgs(function, numframes, min_width_pix, stepsize_pix):
    width = min_width_pix
    for i in range(numframes):
        name = str(100+i)
        array = function(width)
        writeimg(array, name)
        width += stepsize_pix
    return print("Image writing successful.")

def print_2Dimgs(function, numframes, min_width_pix, min_length_pix, stepsize_w_pix, stepsize_l_pix):
    width = min_width_pix
    length = min_length_pix
    for i in range(numframes):
        name = str(100+i)
        array = function(width, length)
        writeimg(array, name)
        width += stepsize_w_pix
        length += stepsize_l_pix
    return print("Image writing successful.")

def print_f_1Dimgs(function, numframes, min_width_pix, stepsize_pix):
    width = min_width_pix
    for i in range(numframes):
        name = str(100+i)
        array = function(width)
        farray = f_grating(array)
        writeimg(farray, name)
        width += stepsize_pix
    return print("Image writing successful.")

def print_f_2Dimgs(function, numframes, min_width_pix, min_length_pix, stepsize_w_pix, stepsize_l_pix):
    width = min_width_pix
    length = min_length_pix
    for i in range(numframes):
        name = str(100+i)
        array = function(width, length)
        farray = f_grating(array)
        writeimg(farray, name)
        width += stepsize_w_pix
        length += stepsize_l_pix
    return print("Image writing successful.")


def slm_2D_simulator(function, maxiter, min_width_pix, min_length_pix):
    slmarray = function(min_width_pix, min_length_pix)
    slmimage = Ger_Sax_algo(slmarray, numcols, numrows, maxiter)
    fourierimage = np.abs(np.fft.fft2(slmimage))
    fourierimage = fourierimage/np.max(fourierimage)*255
    writeimg(fourierimage,"simulim")
    return print("Print successful")

### Optimization code
### Testing Ger-Sax algorithm

def epsilon(u_int, target_im):
    max = np.max(u_int[target_im!=0]) #Max value of the obtained intensity at the tweezers position
    min = np.min(u_int[target_im!=0]) #Min value of the obtained intensity at the tweezers position
    error = (max-min)/(max+min)
    
    # error = np.sum(np.abs(u_int-target_im))
    #print("Error :", error)
    return error

def epsilon_gaussian(u_int, target_im):
    max = np.max(u_int[target_im!=0]) #Max value of the obtained intensity at the tweezers position
    min = np.min(u_int[target_im!=0]) #Min value of the obtained intensity at the tweezers position
    error = (max-min)/(max+min)
    
    # error = np.sum(np.abs(u_int-target_im))
    #print("Error :", error)
    return error


def join_phase_ampl(phase,ampl):
    tmp=np.zeros((ampl.shape[0],ampl.shape[1]),dtype=complex)
    tmp = ampl*np.exp(phase*1j)
    return tmp

def Beam_shape(sizex,sizey,sigma,mu):
    x, y = np.meshgrid(np.linspace(-1,1,sizex), np.linspace(-1,1,sizey))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)

def norm(matrix):
    min=np.min(matrix);max=np.max(matrix)
    return((matrix-min)/(max-min))


def screwone(list):
    for number in list:
        if number == 1:
            number = number - 0.01
    return number

def weights(w,target_im,w_prev,std_int): # This weight function works only where the intensity == 1 (discrete tweezers)
    # targetmatch_indices = np.argwhere(target_im==1)
    # for i in targetmatch_indices:
    #     for x in range(-2,3):
    #         for y in range(-2,3):
    #             w[i[0]+x,i[1]+y] = np.sqrt((target_im[i[0]+x,i[1]+y] / std_int[i[0]+x,i[1]+y])) *w_prev[i[0]+x,i[1]+y]
    #             # w[target_im!=1] = np.sqrt(1 / (1-screwone(std_int[target_im!=1]))) * w_prev[target_im!=1]
    w[target_im==1] = np.sqrt((target_im[target_im==1] / std_int[target_im==1])) * w_prev[target_im==1]
    return (w)

def weights_gaussian(w,target_im,w_prev,std_int): # This weight function works only where the intensity == 1 (discrete tweezers)
    # targetmatch_indices = np.argwhere(target_im==1)
    # for i in targetmatch_indices:
    #     for x in range(-2,3):
    #         for y in range(-2,3):
    #             w[i[0]+x,i[1]+y] = np.sqrt((target_im[i[0]+x,i[1]+y] / std_int[i[0]+x,i[1]+y])) *w_prev[i[0]+x,i[1]+y]
    #             # w[target_im!=1] = np.sqrt(1 / (1-screwone(std_int[target_im!=1]))) * w_prev[target_im!=1]
    w[target_im!=0] = np.sqrt((target_im[target_im!=0] / std_int[target_im!=0])) * w_prev[target_im!=0]
    return (w)

def weightintensity_lukin(target, target_prev, std_int, target_im):
    target[target_im==1] = np.sqrt((np.mean(std_int[target_im==1]) / (std_int[target_im==1]+0.001))) * target_prev[target_im==1]
    return target


# def screwone(list):
#     for number in list:
#         if number == 1:
#             number = number - 0.001
#     return number

# def weights(w,target_im,w_prev,std_int): # This weight function works only where the intensity == 1 (discrete tweezers)
#     w[target_im==1] = np.sqrt((target_im[target_im==1] / std_int[target_im==1])) * w_prev[target_im==1]
#     w[target_im!=1] = np.sqrt(1 / (1-screwone(std_int[target_im!=1]))) * w_prev[target_im!=1]

#     return (w)




def discretize_phase(phase):
    phase=np.round((phase+np.pi)*255/(2*np.pi))
    return(phase)

def undiscretize_phase(phase):
    phase=phase/255*(2*np.pi)-np.pi
    return(phase)

def set_circlemask(inputmatrix, radius):
    image = inputmatrix
    image[np.sqrt((np.arange(image.shape[0])[:,None] - image.shape[0]//2)**2 + (np.arange(image.shape[1]) - image.shape[1]//2)**2) > radius] = 0
  
    return image

def fourier(intensity, phase):
    combintandphase = join_phase_ampl(phase, intensity)
    fft = sfft.fftshift(sfft.fft2(combintandphase))
    fft_phase = np.angle(fft)
    fft_int=np.square(np.abs(fft))
    fft_int = (fft_int) / np.max(fft_int)
    return fft_int, fft_phase

def optimizespacing(k, theta):
    #get single-axis spacings
    d = k / np.cos(radians(theta))
    delta_d = np.random.rand(*d.shape) * np.pi
    sum = 0
    for i in range(len(d)):
        sum += makegrating_sawtooth1d(d[i], delta_d[i], 1000)/len(d)
    sum = (sum + np.pi) % ( 2*np.pi) - np.pi
    initdeviation = std(sum)
    print(initdeviation)
    print("initdev")
    def patterncost(delta_d):
        sum = 0
        for i in range(len(d)):
            sum += makegrating_sawtooth1d(d[i], delta_d[i], 1000)/len(d)
        sum = (sum + np.pi) % ( 2*np.pi) - np.pi
        # stdeviation = std(sum)
        sum[abs(sum) < 0.55] = 0
        sum = abs(sum*10)**3
        print(np.sum(sum))
        return np.sum(sum)
    
    optimized_deltad = minimize(patterncost, delta_d, method='BFGS', options={'maxiter':5000, 'gtol':10e-10})
    optimized_deltak = optimized_deltad.x * np.cos(radians(theta))
    print(optimized_deltad.fun)
    return optimized_deltak


def phasematching2d_sawtooth(k, theta, numpoints,tweezers):
    #get single-axis spacings
    # delta_k = np.random.rand(*k.shape)*4
    numtweezcols = np.unique(np.nonzero(tweezers)[0])
    numtweezrows = np.unique(np.nonzero(tweezers)[1])
    kx = k * np.cos(radians(theta))
    ky = k * np.sin(radians(theta))
    print(len(numtweezrows))
    print(len(numtweezcols))
    offsetx = kx*(kx-1)*2*np.pi / 2 / (len(numtweezcols)-1)
    offsety = ky*(ky-1)*2*np.pi / 2 / (len(numtweezcols)-1)
    offsetmag = np.sqrt(offsetx*offsetx + offsety*offsety)
    offsetangle = np.arctan(offsety / offsetx)
    offsetonk = offsetmag * np.cos((offsetangle - radians(theta)))
    
    sum = 0
    for i in range(len(k)):
        sum += makegrating_sawtooth(k[i], offsetonk[i],theta[i], numpoints)/len(k)
    sum = (sum + np.pi) % ( 2*np.pi) -np.pi

    # optimized_deltad = minimize(patterncost, delta_k, method='BFGS', options={'maxiter':500, 'gtol':10e-8})
    
    # optimized_deltak = optimized_deltad.x
    # print(optimized_deltad.fun)
    return offsetonk, sum

def phasematching2d_cosine(k, theta, numpoints,tweezers):
    #get single-axis spacings
    # delta_k = np.random.rand(*k.shape)*4
    numtweezcols = np.unique(np.nonzero(tweezers)[0])
    numtweezrows = np.unique(np.nonzero(tweezers)[1])
    kx = k * np.cos(radians(theta))
    ky = k * np.sin(radians(theta))
    print(len(numtweezrows))
    print(len(numtweezcols))
    offsetx = kx*(kx-1)*2*np.pi / 2 / (len(numtweezcols)-1) ## UPDATE: This might still not be correct - check if kx is actually num tweezers on width direction
    offsety = ky*(ky-1)*2*np.pi / 2 / (len(numtweezrows)-1)
    offsetmag = np.sqrt(offsetx*offsetx + offsety*offsety)
    offsetangle = np.arctan(offsety / offsetx)
    offsetonk = offsetmag * np.cos((offsetangle - radians(theta)))
    
    sum = 0
    for i in range(len(k)):
        sum += makegrating_cosine(k[i], offsetonk[i],theta[i], numpoints)/len(k)
    sum = (sum + np.pi) % ( 2*np.pi) -np.pi

    # optimized_deltad = minimize(patterncost, delta_k, method='BFGS', options={'maxiter':500, 'gtol':10e-8})
    
    # optimized_deltak = optimized_deltad.x
    # print(optimized_deltad.fun)
    return offsetonk, sum

def makegrating_cosine(k, offset, angle, num_points):
     # Set the parameters
    k = k * 1.5
    x = np.linspace(0, 1, int(num_points*1.5))
    y = np.linspace(0, 1, int(num_points*1.5))
    # Create a meshgrid
    x_mesh, y_mesh = np.meshgrid(x, y)
    # Set the sawtooth pattern parameters
    # Generate the sawtooth pattern along the x direction
    sawtooth_pattern_x = np.cos(2 * np.pi * k * x_mesh + offset)
    sawtooth_pattern = rotate(sawtooth_pattern_x, angle, reshape = False)
    sawtooth_pattern = sawtooth_pattern[int((num_points*0.5) // 2): int((num_points*2.5) //2), int((num_points*0.5) // 2): int((num_points*2.5) //2)]
    sawtooth_finalphase = sawtooth_pattern
    return sawtooth_finalphase * np.pi

def initguess_cosines(tweezers, num_points):
    pixel_coordinates = np.argwhere(tweezers == 1)
    center = np.array(tweezers.shape) // 2
    vectors = pixel_coordinates - center
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    angles = -degrees(angles)
    distances = np.linalg.norm(vectors, axis=1)
    k = distances / num_points * num_points
    offset = 0
    netphase = np.zeros(np.shape(tweezers))
    for i in range(len(distances)):
        # offset = k[i]*(k[i]-1)/(2*len(distances))
        netphase += makegrating_cosine(k[i],offset,angles[i],num_points) / len(distances)
    
    netphase = (netphase + np.pi) % (2*np.pi) - np.pi
    return k, angles, netphase


def randguess_sawtooth(tweezers, num_points):
    pixel_coordinates = np.argwhere(tweezers == 1)
    center = np.array(tweezers.shape) // 2
    vectors = pixel_coordinates - center
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    angles = -degrees(angles)
    distances = np.linalg.norm(vectors, axis=1)
    k = distances / num_points * num_points
    offset =  np.random.rand(*k.shape) * np.pi * 100
    netphase = np.zeros(np.shape(tweezers))
    for i in range(len(distances)):
        # offset = k[i]*(k[i]-1)/(2*len(distances))
        netphase += makegrating_sawtooth(k[i],offset[i],angles[i],num_points) / len(distances)
    
    netphase = (netphase + np.pi) % (2*np.pi) - np.pi
    return k, angles, netphase

def randguess_cosines(tweezers, num_points):
    pixel_coordinates = np.argwhere(tweezers == 1)
    center = np.array(tweezers.shape) // 2
    vectors = pixel_coordinates - center
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    angles = -degrees(angles)
    distances = np.linalg.norm(vectors, axis=1)
    k = distances / num_points * num_points
    offset =  np.random.rand(*k.shape) * np.pi * 100
    netphase = np.zeros(np.shape(tweezers))
    for i in range(len(distances)):
        # offset = k[i]*(k[i]-1)/(2*len(distances))
        netphase += makegrating_cosine(k[i],offset[i],angles[i],num_points) / len(distances)
    
    netphase = (netphase + np.pi) % (2*np.pi) - np.pi
    return k, angles, netphase

def makegrating_sawtooth1d(k,offset, num_points):
    # Set the parameters
    k = k * 1.5
    x = np.linspace(0, 1, int(num_points*1.5))
    # Create a meshgrid
    # Generate the sawtooth pattern along the x direction
    sawtooth_pattern_x = sawtooth(2 * np.pi * k * x + offset)
    sawtooth_pattern = sawtooth_pattern_x[int((num_points*0.5) // 2): int((num_points*2.5) //2)]
    sawtooth_finalphase = sawtooth_pattern
    return sawtooth_finalphase * np.pi

def makegrating_sawtooth(k,offset, angle, num_points):
    # Set the parameters
    k = k * 1.5
    x = np.linspace(0, 1, int(num_points*1.5))
    y = np.linspace(0, 1, int(num_points*1.5))
    # Create a meshgrid
    x_mesh, y_mesh = np.meshgrid(x, y)
    # Set the sawtooth pattern parameters
    # Generate the sawtooth pattern along the x direction
    sawtooth_pattern_x = sawtooth(2 * np.pi * k * x_mesh + offset)
    sawtooth_pattern = rotate(sawtooth_pattern_x, angle, reshape = False)
    sawtooth_pattern = sawtooth_pattern[int((num_points*0.5) // 2): int((num_points*2.5) //2), int((num_points*0.5) // 2): int((num_points*2.5) //2)]
    sawtooth_finalphase = sawtooth_pattern
    return sawtooth_finalphase * np.pi

def initguess_sawtooths2(tweezers, num_points):
    pixel_coordinates = np.argwhere(tweezers == 1)
    center = np.array(tweezers.shape) // 2
    vectors = pixel_coordinates - center
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    angles = -degrees(angles)
    distances = np.linalg.norm(vectors, axis=1)
    k = distances / num_points * num_points / precision
    offset = 0
    netphase = 0
    
    x = np.linspace(0, 1, int(num_points*1.5))
    y = np.linspace(0, 1, int(num_points*1.5))
    # Create a meshgrid
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    for i in range(len(distances)):
        # offset = k[i]*(k[i]-1)/(2*len(distances))
        patterni = sawtooth(2*np.pi*k[i]*x_mesh + offset)
        patterni = rotate(patterni, angles[i], reshape = False)
        patterni = patterni[int((num_points*0.5) // 2): int((num_points*2.5) //2), int((num_points*0.5) // 2): int((num_points*2.5) //2)]

        netphase += patterni * np.pi / len(distances)

    netphase = (netphase + np.pi) % (2*np.pi) - np.pi
    return k, angles, netphase

def finguess_sawtooths(ks,offsets,angles, num_points):

    netphase = np.zeros(np.shape(tweezers))
    for i in range(len(ks)):
        # offset = k[i]*(k[i]-1)/(2*len(distances))
        netphase += makegrating_sawtooth(ks[i],offsets[i],angles[i],num_points) / len(ks)
    
    netphase = (netphase + np.pi) % (2*np.pi) - np.pi
    return netphase

# test2 = initguess_sawtooths(tweezers,1000)
# plt.imshow(test2)
# plt.colorbar()
# plt.show()
def modifiedGWSalgo2tar(n_rep,target1, target2, phase1, phase2, inputbeam, radiusinput):
    SIZE_X, SIZE_Y = np.shape(target1)
    inputbeam = set_circlemask(inputbeam, radiusinput)
    target_im1 = set_circlemask(target1, radiusinput)
    target_im2 = set_circlemask(target2,radiusinput)
    phaseguess1 = set_circlemask(phase1, radiusinput)
    phaseguess2 = set_circlemask(phase2,radiusinput)
    # unknownphase = set_circlemask((2*np.pi*np.random.rand(SIZE_X,SIZE_Y)-np.pi)*0.5, radiusinput)
    unknownphase = set_circlemask(np.ones((SIZE_X,SIZE_Y))*0.7, radiusinput) * 0

    w1=np.ones((SIZE_X,SIZE_Y))
    w2=np.ones((SIZE_X,SIZE_Y))
    w_prev1 = target_im1
    w_prev2 = target_im2
    errors1=[]
    errors2=[]

    for rep in range(n_rep):
        inputphase1 = (phaseguess1+unknownphase + 2*np.pi) %(2*np.pi) -np.pi
        inputphase2 = (phaseguess2+unknownphase + 2*np.pi) %(2*np.pi) -np.pi
        
        input1 = join_phase_ampl(inputphase1, inputbeam)
        input2 = join_phase_ampl(inputphase2, inputbeam)
        fftinput1 = sfft.fftshift(sfft.fft2(input1))
        fftinput2 = sfft.fftshift(sfft.fft2(input2))
        # uscaled_atomplane = np.repeat(np.repeat(u_atomplane, precision, axis=0), precision, axis=1)
        fftintensity1 = np.square(np.abs(fftinput1))
        fftintensity2 = np.square(np.abs(fftinput2))
        fftstdintensity1 = fftintensity1 /np.max(fftintensity1)
        fftstdintensity2 = fftintensity2 / np.max(fftintensity2)
        fftphase1 = np.angle(fftinput1)
        fftphase2 = np.angle(fftinput2)
        threshold = 0.01
        errors1.append(np.sum((target_im1 - fftstdintensity1)[target_im1 > threshold]))
        errors2.append(np.sum((target_im2 - fftstdintensity2)[target_im2 > threshold]))

        w1[target1 > threshold] = np.sqrt(target_im1[target1 > threshold] / fftstdintensity1[target1 > threshold]) * w_prev1[target1 > threshold]
        w2[target2 > threshold] = np.sqrt(target_im2[target2 > threshold] / fftstdintensity2[target2 > threshold]) * w_prev2[target2 > threshold]
        w1 = w1 / np.max(w1)
        w2 = w2 / np.max(w2)
        w_prev1 = w1
        w_prev2 = w2
        
        input1 = sfft.ifft2(sfft.ifftshift(join_phase_ampl(fftphase1,w1)))
        input2 = sfft.ifft2(sfft.ifftshift(join_phase_ampl(fftphase2,w2)))
        inputphase1 = np.angle(input1)
        inputphase2 = np.angle(input2)
        ## approx unknown phase here)
        unknownphase1 = (inputphase1 - phaseguess1 + np.pi) % (2*np.pi) - np.pi
        unknownphase2 = (inputphase2 - phaseguess2 + np.pi) % (2*np.pi) - np.pi
        unknownphase = ((unknownphase1 + unknownphase2) / 2 + np.pi) %(2*np.pi) - np.pi
        # unknownphase = np.mean([unknownphase1,unknownphase2], axis=0)
        kernel = np.ones((SIZE_X, SIZE_Y))
        shifts = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        shiftedarrays = [np.roll(unknownphase, shift, axis=(0, 1)) for shift in shifts]
        total_shiftedarrays = np.sum(shiftedarrays, axis=0)
        unknownphase = total_shiftedarrays / 9.0
        unknownphase = set_circlemask(unknownphase, radiusinput)
        # phase_slm=discretize_phase(phase_slm)
        # phase_slm = phase_slm
        # Final_ampl_phase = phase_slm
        # phase_slm=undiscretize_phase(phase_slm)
        
        
    
    return unknownphase, errors1, errors2, fftstdintensity1, fftstdintensity2

def modifiedGWSalgo3tar(n_rep,target1, target2, target3, phase1, phase2, phase3, inputbeam, radiusinput):
    SIZE_X, SIZE_Y = np.shape(target1)
    inputbeam = set_circlemask(inputbeam, radiusinput)
    target_im1 = set_circlemask(target1, radiusinput)
    target_im2 = set_circlemask(target2,radiusinput)
    target_im3 = set_circlemask(target3,radiusinput)

    phaseguess1 = set_circlemask(phase1, radiusinput)
    phaseguess2 = set_circlemask(phase2,radiusinput)
    phaseguess3 = set_circlemask(phase3,radiusinput)

    # unknownphase = set_circlemask((2*np.pi*np.random.rand(SIZE_X,SIZE_Y)-np.pi)*0.5, radiusinput)
    unknownphase = set_circlemask(np.ones((SIZE_X,SIZE_Y))*0.7, radiusinput) * 0
    w1=np.ones((SIZE_X,SIZE_Y))
    w2=np.ones((SIZE_X,SIZE_Y))
    w3=np.ones((SIZE_X,SIZE_Y))
    w_prev1 = target_im1
    w_prev2 = target_im2
    w_prev3 = target_im3
    errors1=[]
    errors2=[]
    errors3=[]

    for rep in range(n_rep):
        inputphase1 = (phaseguess1+unknownphase + np.pi) %(2*np.pi) -np.pi
        inputphase2 = (phaseguess2+unknownphase + np.pi) %(2*np.pi) -np.pi
        inputphase3 = (phaseguess3+unknownphase + np.pi) %(2*np.pi) -np.pi

        
        input1 = join_phase_ampl(inputphase1, inputbeam)
        input2 = join_phase_ampl(inputphase2, inputbeam)
        input3 = join_phase_ampl(inputphase3, inputbeam)

        fftinput1 = sfft.fftshift(sfft.fft2(input1))
        fftinput2 = sfft.fftshift(sfft.fft2(input2))
        fftinput3 = sfft.fftshift(sfft.fft2(input3))

        # uscaled_atomplane = np.repeat(np.repeat(u_atomplane, precision, axis=0), precision, axis=1)
        fftintensity1 = np.square(np.abs(fftinput1))
        fftintensity2 = np.square(np.abs(fftinput2))
        fftintensity3 = np.square(np.abs(fftinput3))

        fftstdintensity1 = fftintensity1 /np.max(fftintensity1)
        fftstdintensity2 = fftintensity2 / np.max(fftintensity2)
        fftstdintensity3 = fftintensity3 / np.max(fftintensity3)

        fftphase1 = np.angle(fftinput1)
        fftphase2 = np.angle(fftinput2)
        fftphase3 = np.angle(fftinput3)

        threshold = 0.01
        errors1.append(np.sum((target_im1 - fftstdintensity1)[target_im1 > threshold]))
        errors2.append(np.sum((target_im2 - fftstdintensity2)[target_im2 > threshold]))
        errors3.append(np.sum((target_im3 - fftstdintensity3)[target_im3 > threshold]))

        w1[target1 > threshold] = np.sqrt(target_im1[target1 > threshold] / fftstdintensity1[target1 > threshold]) * w_prev1[target1 > threshold]
        w2[target2 > threshold] = np.sqrt(target_im2[target2 > threshold] / fftstdintensity2[target2 > threshold]) * w_prev2[target2 > threshold]
        w3[target3 > threshold] = np.sqrt(target_im3[target3 > threshold] / fftstdintensity3[target3 > threshold]) * w_prev3[target3 > threshold]

        w1 = w1 / np.max(w1)
        w2 = w2 / np.max(w2)
        w3 = w3 / np.max(w3)

        w_prev1 = w1
        w_prev2 = w2
        w_prev3 = w3
      
        input1 = sfft.ifft2(sfft.ifftshift(join_phase_ampl(fftphase1,w1)))
        input2 = sfft.ifft2(sfft.ifftshift(join_phase_ampl(fftphase2,w2)))
        input3 = sfft.ifft2(sfft.ifftshift(join_phase_ampl(fftphase3,w3)))

        inputphase1 = np.angle(input1)
        inputphase2 = np.angle(input2)
        inputphase3 = np.angle(input3)

        ## approx unknown phase here)
        unknownphase1 = (inputphase1 - phaseguess1 +np.pi) % (2*np.pi) - np.pi
        unknownphase2 = (inputphase2 - phaseguess2 +np.pi) % (2*np.pi) - np.pi
        unknownphase3 = (inputphase3 - phaseguess3 +np.pi) % (2*np.pi) - np.pi

        unknownphase = ((unknownphase1 + unknownphase2+unknownphase3) / 3 +np.pi) %(2*np.pi) - np.pi

        shifts = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        shiftedarrays = [np.roll(unknownphase, shift, axis=(0, 1)) for shift in shifts]
        total_shiftedarrays = np.sum(shiftedarrays, axis=0)
        unknownphase = total_shiftedarrays / 9.0

        # unknownphase = np.mean([unknownphase1, unknownphase2, unknownphase3], axis=0)
        unknownphase = set_circlemask(unknownphase, radiusinput)
        # phase_slm=discretize_phase(phase_slm)
        # phase_slm = phase_slm
        # Final_ampl_phase = phase_slm
        # phase_slm=undiscretize_phase(phase_slm)
    
    return unknownphase, errors1, errors2,errors3, fftstdintensity1, fftstdintensity2,fftstdintensity3



def phasetestepsilon(u_int, target_im):
    max = np.max(u_int[target_im != 0]) #Max value of the obtained intensity at the tweezers position
    min = np.min(u_int[target_im != 0]) #Min value of the obtained intensity at the tweezers position
    error = (max-min)/(max+min)
    
    # error = np.sum(np.abs(u_int-target_im))
    #print("Error :", error)
    return error

def phasetestweights(w,target_im, startingimg, w_prev, std_int): # This weight function works only where the intensity == 1 (discrete tweezers)

    threshold = 0.3
    #             # w[target_im!=1] = np.sqrt(1 / (1-screwone(std_int[target_im!=1]))) * w_prev[target_im!=1]    
    w[startingimg > threshold] =  np.sqrt((target_im[startingimg> threshold] / std_int[startingimg> threshold])) * w_prev[startingimg> threshold]


    return (w)



def virtual_slmsimulation(phase_slmabberation, inputbeam, inputphase):
    combinedphase = (phase_slmabberation + inputphase + np.pi) % (2*np.pi) - np.pi
    # PS_shape=Beam_shape(x,y,255,0)
    virtual_slm = set_circlemask(join_phase_ampl(combinedphase,inputbeam), 650)
    virtual_slm = sfft.fftshift(sfft.fft2(virtual_slm))
    # virtual_slm = np.repeat(np.repeat(virtual_slm, precision, axis=0), precision, axis=1)
    virtual_slmintensity=np.square(np.abs(virtual_slm))
    virtual_slmintensity = virtual_slmintensity / np.max(virtual_slmintensity)

    return virtual_slmintensity

# derivephase(tweezers, np.ones(np.shape(sawtoothinitinputs)), sawtoothinitinputs, tweezers, 150, 650)
def derivephase(virtual_slm, inputbeam, shapephase, initialintensity, n_rep, radiusinput):
    target_im = virtual_slm
    SIZE_X, SIZE_Y = np.shape(shapephase)
    # The initial weights are all = 1.
    w=np.ones((SIZE_X*precision,SIZE_Y*precision))
    # The amplitude in the fourier plane is a Gaussian (beam)
    ### radius
    # inputbeam = Beam_shape(SIZE_Y,SIZE_X,300,0)
    # inputbeam = np.ones((SIZE_X,SIZE_Y))
    inputbeam = np.repeat(np.repeat(inputbeam, precision, axis=0), precision, axis=1)
    PS_shape=set_circlemask(inputbeam, radiusinput*precision)
    w_prev=target_im
    errors=[]
    u=np.zeros((SIZE_X,SIZE_Y),dtype=complex)
    # phase_slm=shapephase
    phase_slm = set_circlemask(shapephase, radiusinput)
    
    # phase_slm=2*np.pi*np.random.rand(SIZE_X,SIZE_Y)-np.pi

    for rep in range(n_rep):
        # Fourier plane, random phase (at the round 1) and gaussian beam
        phase_slm = np.repeat(np.repeat(phase_slm, precision, axis=0), precision, axis=1)
        u_slm=join_phase_ampl(phase_slm,PS_shape)
        # u_slm = join_phase_ampl(phase_slm,PS_shape.T)
            # To the real plane...
        u_slm = sfft.fft2(u_slm)
        u_atomplane = sfft.fftshift(u_slm)
        # Calculate the intensity
        intensity=np.square(np.abs(u_atomplane))
        # Let's renormalize the intensity in the range [0,1]
        std_int= intensity/np.max(intensity)
        # What's the distance from the target intensity?
        errors.append(phasetestepsilon(std_int, target_im))
        phase_atomplane=np.angle(u_atomplane)
        ## Here I don't care about the discretization of the phase because we're in real space (that is actually the fourier plane for the code)
        #Generate weights and use them to obtain u
        w=phasetestweights(w,target_im, initialintensity, w_prev,std_int)
        w = w / np.max(w) 
        w_prev=w
        u_atomplane=join_phase_ampl(phase_atomplane,w)
        # Back to our fourier plane
        # reshape_u_atomplane = u_atomplane.reshape(SIZE_X,precision,SIZE_Y,precision)
        # u_atomplane = np.mean(reshape_u_atomplane, axis=(-3,-1))
        u_atomplane = sfft.ifftshift(u_atomplane)
        u_slm = sfft.ifft2(u_atomplane)
        u_slm = u_slm.reshape(SIZE_X,precision,SIZE_Y,precision)
        u_slm = np.mean(u_slm, axis=(-3,-1))


        # The phase that we need to imprint by the SLM is :
        phase_slm=set_circlemask(np.angle(u_slm), radiusinput)
        # This part discretizes the values of phase. The SLM can only produce values in the range [0,255]
        # that corresponds to [0,2pi]. Some values (e.g. 0.5 => 0.5 * 2pi/255) cannot be reproduced by the SLM
        # This discretization takes that in account. (The final phase will be the discretized one)
        phase_slm=discretize_phase(phase_slm)
        phase_slm = phase_slm
        Final_ampl_phase = phase_slm
        phase_slm=undiscretize_phase(phase_slm)
    
    virtual_slmphase = phase_slm
    return virtual_slmphase, errors, std_int

def virtual_phaseretrievalcycle(phase_slmabberation, inputphase, std_int):
    virtual_slm = virtual_slmsimulation(phase_slmabberation, inputphase)
    virtual_slmphase, algoerrors = derivephase(virtual_slm, inputphase)
    error = np.sum(np.abs(std_int - virtual_slm))
    return virtual_slm, virtual_slmphase, error, algoerrors

def runcycles(numcycles, phase_slmabberation, phase_slminitial, slmintensity):
    errors = []
    algoerrors = []
    pictures = []
    phases = []
    corrections = []
    iterphase = phase_slminitial
    
    for i in range(numcycles):        
        virtual_slm = virtual_slmsimulation(phase_slmabberation, iterphase)
        virtual_slmphase, algoerror, outputimg = derivephase(virtual_slm, iterphase, slmintensity)
        error = np.sum(np.abs(slmintensity - virtual_slm))
        correction = (virtual_slmphase - iterphase + np.pi) % (2*np.pi) - np.pi
        iterphase = (iterphase - correction + np.pi) % (2*np.pi) - np.pi
        errors.append(error)
        algoerrors.append(algoerror)
        pictures.append(virtual_slm)
        phases.append(iterphase)
        corrections.append(correction)
    
    netphasecorrection = (iterphase - phase_slminitial + np.pi) % (2*np.pi) - np.pi
    return pictures, iterphase, errors, netphasecorrection, algoerrors, phases, corrections
            

def sawtooths(tweezers, num_points):
    pixel_coordinates = np.argwhere(tweezers == 1)
    center = np.array(tweezers.shape) // 2
    vectors = pixel_coordinates - center
    angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    for i in range(0,len(angles)):
        if vectors[:,1][i] < 0:
            angles[i] += np.pi
    distances = np.linalg.norm(vectors, axis=1)
    k = distances / num_points * num_points / precision
    numtweezcols = np.unique(np.nonzero(tweezers)[0])
    numtweezrows = np.unique(np.nonzero(tweezers)[1])
    kx = k * np.cos(angles)
    ky = k * np.sin(angles)
    print(len(numtweezrows))
    print(len(numtweezcols))
    offsetx = kx*(kx-1)*2*np.pi / 2 / (len(numtweezcols)-1)
    offsety = ky*(ky-1)*2*np.pi / 2 / (len(numtweezcols)-1)
    offsetmag = np.sqrt(offsetx*offsetx + offsety*offsety)
    offsetangle = np.arctan(offsety / offsetx)
    offsetonk = offsetmag * np.cos((offsetangle - angles))
  
    # Create a meshgrid
    netphase = np.zeros((num_points,num_points))
    xlen = np.shape(netphase)[1]
    ylen = np.shape(netphase)[0]
    
    for x in (range(0- xlen // 2, xlen- xlen // 2)):
        for y in (range(0- ylen // 2, ylen- ylen // 2)):
            xval = x / (xlen)
            yval = y / (ylen)
            if x == 0:
                xval = 0.001
            pixelvalue = 0
            theta2 = np.arctan(yval/xval)
            if xval < 0:
                theta2 += np.pi
            temppixval = (2*np.pi*k* np.sqrt(xval*xval+yval*yval) * np.sin(np.pi/2 -theta2 + angles) + offsetonk)
            # temppixval = [sawtooth(i) for i in temppixval]
            temppixval = ((temppixval/(2*np.pi) - np.floor(temppixval/(2*np.pi)))*2 - 1)
            pixelvalue = np.sum(temppixval) / len(angles)
     
            netphase[y + ylen // 2,x + xlen // 2] = pixelvalue
    
        
    netphase = (netphase + np.pi) % (2*np.pi) - np.pi
    return angles, netphase



### Convenience

def tweezerscrop(array, padding):
    # Find indices of nonzero elements
    nonzero_indices = np.nonzero(array)

    # Get the bounding box of nonzero elements
    min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])

    # Add padding to the bounding box
    min_row_pad = max(min_row - padding, 0)
    max_row_pad = min(max_row + padding + 1, array.shape[0])
    min_col_pad = max(min_col - padding, 0)
    max_col_pad = min(max_col + padding + 1, array.shape[1])

    # Extract the grid with padding
    grid_with_padding = array[min_row_pad:max_row_pad, min_col_pad:max_col_pad]

    return grid_with_padding, min_row_pad, max_row_pad, min_col_pad, max_col_pad