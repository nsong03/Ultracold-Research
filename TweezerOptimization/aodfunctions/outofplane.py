from .dependencies import *
from .settings import *
from .general import *










########## Direct Computation dependencies 


def Bernstein_Poly(x, n, i):
    return np.where(x <= 1, binom(n,i) * x**i * (1-x)**(n-i),            # If x <= 1     
                    0)
def Int_Bernstein_Poly(x, n, i):
    return np.where(x <= 1, x**(i+1) * binom(n,i) * hyp2f1(i+1,i-n,i+2,x) / (i+1),            # If x <= 1     
                    1/(n+1))

def dBernstein_Poly(x, n, i):
    return binom(n,i) * (i-n*x) * x**(i-1) * (1-x)**(n-i-1)
def ddBernstein_Poly(x, n, i):
    return binom(n,i) * (i**2+(n-1)*n*x**2+i*(-1-2*(n-1)*x)) * x**(i-2) * (1-x)**(n-i-2)

def triangle_to_linear(row, col):
    return row*(row+1)/2 + col

def linear_to_triangle(linear):
    # Calculate the row using the quadratic formula
    row = int((np.sqrt(8*linear + 1) - 1) // 2)
    # Calculate the first linear index in that row
    start_of_row = row * (row + 1) // 2
    # Calculate the column
    col = linear - start_of_row
    return row, col

def backtract_beam_rad(waist,focal_length,wavelength):
    return focal_length * wavelength/(np.pi*waist)
def forward_focal_waist(radius,focal_length,wavelength):
    return focal_length * wavelength/(np.pi*radius)












 

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


















