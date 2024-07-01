from .dependencies import *
from .settings import *
from .simulation import *

# def

# def camerafeedback_intensityuniformity(feedbackpencost, inputphase, target_im, slmobject, cameraobject):
#     """All inputs are assumed to be of the same dimensionality, 1300 by 1300. Note that magnification adds on to the target, so
#     if target is already 3900 by 3900 magnification of 2 will make the simulation space much(!) larger. Beamtypes available are Gaussian or Constant."""
#     # Remember, the calculation region is only numpixels by numpixels
#     targetintensity = targetintensity.copy()
#     # targetintensity = targetintensity / cp.max(targetintensity)
#     # Just in case we're using a highly precise target (so not delta function)
#     targetmagnification = cp.shape(targetintensity)[0] // numpixels
#     targetintensity = expand(targetintensity, magnification)
#     magnification = targetmagnification * magnification
#     slmphase = set_circlemask(expand(initialphase, magnification), numpixels *magnification)
#     inputbeam = set_circlemask(createbeam(beamtype, numpixels * magnification, sigma, mu), numpixels * magnification)
#     slmplane = join_phase_ampl(slmphase, inputbeam)
    
#     weights=cp.zeros((numpixels * magnification, numpixels*magnification))
#     weights_previous = targetintensity.copy()
    
#     # stdinttracker = [] # For use in error calculations
#     tweezerlocation = cp.where(targetintensity == 1)
#     err_maxmindiff = []
#     err_uniformity = []
#     err_powereff = []

#     for _ in range(iterations):
#         startingpower = cp.sum(cp.abs(slmplane)**2)
#         fourierplane = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slmplane), norm="ortho"))
#         fourierintensity = cp.square(cp.abs(fourierplane))
#         stdint = cp.divide(fourierintensity, cp.max(fourierintensity))

#         err_maxmindiff.append(Err_MaxMinDiff(stdint, tweezerlocation))
#         err_uniformity.append(Err_Uniformity(stdint, targetintensity))
#         err_powereff.append(Err_PowerEff(stdint, tweezerlocation))

#         weights = costfunction(weights, weights_previous, targetintensity, stdint, harmonicremoval, badharmonics_pixelcoords)
#         weights_previous = weights.copy()
#         ## This might be a bit confusing, but weights is now the amplitude and we re-combine it with the phase to get the next iteration
#         fourierangle = cp.angle(fourierplane)
#         fourierplane = join_phase_ampl(fourierangle, weights)
#         slmplane = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(fourierplane), norm="ortho"))     
#         endingpower = cp.sum(cp.abs(slmplane)**2)
#         slmplane = cp.multiply(cp.divide(slmplane, endingpower), startingpower)
#         slmplane_numpixels = slmplane.copy()
#         slmplane_numpixels = cp.mean(slmplane_numpixels.reshape(numpixels, magnification, numpixels, magnification), axis=(-3,-1))
        
#         slmphase = undiscretize_phase(discretize_phase(set_circlemask(cp.angle(slmplane_numpixels), numpixels)))
#         readout_slmphase = slmphase.copy()
#         slmplane = join_phase_ampl(expand(slmphase, magnification), inputbeam)
    
#     errors = [err_maxmindiff, err_uniformity, err_powereff]
#     labels = ["MaxMinDiff","Uniformity", "Power Efficiency"]

#     readout = OptimizedOutput()
#     readout.set_all(readout_slmphase, stdint, errors, labels)
    
#     return readout










