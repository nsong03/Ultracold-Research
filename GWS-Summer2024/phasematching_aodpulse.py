# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:17:12 2024

@author: songo
"""

from slmfunctions.dependencies import *
from slmfunctions.settings import *
from slmfunctions.general import *
from slmfunctions.imageprocessing import *
from slmfunctions.phaseretrieval import *

imgname = f"3by3_3spacing_phase" # LOOKUP NAME
## LOAD IN PRECOMPUTE
save_dir = r'C:\cleen\nsong\2-Research\UAC Research\PhaseStorage\May2024'
phaseimg_path = os.path.join(save_dir, f"{imgname}_phase.bmp")
intimg_path = os.path.join(save_dir, f"{imgname}_virtualint.jpg")

phaseimg = np.array(Image.open(phaseimg_path).convert('L')) / 255*2*np.pi - np.pi
intensityimg = np.array(Image.open(intimg_path).convert('L'))
