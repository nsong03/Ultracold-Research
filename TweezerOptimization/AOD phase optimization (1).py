# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:06:12 2024

@author: luyue
"""


import numpy as np
import scipy
from scipy.fft import fft, ifft, fftfreq
import torch
# import torchvision
from torch import nn
# import matplotlib
from matplotlib import pyplot as plt
# import tqdm
import ipyparallel as ipp
from matplotlib import colormaps
from collections import OrderedDict
# from util import idx_to_boolarray, boolarray_to_idx, idx_to_szarray, szarray_to_idx, boolarray_to_szarray, szarray_to_boolarray

class Tones:
    def __init__(self, freqs, amps):
        self.freqs = np.array(freqs) # todo: sort this 
        self.amps = np.array(amps) # todo: sort this using freqs
        self.realamps = np.abs(amps)
        self.phis = np.mod(np.angle(amps), 2*np.pi) 
        self.att = np.abs(amps)**2
        # self.tones = np.vstack([freqs,amps]).T
        tones = dict(zip(freqs, amps))
        self.tones = OrderedDict(sorted(tones.items()))
    def visualize(self, x_start, x_end):
        fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=False)
        colors = plt.cm.hsv(self.phis/2/np.pi)
        axs[0].bar(self.freqs, self.realamps, color = colors, width = 0.1)
        axs[0].set_xlim(x_start, x_end)
        axs[1].plot(self.freqs[1:],np.mod(self.phis[1:]-self.phis[:-1]-np.linspace(0, 2*PI, len(self.freqs))[:-1]+PI,2*PI)-PI,marker=".")
        # axs[1].plot(self.freqs,np.mod(self.phis, 2*PI))
        plt.show()
    def visualize_many(self, x_start, x_end, *others):
        num_rows = len(others)+1
        fig, axs = plt.subplots(ncols=1, nrows=num_rows, sharex='col', sharey='col')
        for counter,tunes_obj in enumerate(others.append(self)):
            colors = plt.cm.hsv(tunes_obj.phis/2/np.pi)
            axs[counter].bar(tunes_obj.freqs, tunes_obj.realamps, color = colors, width = 0.2)
            axs[counter].set_xlim(x_start, x_end)
        plt.show()
    def num_mult(self, factor):
        return Tones(self.freqs, factor*self.amps)
    def add(self, other, factor = 1):
        new_tones = self.tones
        for otherfreq,otheramp in zip(other.freqs,other.amps):
            if otherfreq in new_tones:
                new_tones[otherfreq] += otheramp * factor
            else:
                new_tones[otherfreq] = otheramp * factor
        new_tones = OrderedDict(sorted(new_tones.items()))
        # return new_tones
        return Tones(list(new_tones.keys()), list(new_tones.values()))
    def multiply(self, other, factor = 1):
        new_tones = dict()
        for selffreq,selfamp in zip(self.freqs,self.amps):
            for otherfreq,otheramp in zip(other.freqs,other.amps):
                freq_diff = round(selffreq - otherfreq,5)
                amps_diff = selfamp * np.conj(otheramp) * factor/2
                if freq_diff < 0:
                    freq_diff = -freq_diff
                    amps_diff = np.conj(amps_diff)
                if freq_diff in new_tones:
                    new_tones[freq_diff] += amps_diff
                else:
                    new_tones[freq_diff] = amps_diff
                freq_sum = round(selffreq + otherfreq,5)
                amps_sum = selfamp * otheramp * factor/2
                if freq_sum in new_tones:
                    new_tones[freq_sum] += amps_sum
                else:
                    new_tones[freq_sum] = amps_sum
        new_tones = OrderedDict(sorted(new_tones.items()))
        # return new_tones
        return Tones(list(new_tones.keys()), list(new_tones.values()))
                


def AODphase_func(t, omega_list, phi_list, mod_depth_list):
    if isinstance(mod_depth_list, np.ndarray):
        mod_depth_list = mod_depth_list
    else:
        mod_depth_list = np.full(len(omega_list), mod_depth_list)
    return np.sum(mod_depth_list * np.cos(omega_list*t+phi_list))


PI = np.pi

optimize_yes = True
plot_yes = True
Ntweezer = 40
halfNtwz = Ntweezer/2
mod_depth  = PI/3/np.sqrt(Ntweezer)

center_f = 100
delta_f = 1
start_f = center_f - (Ntweezer-1)/2 * delta_f
end_f = center_f + (Ntweezer-1)/2 * delta_f
f_list = np.arange(start_f,end_f+delta_f, delta_f)\
    +0.05*(-1)**np.arange(Ntweezer)\
    +0*0.01* np.concatenate((np.zeros(Ntweezer // 2), np.ones( Ntweezer- Ntweezer // 2)))
# f_list = np.arange(start_f,end_f+delta_f, delta_f)
# for ii,f in enumerate(f_list):
#     if f>100: f_list[ii]+=0.1
omega_list = 2*PI * f_list


delta_phi_list = 2*PI * np.arange(Ntweezer-1)/(Ntweezer-1)
# half_delta_phi_list = PI * np.arange(halfNtwz-1)/(halfNtwz-1)
# half_phi_list1 = np.cumsum(np.append([0],half_delta_phi_list))
# half_phi_list2 = np.cumsum(np.append([np.pi],half_delta_phi_list))
# phi_list = np.empty((half_phi_list1.size + half_phi_list2.size,), dtype=half_phi_list1.dtype)
# phi_list[0::2] = half_phi_list1
# phi_list[1::2] = half_phi_list2

phi_list = np.cumsum(np.append([0],delta_phi_list))
 #+ np.random.random(Ntweezer) /3
# phi_list = np.zeros(Ntweezer)
# phi_list = np.arange(Ntweezer)
# phi_list = np.random.random(Ntweezer) * 2*np.pi
# phi_list = np.array([-1.33752532, -0.65970151, -0.71825826, -0.09208327,  0.64692483,
#  1.75663285,  3.20380986,  4.25861905,  6.03176181,  7.89407776,
# 10.11952813, 12.04396552, 14.34157623, 17.91088194, 19.47540109,
# 23.22457256, 25.73142014, 30.1675673 , 32.63045535, 36.44295336,
# 39.79192042, 44.22843148, 48.61457814, 52.71084647, 57.45734116,
# 62.24770336, 67.24170224, 72.94785145, 78.55476839, 84.58747411,
# 89.9865872 ])

mod_list = mod_depth*np.ones(Ntweezer)
phase_list = mod_list * np.exp(1j*phi_list)
tones_first = Tones(f_list, phase_list)

tones_second = tones_first.multiply(tones_first, factor = 1/2)
# tones_second.visualize(80,120)

tones_third = tones_second.multiply(tones_first, factor = 1/3)

tones_approx_twz = tones_first.add(tones_third, factor = -1)

if plot_yes:
    # tones_first.visualize(80,120)
    # tones_third.visualize(80,120)
    tones_approx_twz.visualize(90,110)


def thirdtone_cost_func_0(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    return np.sum(abs(np.array([tones_third.tones[ff] for ff in f_list])))

def thirdtone_cost_func_2(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    return np.sum(abs(np.array([tones_third.tones[ff] for ff in f_list]))**2)

def thirdtone_cost_func_1(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    A = np.array([tones_third.tones[ff] for ff in f_list])
    B = np.array([tones_first.tones[ff] for ff in f_list])
    
    return np.sum(np.real(np.vdot(A,B)))

def thirdtone_cost_func_3(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    A = np.array([tones_third.tones[ff] for ff in f_list])
    B = np.array([tones_first.tones[ff] for ff in f_list])
    C = np.real(A * np.conj(B))
    return len(C)*np.sum(C*C) - np.sum(C)**2

def thirdtone_cost_func_4(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    A = np.array([tones_third.tones[ff] for ff in f_list])
    B = np.array([tones_first.tones[ff] for ff in f_list])
    C = abs((A-B)**2)
    return len(C)*np.sum(C*C) - np.sum(C)**2
def thirdtone_cost_func_5(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    A = np.array([tones_third.tones[ff] for ff in f_list])
    B = np.array([tones_first.tones[ff] for ff in f_list])
    C = abs((A+B)**2)
    return len(C)*np.sum(C*C) - np.sum(C)**2

def thirdtone_cost_func_6(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    tones_fourth = tones_third.multiply(tones_first, factor = 1/4)
    tones_fifth = tones_fourth.multiply(tones_first, factor = 1/5)
    A3 = np.array([tones_third.tones[ff] for ff in f_list])
    A1 = np.array([tones_first.tones[ff] for ff in f_list])
    A5 = np.array([tones_fifth.tones[ff] for ff in f_list])
    C = abs((A1-A3+A5)**2)
    return 1e4*(len(C)*np.sum(C*C) - np.sum(C)**2)

def thirdtone_cost_func_7(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    tones_fourth = tones_third.multiply(tones_first, factor = 1/4)
    tones_fifth = tones_fourth.multiply(tones_first, factor = 1/5)
    tones_sixth = tones_fifth.multiply(tones_first, factor = 1/6)
    tones_seventh = tones_sixth.multiply(tones_first, factor = 1/7)
    A3 = np.array([tones_third.tones[ff] for ff in f_list])
    A1 = np.array([tones_first.tones[ff] for ff in f_list])
    A5 = np.array([tones_fifth.tones[ff] for ff in f_list])
    A7 = np.array([tones_seventh.tones[ff] for ff in f_list])
    C = abs((A1-A3+A5-A7)**2)
    return len(C)*np.sum(C*C) - np.sum(C)**2

def thirdtone_cost_func_heating(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    # tones_fourth = tones_third.multiply(tones_first, factor = 1/4)
    # tones_fifth = tones_fourth.multiply(tones_first, factor = 1/5)
    A3 = []
    for ff in tones_third.freqs:
        for f_ref in f_list:
            if 0.001<abs(ff-f_ref)<0.3:
                A3.append(tones_third.tones[ff])    
    return np.sum(abs(np.array(A3)))

def thirdtone_cost_func_maxpower(phi_opt): 
    phase_list = mod_list * np.exp(1j*phi_opt)
    tones_first = Tones(f_list, phase_list)
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    A = np.array([tones_third.tones[ff] for ff in f_list])
    B = np.array([tones_first.tones[ff] for ff in f_list])
    C = abs((A+B)**2)
    return np.sum(C)

print(thirdtone_cost_func_heating(phi_list))

print(f"phi opt init cost func = {thirdtone_cost_func_heating(phi_list)}")

if optimize_yes:
    # optimized = scipy.optimize.basinhopping(thirdtone_cost_func_6, phi_list, niter=100, T=1.0, stepsize=1, \
    #                                         minimizer_kwargs=None, take_step=None, accept_test=None, callback=None,\
    #                                         interval=50, disp=True, niter_success=None, seed=None,\
    #                                         target_accept_rate=0.5, stepwise_factor=0.9)

    optimized = scipy.optimize.minimize(thirdtone_cost_func_heating, phi_list, args=(),\
                                        method=None, jac=None, hess=None, hessp=None, \
                                            bounds=None, constraints=(), tol=None, \
                                                callback=None, options=None)
    print(f"phi opt optimized cost func = {optimized.fun}")
    
    phi_list = optimized.x
    phase_list = mod_list * np.exp(1j*phi_list)
    
    # print(thirdtone_cost_func_temp(optimized.x))
    
    tones_first = Tones(f_list, phase_list)
    # tones_first.visualize(80,120)
    
    tones_second = tones_first.multiply(tones_first, factor = 1/2)
    tones_third = tones_second.multiply(tones_first, factor = 1/3)
    # tones_third.visualize(80,120)
    
    tones_fourth = tones_third.multiply(tones_first, factor = 1/4)
    tones_fifth = tones_fourth.multiply(tones_first, factor = 1/5)
    # tones_fifth.visualize(80,120)
    
    tones_sixth = tones_fifth.multiply(tones_first, factor = 1/6)
    tones_seventh = tones_sixth.multiply(tones_first, factor = 1/7)
    # tones_seventh.visualize(80,120)
    
    tones_approx_twz = tones_first.add(tones_third.add(tones_fifth,factor=-1), factor = -1)
    # tones_approx_twz = tones_first.add(tones_third.add(tones_fifth.add(tones_seventh,factor=-1),factor=-1), factor = -1)
    # tones_approx_twz.visualize(80,120)
    
    
    if plot_yes:
        # tones_first.visualize(80,120)
        # tones_third.visualize(80,120)
        # tones_fifth.visualize(80,120)
        # tones_seventh.visualize(80,120)
        tones_approx_twz.visualize(90,110)


# %% section realtime
# phi_list=optimized.x
# phi_list = np.random.random(Ntweezer) * 2*np.pi
# delta_phi_list = 2*PI * np.arange(Ntweezer-1)/(Ntweezer-1)
# phi_list = np.cumsum(np.append([0],delta_phi_list))


f_gcd = (np.gcd.reduce((1e6*f_list).astype(int))/1e6)
T_num = np.min([np.max([1,int(40*f_gcd)]),40])
RBW = 1 / T_num

N_points = int(2000*T_num/f_gcd)

T_cycle = T_num / f_gcd
t_list = np.linspace(0.0, T_cycle, N_points , endpoint=False)
phase_t_list = np.array([AODphase_func(t, omega_list, phi_list, mod_depth) for t in t_list])
wavefront_t_list = np.exp(1j*phase_t_list)
phase_t_list_second_order = phase_t_list**2 / 2
phase_t_list_third_order = phase_t_list**3 /6

fig, axs = plt.subplots(ncols=3, nrows=4, sharex='col', sharey='col')

axs[0,0].plot(t_list[:N_points//T_num], phase_t_list[:N_points//T_num])
axs[1,0].plot(t_list[:N_points//T_num], phase_t_list_second_order[:N_points//T_num])
axs[2,0].plot(t_list[:N_points//T_num], phase_t_list_third_order[:N_points//T_num])
axs[3,0].plot(t_list[:N_points//T_num], np.imag(wavefront_t_list[:N_points//T_num]))
axs[3,0].plot(t_list[:N_points//T_num], np.real(wavefront_t_list[:N_points//T_num]))
# plt.show()

fft_list = fftfreq(N_points, T_cycle/N_points)[1:N_points//2]
fft_phase_t_list = fft(phase_t_list)[1:N_points//2]/np.sqrt(N_points)
fft_phase_t_list_second_order = fft(phase_t_list_second_order)[1:N_points//2]/np.sqrt(N_points)
fft_phase_t_list_third_order = fft(phase_t_list_third_order)[1:N_points//2]/np.sqrt(N_points)
fft_wavefront_t_list = fft(wavefront_t_list)[1:N_points//2]/np.sqrt(N_points)
fft_wavefront_att = np.abs(fft_wavefront_t_list)**2


axs[0,1].plot(fft_list, np.abs(fft_phase_t_list))
axs[0,1].set_xlim(70,130)
axs[0,2].plot(fft_list, np.abs(fft_phase_t_list))
# axs[0,2].set_xlim(0,520)

axs[1,1].plot(fft_list, np.abs(fft_phase_t_list_second_order))
axs[1,1].set_xlim(70,130)
axs[1,2].plot(fft_list, np.abs(fft_phase_t_list_second_order))
# axs[1,2].set_xlim(0,520)
    
axs[2,1].plot(fft_list, np.abs(fft_phase_t_list_third_order))
axs[2,1].set_xlim(70,130)
axs[2,2].plot(fft_list, np.abs(fft_phase_t_list_third_order))
# axs[2,2].set_xlim(0,520)

axs[3,1].plot(fft_list, abs(fft_wavefront_t_list))
axs[3,1].set_xlim(70,130)
axs[3,2].plot(fft_list, abs(fft_wavefront_t_list))
# axs[3,2].set_xlim(0,520)
plt.show()


    
count_start = int(start_f*T_num/f_gcd-1)
count_end = int(end_f*T_num/f_gcd) 
twz_att = fft_wavefront_att[count_start:count_end:int(T_num/f_gcd)]
# plt.plot(fft_list, fft_wavefront_att)
plt.plot(fft_list, abs(fft_wavefront_t_list))
plt.title(f"twz avg int = {np.mean(twz_att):.3f}, uniformity std = {np.sqrt(np.mean(twz_att**2) - np.mean(twz_att)**2):.3f}")
plt.xlim(89,111)
plt.show()


# count_start = int(start_f*T_num/delta_f-1)
# count_end = int(end_f*T_num/delta_f)    
# plt.plot(fft_list[count_start:count_end], abs(fft_phase_t_list_third_order)[count_start:count_end],marker="*")
# # plt.xlim(99,101)
# plt.show()

# 
def thirdorder_cost_func(phi_opt): 
    count_start = int(start_f*T_num/delta_f-1)
    count_end = int(end_f*T_num/delta_f)  
    phase_t_list = np.array([AODphase_func(t, omega_list, phi_opt, mod_depth) for t in t_list])
    phase_t_list_third_order = phase_t_list**3 /6
    fft_phase_t_list_third_order = fft(phase_t_list_third_order)[1:N_points//2]/np.sqrt(N_points)
    return np.sum(abs(fft_phase_t_list_third_order)[count_start:count_end])

def thirdorder_cost_func_real(phi_opt): 
    count_start = int(start_f*T_num/delta_f-1)
    count_end = int(end_f*T_num/delta_f)  
    phase_t_list = np.array([AODphase_func(t, omega_list, phi_opt, mod_depth) for t in t_list])
    phase_t_list_third_order = phase_t_list**3 /6
    fft_phase_t_list = fft(phase_t_list)[1:N_points//2]/np.sqrt(N_points)
    fft_phase_t_list_third_order = fft(phase_t_list_third_order)[1:N_points//2]/np.sqrt(N_points)
    return np.real(np.vdot(fft_phase_t_list_third_order[count_start:count_end],fft_phase_t_list[count_start:count_end]))**2


# phi_list = np.zeros(Ntweezer)
# print(f"phi opt init cost func = {thirdorder_cost_func(phi_list)}")

# optimized = scipy.optimize.minimize(thirdorder_cost_func, phi_list, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)

# print(f"phi opt init cost func = {optimized.fun}")

# print(optimized.x)

# t_list = np.linspace(0, stop)