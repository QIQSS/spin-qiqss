# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Librairies

# %%
# Librairies et fonctions
# %gui qt
# %matplotlib inline
# %load_ext autoreload
# %autoreload 3

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from time import sleep
import datetime
import h5py
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import importlib
import re
import os
import gc
from time import time


import sys
sys.path.append("..")
# Functions for OPX use
from qm.qua import (
    program,
    fixed,
    declare_stream,
    wait, align,
    stream_processing,
    declare,
    save, assign,
    play, amp,
    for_, for_each_,
    while_,
    pause,
    update_frequency,
    reset_if_phase,
    ramp_to_zero,
    strict_timing_
)
from qm import QuantumMachinesManager, SimulationConfig
from qualang_tools.loops import from_array

### UTIL
from utils.qua_custom import (
    readout_demod_macro,
    readout_demod_sliced_macro,
    close_everything,
    duplicate_element,
    readout_demod_sliced_macro,
    make_gate_sequence,
)
from utils.file_saving import (
    make_path_fn,
    expand_filename,
    get_cell_content,
    sweep_file,
    get_file_variables,
    get_file_code,
)
from utils.spin_fit_analysis import (
    fit_slices,
    DistributionSTFitResult,
    find_iq_rotation
)
from utils.functions import (
    oscillations_rabi,
    oscillations_rabi_fourier
)

from utils import plots
from utils.plots import mk_extent

def collect():
    collected = gc.collect() 
    print(f"Garbage collector: Collected {collected} objects.")



# %%
data_path = "D:/Intel_Tunel_Falls_12QD_01/data"
path = make_path_fn(data_path)

# %% [markdown]
# # Sweep fréquence

# %%
# from config import qop_ip, cluster_name, u, config

import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

frequencies = np.arange(200e6, 1000e6+1, .01e6)

############################
with program() as sweep_rf:

    f = declare(int)  # QUA variable for the readout frequency
    r_st = declare_stream()
    t_st = declare_stream()

    with for_(*from_array(f, frequencies)):

        update_frequency("RF-SET1", f)
        r, t = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
                mode="rt",
        )

        save(r, r_st)
        save(t, t_st)

    with stream_processing():
        r_st.save_all("R")
        t_st.save_all("Theta")


qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(sweep_rf)

filename = expand_filename(path()+"%T_freq_sweep.hdf5")
with sweep_file(
    filename,
    ax_names=["frequence (Hz)"],
    ax_values=[frequencies],
    out_names=["R", "Theta"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %%
with h5py.File(filename, 'r') as f:
    ax_names = f["data"].attrs["sweeped_ax_names"]
    out_names = f["data"].attrs["result_data_names"]

    x = f["data"]["frequence (Hz)"][:]
    r = f["data"]["R"][:]
    t = f["data"]["Theta"][:]

# %%
# %matplotlib qt
# traitement (déroule, dépente)
t = np.unwrap(t)
t = t - (t[-1]-t[0])/(x[-1]-x[0]) * x
###
# trace
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(x*1e-6, r)
ax[1].plot(x*1e-6, t)

ax[0].set_ylabel("R")
ax[0].grid(True)
ax[0].set_title(filename)

ax[1].set_xlabel("Frequency (MHz)")
ax[1].set_ylabel("Theta")
ax[1].grid(True)

fig.tight_layout()
plt.show()

# %% [markdown]
# # Video mode
# Mettre (0, 0) au point d'initialisation.

# %%
# %gui qt

import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config
from utils.videomode_lib.videomode import VideoModeWindow, Sweep
from params import *


short_axis = Sweep.from_nbpts(-4e-3, 4e-3, 101, "P3", attenuation_db, interlace=0)
long_axis = Sweep.from_nbpts(-4e-3, 4e-3, 101, "P2", attenuation_db, interlace=1)

before_wait = 100
short_duration = cw_len + before_wait
long_duration = short_duration * short_axis.nbpts

# Def config
with program() as videomode:
    update_frequency("RF-SET1", cw_readout_freq)
    n, m = declare(fixed), declare(fixed)
    i_st = declare_stream()
    q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)
    with while_(True):
        ramp_to_zero(short_axis.element)
        ramp_to_zero(long_axis.element)
        pause()
        with for_each_(n, long_axis.stickysteps * 10):
            play("step"*amp(n), long_axis.element, duration=long_duration*u.ns)
            with for_each_(m, short_axis.stickysteps * 10):
                play("step"*amp(m), short_axis.element, duration=short_duration*u.ns)
                wait(2*before_wait * u.ns, "RF-SET1")
                i, q = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                save(i, i_st)
                save(q, q_st)
            ramp_to_zero(short_axis.element)
        ramp_to_zero(long_axis.element)
    
    with stream_processing():
        i_st.buffer(short_axis.nbpts).buffer(long_axis.nbpts).save("I")
        q_st.buffer(short_axis.nbpts).buffer(long_axis.nbpts).save("Q")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qmm.close_all_qms()
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(videomode)


vm = VideoModeWindow.from_job(
   job, save_path = path(),
   out_name = "I",
   short_axis = short_axis,
   long_axis = long_axis,
   play = 1
)
# vm2 = VideoModeWindow.from_job(
#    job, save_path = path(),
#    out_name="Q",
#    short_axis=short_axis,
#    long_axis=long_axis,
#    play = 1
# )

# %% [markdown]
# # Sweep 1d: point de mesure
# trouver le point de mesure optimal

# %% [markdown]
# ## Code de mesure

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Parameters
n_avg = 300
n_detuning = 101

p2_list = np.linspace(-0.5e-3, 1.3e-3, n_detuning)[::-1]
p3_list = -p2_list

############################
# Gate space
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
# Sweep du point de lecture
with program() as psb_readout_point:
    n = declare(int)
    p2, p3 = declare(fixed), declare(fixed)

    i_st = declare_stream()
    q_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)
    sequence.ramp_to_zero()
    save(cw_amp,'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        with for_each_((p2, p3), (p2_list*gain, p3_list*gain)):

            reset_if_phase("RF-SET1")
            sequence.add_step(voltage_point_name="init", duration=10_000)
            # sequence.add_step(voltage_point_name="zero_dc", ramp_duration=1_500, duration=16)
            sequence.add_step(voltage_point_name="load_deep", duration=100_000)

            align("RF-SET1", *gates)
            #wait(to_readout_duration, "RF-SET1")
            sequence.add_step(level=[p2, p3], duration=cw_len)#, ramp_duration=to_readout_duration)
            wait(5*u.us, "RF-SET1")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            save(i, i_st)
            save(q, q_st)

            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
            sequence.ramp_to_zero(duration=1)

    with stream_processing():
        i_st.buffer(len(p2_list)).save_all("I")
        q_st.buffer(len(p2_list)).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_point)

filename = expand_filename(path()+"%T_readout_point.hdf5")
print(filename)
with sweep_file(
    filename,
    ax_names=["count", "detuning_axis"],
    ax_values=[n_avg, len(p2_list)],
    out_names=["I", "Q"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    p2_list = p2_list,
    p3_list = p3_list,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %% [markdown]
# ## Analyse et affichage

# %%
# Chargement des données
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]

    p2_list = file["meta"].attrs["p2_list"]
    p3_list = file["meta"].attrs["p3_list"]

    n_count = file["data/count"].shape[0]
    n_detuning = file["data/detuning_axis"].shape[0]

# Calcul des histogrammes
hist, bins_i, bins_q = np.histogram2d(data_i.flatten(), data_q.flatten(), bins=100)
hist_i, hist_q = hist.sum(axis=1), hist.sum(axis=0)
bins_ic, bins_qc = (bins_i[1:] + bins_i[:-1]) / 2, (bins_q[1:] + bins_q[:-1]) / 2
theta = find_iq_rotation(data_i, data_q, iq_phase, log_norm=False, verbosity=2)


# %% [markdown]
# ### Trouver le point de détuning optimal

# %%
detuning_idx = 35

hist_vs_detuning = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_i)[0],
    0, data_i
)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    hist_vs_detuning,
    extent=[0, n_detuning, bins_i[0], bins_i[-1]],
    interpolation="none",
    aspect="auto",
    origin="lower"
)
ax.text(0, bins_i[0], "(1, 1)", c="r", size=16)
ax.text(n_detuning, bins_i[0], "(2, 0)", c="r", ha="right", size=16)
#ax.set_title("Histogramme à différent detuning")
ax.set_xlabel("Detuning index")
ax.set_ylabel("Histogram I")
cb = fig.colorbar(im, ax=ax, label="Count")

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
tick_positions = np.linspace(0, n_detuning - 1, 6, dtype=int)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels([f"{p2_list[i]:.4f}" for i in tick_positions])
ax2.set_xlabel("P2 (V)")
plt.tight_layout()

if detuning_idx is not None:
    ax.axvline(detuning_idx, c="red", ls=":")
    print(
        f"Couple de détuning: (P2, P3) = "
        f"({round(p2_list[detuning_idx], 5)}, {round(p3_list[detuning_idx], 5)})"
    )
    #plt.text(0.5, 0.9, f"ramp time: {to_readout_duration}ns", c="w", transform=plt.gca().transAxes)
    #plt.savefig(fr"R:\Student\Alexis\mesures\2026-intel\2026-03-06-tuning\ramp_time_{to_readout_duration}ns.svg")
   
    plt.figure(figsize=(6,2))
    plt.plot(bins_ic, hist_vs_detuning[:, detuning_idx])
    plt.xlabel("Histogram I")
    plt.ylabel("Count")

# %% [markdown]
# # Sweep 1d: temps de mesure (sliced readout)
# trouver le temps optimal

# %% [markdown]
# ## Code de mesure

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_slices = 200
n_cores = 2
n_avg = 8_000
print(f"total readout time {n_slices*cw_len_short}")

############################
# Démodulation sliced
rf_sets = duplicate_element(config, "RF-SET1", n_cores)

sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_readout_time:
    n = declare(int)

    i_st = declare_stream()
    q_st = declare_stream()

    for rf in rf_sets:
        update_frequency(rf, cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        for rf in rf_sets:
            reset_if_phase(rf)
            
        sequence.add_step(voltage_point_name="init")
        sequence.add_step(voltage_point_name="zero_dc", ramp_duration=1_500, duration=0)
        sequence.add_step(voltage_point_name="load_deep", duration=100_000, ramp_duration=20)

        # sequence.add_step(voltage_point_name="empty", duration=100_000)
        # sequence.add_step(voltage_point_name="load_deep", ramp_duration=100, duration=100_000)

        align(*rf_sets, *gates)
        # for rf in rf_sets:
        #     wait(to_readout_duration, *rf_sets)
        sequence.add_step(
            voltage_point_name="readout",
            # ramp_duration=to_readout_duration,
            duration=n_slices * cw_len_short
        )

        i, q = readout_demod_sliced_macro(
            element="RF-SET1",
            operation="readout_short",
            element_output="out",
            amplitude=cw_amp*10,
            n_slices=n_slices,
            cw_len=cw_len_short,
            i_st=i_st,
            q_st=q_st,
            n_cores=n_cores,
        )
        align(*rf_sets, *gates)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
        sequence.ramp_to_zero(duration=1)

    with stream_processing():
        i_st.buffer(n_slices).save_all("I")
        q_st.buffer(n_slices).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_time)

filename = expand_filename(path()+"%T_readout_time.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "time_us"],
    ax_values=[n_avg, np.arange(0, n_slices * cw_len_short * 1e-3, cw_len_short * 1e-3)],
    out_names=["I", "Q"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %% [markdown]
# ## Analyse et affichage

# %%
moving_integration_width_us = 50

first_idx = 3
n_bins = 500

# Chargement des données
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:, first_idx:-1]
    time_list = file["data/time_us"][first_idx:-1]


# Construction des données avec intégration cumulée et fenêtrée
dt = time_list[1] - time_list[0]
window_width = int(moving_integration_width_us // dt)

data_cum = data_i.cumsum(axis=1) / np.arange(1, len(time_list)+1)

window = np.ones(window_width) / window_width
time_list_mov = np.convolve(time_list, window, mode="valid")
data_mov = np.apply_along_axis(lambda arr: np.convolve(arr, window, mode="valid"), 1, data_i)


# Calcul des histogrammes
hist_mov, bins = np.histogram(data_mov, bins=n_bins)
hist_cum, _ = np.histogram(data_cum, bins=bins)

# Calcul du centre des bins
bins_center = (bins[1:] + bins[:-1]) / 2

# Calcul des histogrammes en fonction du temps de readout
hist_cum_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins)[0], 0, data_cum)

hist_mov_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins)[0], 0, data_mov)

# %% [markdown]
# ### Trouver le temps d'intégration optimal

# %%
plt.imshow(hist_cum_vs_time.T, extent=[*bins[[0, -1]], *time_list[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
# plt.imshow(np.log10(hist_I_vs_time.T), extent=[*bins_i_cum[[0, -1]], *time_list[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
plt.xlabel("Histogram I")
plt.ylabel("Integration time ($\mu$s)")
# plt.xlim(-0.0026, -0.0024)
cb = plt.colorbar(label="Count")

# %%
plt.imshow(hist_mov_vs_time.T, extent=[*bins[[0, -1]], *time_list_mov[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
plt.xlabel("Histogram I")
plt.ylabel("Time of the measurement ($\mu$s)")
# plt.xlim(-0.00605, -0.00575)
cb = plt.colorbar(label="Count")

# %% [markdown]
# #### Tracking des paramètres des gaussiennes en fonction du temps

# %%
# %matplotlib inline
default_time = 50e-6  # Temps d'intégration sur lequel se baser pour trouver les paramètres initiaux de fit
default_time_idx = int(default_time // (dt * 1e-6))  # Index correspondant à ce temps d'intégration
tranche = hist_cum_vs_time[:, default_time_idx]  # Tranche d'histogramme à ce temps d'intégration

res = DistributionSTFitResult.from_bins_hist(bins_center, tranche, compute_visibility=1, p0=None, verbosity=5)

# %%
default_time = 50e-6  # Temps d'intégration sur lequel se baser pour trouver les paramètres initiaux de fit
default_time_idx = int(default_time // (dt * 1e-6))  # Index correspondant à ce temps d'intégration
tranche = hist_cum_vs_time[:, default_time_idx]  # Tranche d'histogramme à ce temps d'intégration

p0 = DistributionSTFitResult.from_bins_hist(bins_center, tranche).popt  # Trouver une seule fois les paramètres optimaux à utiliser pour les fit et les passer à la fonction de fit pour tous les temps d'intégration
all_fits = [
    DistributionSTFitResult.from_bins_hist(bins_center, tranche, compute_visibility=True, p0=p0) for tranche in hist_cum_vs_time.T
]  # Fit des données pour tous les temps d'intégration

# Trouver le temps d'intégration avec la visibilité maximale
optimal_tm_index = max(range(len(all_fits)), key=lambda i: all_fits[i].visibility if all_fits[i].visibility is not None else 0)

optimal_tm = time_list[optimal_tm_index]
optimal_fit = all_fits[optimal_tm_index]

# Afficher les données
print(f"Temps d'intégration optimal: {optimal_tm} us")
print(f"Visibilité: {optimal_fit.visibility*100:.1f}%")
print(f"Threshold: {optimal_fit.threshold} (u.a.)")


# %%
# %matplotlib inline
plt.figure(figsize=(6, 4))
vis = np.array([fit.visibility if fit.visibility is not None else np.nan for fit in all_fits])
plt.plot(time_list, vis, label="Visibility")
plt.scatter(time_list[optimal_tm_index], optimal_fit.visibility, marker="*", c="red", label=f"Optimal visibility: {optimal_fit.visibility*100:.4f}%")
plt.axvline(time_list[optimal_tm_index], c="grey", zorder=-1, linestyle='--')
plt.text(time_list[optimal_tm_index]*1.5, np.nanmin(vis), f"${time_list[optimal_tm_index]}\mu s$", horizontalalignment="left")
plt.xlabel("Integration time (us)")
plt.ylabel("Visibility (%)")
plt.legend()
#plt.xscale('log')
plt.tight_layout()

# %%
# Graph de la fidélité en fonction du threshold, pour le meilleur temps d'intégration
results = [optimal_fit.popt.get_visibility(bins_center, th)
           for th in bins_center]
fid_sin, fid_tri, vis = map(np.array, zip(*results))
plt.figure(figsize=(4,2.5))
plt.plot(bins_center, fid_sin, label="$F_s$")
plt.plot(bins_center, fid_tri, label="$F_t$")
plt.plot(bins_center, vis, label="$V$")
plt.legend()
plt.ylabel("Fidelity/Visibility")
plt.xlabel("Demod bin (a.u.)")
window = 0.004
plt.ylim(np.max(vis)-10*window, 1)
plt.xlim(optimal_fit.threshold*(1-window), optimal_fit.threshold*(1+window))


# %% [markdown]
# # Balayage de la vitesse de rampe: trouver le taux tunnel
# Cette expérience traverse l'anti-croisement à différentes vitesses et trace la les proportions de singulets/triplets préparés.

# %% [markdown]
# ## Code de mesure

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 500

min_ramp_time = 52  # Durée minimale de la rampe (ns)
max_ramp_time = 10_000  # Durée maximale de la rampe (ns)
ramp_time_increment = 520  # Incrément de la durée de la rampe (ns)

############################
ramp_times = np.arange(min_ramp_time, max_ramp_time+1, ramp_time_increment)
n_ramp_times = len(ramp_times)

# Gate space
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_ramp_time:
    n = declare(int)
    ramp_time = declare(int)

    init_st = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()
    bin_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):

        with for_(ramp_time, min_ramp_time, ramp_time<max_ramp_time+1, ramp_time+ramp_time_increment):
            
            reset_if_phase("RF-SET1")
            sequence.add_step(voltage_point_name="init")
            sequence.add_step(voltage_point_name="zero_dc", ramp_duration=10_000, duration=52)
            sequence.add_step(voltage_point_name="load", ramp_duration=ramp_time)

            align("RF-SET1", *gates)
            wait(ramp_time, "RF-SET1")
            sequence.add_step(voltage_point_name="readout", ramp_duration=ramp_time)

            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            
            # save(init_i, init_st)
            save(i, i_st)
            save(q, q_st)
            
            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*10)
            sequence.ramp_to_zero(duration=1)

    with stream_processing():
        i_st.buffer(n_ramp_times).save_all("I")
        q_st.buffer(n_ramp_times).save_all("Q")
        # init_st.buffer(n_ramp_times).save_all("init_readout")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_ramp_time)

filename = expand_filename(path()+"%T_ramp_time.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "ramp_time_ns"],
    ax_values=[n_avg, ramp_times],
    out_names=["I", "Q"],#, "init_readout"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %% [markdown]
# ## Analyse et affichage

# %%
n_bins = 100
# Chargement des données
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    ramp_time_list = file["data/ramp_time_ns"][:]

hist, bins_i = np.histogram(data_i.flatten(), bins=n_bins)
bins_ic = (bins_i[1:] + bins_i[:-1]) / 2
hist_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_i)[0],
    0, data_i
)


# %%
plt.figure()
plt.pcolormesh(ramp_time_list, bins_ic, hist_vs_time)
#plt.axhline(threshold, label="threshold", c="r")
plt.xlabel("Ramp time (ns)")
plt.ylabel("Histogram I")
cb = plt.colorbar(label="Count")
plt.grid(linestyle="--", c='grey', alpha=0.5)


# %%
fit_res = DistributionSTFitResult.from_bins_hist(bins_ic, hist, compute_visibility=True, verbosity=5)
print(f"Threshold: {fit_res.threshold}")
print(f"Visibility: {fit_res.visibility}")

# %%
prob_s = ((data_i < fit_res.threshold).mean(0))
print(prob_s.shape)

# bin_idx = np.argmax(bins_ic > fit_res.threshold)
# prob_s = hist_vs_time[:bin_idx].sum(axis=0) / hist_vs_time.sum(axis=0)

plt.figure(figsize=(6, 4))
plt.plot(ramp_time_list, prob_s*100, marker="o")
plt.xlabel("Ramp time (ns)")
plt.ylabel("Singlet initialisation probability (%)")
# plt.ylim(30)
plt.tight_layout()

# %% [markdown]
# # Balayage du point de loading
# ## Code de mesure

# %%
for ramp_rate in np.arange(100, 15_001, 50, dtype=int):
    # Vitesse de rampe constante
    import config as cfg_file
    importlib.reload(cfg_file)
    from config import qop_ip, cluster_name, config, config_copy
    from params import *
    import params as params_file

    # Paramètres
    n_avg = 500
    n_points = 201
    p2_list = np.linspace(-3e-3, -0.5e-3, n_points)
    p3_list = np.linspace(3e-3, 0.5e-3, n_points)
    ramp_duration_list = (np.abs(p2_list) * (ramp_rate / 1e-3)).astype(int)  # 6us / mV


    ############################
    # Gate space
    sequence = make_gate_sequence(config, gates, operation_points, gain)

    # PSB readout program
    with program() as psb_ramp_time:
        n = declare(int)
        ramp_time = declare(int)
        p2, p3 = declare(fixed), declare(fixed)
        ramp_duration = declare(int)

        init_st = declare_stream()
        i_st = declare_stream()
        q_st = declare_stream()
        bin_st = declare_stream()

        update_frequency("RF-SET1", cw_readout_freq)

        sequence.ramp_to_zero()
        save(cw_amp, 'cw_amp')
        with for_(n, 0, n<n_avg, n+1):
            with for_each_((p2, p3, ramp_duration), (p2_list*gain, p3_list*gain, ramp_duration_list)):
                reset_if_phase("RF-SET1")

                align("RF-SET1", *gates)
                sequence.add_step(voltage_point_name="readout")
                init_i, _ = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)
                
                # sequence.add_step(voltage_point_name="init")
                # sequence.add_step(voltage_point_name="zero_dc", duration=16)
                sequence.add_step(level=[p2, p3], ramp_duration=ramp_duration, duration=52)
                # sequence.add_step(voltage_point_name="load", ramp_duration=6000, duration=ramp_time)
                align("RF-SET1", *gates)
                
                sequence.add_step(voltage_point_name="readout")
                i, q = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)
                
                save(init_i, init_st)
                save(i, i_st)
                save(q, q_st)
                
                sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*10)
                sequence.ramp_to_zero(duration=1)

        with stream_processing():
            i_st.buffer(n_points).save_all("I")
            q_st.buffer(n_points).save_all("Q")
            init_st.buffer(n_points).save_all("init_readout")

    # Exécuter le programme
    qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(psb_ramp_time)

    filename = expand_filename(path()+f"%T_load_point_{ramp_rate}us_per_mV.hdf5")
    with sweep_file(
        filename,
        ax_names=["count", "n_points"],
        ax_values=[n_avg, n_points],
        out_names=["I", "Q", "init_readout"],
        # -- meta:
        p2_list = p2_list,
        p3_list = p3_list,
        ramp_duration_list = ramp_duration_list,
        cell = get_cell_content(), 
        config = config_copy,
        cw_amp = cw_amp,
        params = get_file_code(params_file),
        ramp_rate = ramp_rate,
    ) as f:
        while not f.flush_data(job.result_handles):
            sleep(.1)

# %% [markdown]
# ## Analyse et affichage

# %%
path = r"d:\Intel_Tunel_Falls_12QD_01\data\20260302\load_point_meas"
files = [path + "\\" + file for file in os.listdir(path)]

# %%
n_bins = 100
# Chargement des données
ramp_rates = []
p_s_array = np.empty((len(files), 201))
for i, filename in enumerate(files):
    # print(i)

    with h5py.File(filename, 'r') as file:
        data_init = file["data/init_readout"][:]
        data_i = file["data/I"][:]

        p2_list = file["meta"].attrs["p2_list"]
        p3_list = file["meta"].attrs["p3_list"]
        ramp_duration_list = file["meta"].attrs["ramp_duration_list"]
        ramp_duration =  int(re.search(".*_load_point_(.*)us_per_mV.*", filename).group(1))
        
        point_idxs = file["data/n_points"][:]

    ramp_rates.append(ramp_duration)

    # Calcul des histogrammes
    hist, bins_i = np.histogram(data_i.flatten(), bins=n_bins)
    hist_init, _ = np.histogram(data_init.flatten(), bins=bins_i)

    # Calcul du centre des bins
    bins_ic = (bins_i[1:] + bins_i[:-1]) / 2

    # Calcul des histogrammes en fonction du temps de readout
    hist_vs_time = np.apply_along_axis(
        lambda arr: np.histogram(arr, bins_i)[0],
        0, data_i
    )
    hist_vs_time_init = np.apply_along_axis(
        lambda arr: np.histogram(arr, bins_i)[0],
        0, data_init
    )

    # Faire les fits
    fit_init = fit_distribution_ST(bins_ic, hist_init, compute_visibility=True, debug_plot=0)
    fit_res = fit_distribution_ST(bins_ic, hist, compute_visibility=True, debug_plot=0)
    # fit_res_vec = []
    # for j, tranche in enumerate(hist_vs_time.T):
    #     tranche_fit = fit_distribution_ST(bins_ic, tranche, p0=None, find_threshold=True, debug_plot=False)
    #     if tranche_fit.threshold is None:
    #         print(f"file {i}:{filename}, tranche {j}")
    #     fit_res_vec.append(tranche_fit)

    # Post-selection si les threshold existent
    if (fit_init.threshold is None) or (fit_res.threshold is None):
        p_s_array[i] = np.full((201,), np.nan)
        continue
    data_copy = data_i[:]
    init_thr = fit_init.threshold
    valid_idx = data_init < init_thr
    data_copy[~valid_idx] = 0
    prob_s = ((data_copy < fit_res.threshold) & (data_copy != 0)).sum(axis=0) / (data_copy != 0).sum(axis=0)
    p_s_array[i] = prob_s



# %%
plt.pcolormesh(p2_list, ramp_rates, p_s_array)
plt.xlabel("Load point (on detuning axis)")
plt.ylabel("Ramp rate (ns/uV)")
plt.clim(0.3, 0.5)
cb = plt.colorbar(label="Singlet to triplet probability (2 readout)")
plt.title("Singlet to triplet probability (2 readout)")

# %% [markdown]
# # Sweep 2d: wait/ramp initialisation
# Détermination des paramètres pour l'initialisation

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 500

min_wait_time = 52
max_wait_time = 50_000
wait_time_increment = 2500

min_ramp_time = 16
max_ramp_time = 400
ramp_time_increment = 32

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

wait_times = np.arange(min_wait_time, max_wait_time, wait_time_increment)
ramp_times = np.arange(min_ramp_time, max_ramp_time, ramp_time_increment)

with program() as prgrm:
    wait_time = declare(int)
    ramp_time = declare(int)
    n = declare(int)
    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()
    p2, p3 = declare(fixed), declare(fixed)
    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')

    play(element="Trigger", pulse="step", duration=50)
    play(element="Trigger", pulse="step"*amp(-1), duration=50)

    with for_(n, 0, n<n_avg, n+1):
        with for_(ramp_time, min_ramp_time, ramp_time<max_ramp_time, ramp_time+ramp_time_increment):
            with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):
                reset_if_phase("RF-SET1")

                sequence.add_step(voltage_point_name="load_deep", duration=4_000, ramp_duration=4000)
                align("RF-SET1", *gates)

                sequence.add_step(voltage_point_name="readout")
                i_init, q_init = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )

                sequence.add_step(voltage_point_name="init", duration=wait_time)
                sequence.add_step(voltage_point_name="load_deep", ramp_duration=ramp_time, duration=52)

                align("RF-SET1", *gates)
                sequence.add_step(voltage_point_name="readout")
                i, q = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )

                sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
                sequence.ramp_to_zero(duration=1)
                save(i, i_st)
                save(q, q_st)
                save(i_init, i_st_init)
                save(q_init, q_st_init)

    with stream_processing():
        i_st.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("I")
        q_st.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("Q")
        i_st_init.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("I_init")
        q_st_init.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("Q_init")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

print(filename:=expand_filename(path()+"%T_init_fidelity.hdf5"))
start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "ramp_times", "wait_times"],
    ax_values=[n_avg, ramp_times, wait_times],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file),
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# simulation_config = SimulationConfig(duration=75000)  # In clock cycles
# job = qmm.simulate(config, prgrm, simulation_config)
# samples = job.get_simulated_samples()
# samples.con1.plot()
#waveform_report = job.get_simulated_waveform_report()
# Cast the waveform report to a python dictionary
#waveform_dict = waveform_report.to_dict()
# Visualize and save the waveform report
#waveform_report.create_plot(job.get_simulated_samples(), plot=True)

# %%
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260313\20260313-181533_init_fidelity.hdf5"
# filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260316\20260316-190531_init_fidelity.hdf5"
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    ramp_times = file["data/ramp_times"][:]
    wait_times = file["data/wait_times"][:]

n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2
hists_init = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i.reshape((data_i_init.shape[0], -1)))

# %%
theta = find_iq_rotation(data_i_init, data_q_init, iq_phase, verbosity=2)
theta2 = find_iq_rotation(data_i, data_q, iq_phase, verbosity=2)

# %%
n_slices, resolution = 100, 10
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

fig, axs = plots.hist_2d_with_thresholds(hists_init, bins_i_init, thresholds_init, thresholds)
fig, axs = plots.hist_2d_with_thresholds(hists, bins_i_init, thresholds_init, thresholds)


# %%
# %matplotlib inline
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)
t_to_s = (~singlets_init & singlets).sum(0) / (~singlets_init).sum(0)

fig, axs = plt.subplots(1, 2, sharey=True, width_ratios=[4, 1], figsize=(6, 4))
im = axs[0].imshow(s_to_s*100, 
# vmin=0, vmax=100,
interpolation="none", aspect="auto", extent=mk_extent(wait_times*1e-3, ramp_times), origin="lower")
fig.colorbar(im, label="Singlet probability", ax=axs[0])
axs[0].set(
    title="Init Singlet -> Readout Singlet",
    xlabel="Wait time (us)",
    ylabel="Ramp time (ns)"
)

axs[1].plot(s_to_s.mean(axis=1) * 100, ramp_times)
axs[1].set_xlim(0, 100)
axs[1].set(
    xlabel="Singlet prob (%)"
)
plt.tight_layout()

fig, axs = plt.subplots(1, 2, sharey=True, width_ratios=[4, 1], figsize=(6, 4))
im = axs[0].imshow(t_to_s*100, 
# vmin=0, vmax=100,
interpolation="none", aspect="auto", extent=mk_extent(wait_times*1e-3, ramp_times*1e-3), origin="lower")
fig.colorbar(im, label="Singlet probability", ax=axs[0])
axs[0].set(
    title="Init Triplet -> Readout Singlet",
    xlabel="Wait time (us)",
    ylabel="Ramp time (us)"
)

axs[1].plot(t_to_s.mean(axis=1) * 100, ramp_times*1e-3)
axs[1].set_xlim(0, 100)
axs[1].set(
    xlabel="Singlet prob (%)"
)
plt.tight_layout()

# %% [markdown]
# ### On regarde si la décroissance à des rampes très rapides n'est pas causée par une erreur de readout

# %%
data_tmp_init = data_i_init.transpose([1, 0, 2]).reshape((data_i.shape[1], -1))
data_tmp = data_i.transpose([1, 0, 2]).reshape((data_i.shape[1], -1))

hist_along_ramp_time_init = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins=bins_i_init)[0],
    1, data_tmp_init
)
hist_along_ramp_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins=bins_i_init)[0],
    1, data_tmp
)

plt.imshow(hist_along_ramp_time_init, aspect="auto", interpolation="none", extent=mk_extent(wait_times*1e-3, ramp_times), origin="lower")
plt.colorbar()

plt.figure()
plt.imshow(hist_along_ramp_time, aspect="auto", interpolation="none", extent=mk_extent(wait_times*1e-3, ramp_times), origin="lower")
plt.colorbar()

# %% [markdown]
# ## Mesure d'initialisation par double readout

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 3600 * 1000 * 13
#n_avg = 25_000 * 1000 * 10
buffer_size = 1000

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_readout_time:
    n = declare(int)

    i_st = declare_stream()
    q_st = declare_stream()
    i2_st = declare_stream()
    q2_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        reset_if_phase("RF-SET1")

        sequence.add_step(voltage_point_name="readout")
        i, q = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )
        align("RF-SET1", *gates)

        # sequence.add_step(voltage_point_name="init", ramp_duration=1_440_000//2)
        sequence.add_step(voltage_point_name="init", duration=25_000)
        # sequence.add_step(voltage_point_name="zero_dc", ramp_duration=1_500, duration=0)
        # sequence.add_step(voltage_point_name="load_deep", ramp_duration=2_667_000, duration=0)
        sequence.add_step(voltage_point_name="load_deep", ramp_duration=200, duration=16)
        align("RF-SET1", *gates)
        
        sequence.add_step(voltage_point_name="readout")
        i2, q2 = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )
        align("RF-SET1", *gates)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
        sequence.ramp_to_zero(duration=1)
        save(i, i_st)
        save(q, q_st)
        save(i2, i2_st)
        save(q2, q2_st)
        
        wait(1_000_000 * u.ns)
    
    with stream_processing():
        i_st.buffer(buffer_size).save_all("Iinit")
        q_st.buffer(buffer_size).save_all("Qinit")
        i2_st.buffer(buffer_size).save_all("I")
        q2_st.buffer(buffer_size).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_time)

# filename = expand_filename(path()+"%T_ramp_init.hdf5"); print(filename)
filename = expand_filename(path()+"%T_readout_drift_nuit.hdf5"); print(filename)

from time import time

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "time"],
    ax_values=[n_avg //buffer_size, buffer_size],
    out_names=["I", "Q", "Iinit", "Qinit"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
import matplotlib.dates as mdates
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260226\20260226-194909_readout_drift_nuit.hdf5"
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260227\20260227-161907_readout_drift.hdf5"
# filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260307/20260307-183452_readout_drift.hdf5"
# filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260309/20260309-170250_derive.hdf5"
with h5py.File(filename, 'r') as file:
    data_i_with_nans = file["data/I"][:]#.flatten()
    data_q_with_nans = file["data/Q"][:]
    start_time = file["meta"].attrs["CREATION_TIME"]
    stop_time = file["meta"].attrs["LAST_CALL_TIME"]

    start_dt = datetime.datetime.fromtimestamp(start_time)
    stop_dt  = datetime.datetime.fromtimestamp(stop_time)

valid = ~np.isnan(data_i_with_nans).any(axis=1)
data_i = data_i_with_nans[valid].flatten()
data_q = data_q_with_nans[valid].flatten()
data_r = np.sqrt(data_i**2 + data_q**2)
data_t = np.arctan2(data_q, data_i)

times = np.linspace(
    mdates.date2num(start_dt),
    mdates.date2num(stop_dt),
    len(data_i)
)
arrays = (data_r, data_t, data_i, data_q, times)

# %%
# %matplotlib inline
fig, ax = plt.subplots()
ax.plot_date(times, data_i, ls="none", marker=".")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel("Temps de mesure")
plt.ylabel("I")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# %%
# reshape et calcul des histogrammes
r, t, i, q, time = map(lambda arr: np.reshape(arr, (-1, 1000)), arrays)
def fn(array):
    _, bins = np.histogram(array, bins=100)
    return np.apply_along_axis(lambda line: np.histogram(line, bins=bins)[0], 1, array), bins
hist_r, hist_t, hist_i, hist_q = map(fn, (r, t, i, q))
t_axis = time[:,0] - time[:,0][0]

# %%
fig, axs  = plt.subplots(1, 2, figsize=(10, 5))
for (hist, bins), ax in zip((hist_r, hist_t), axs):
    extent = [bins[0], bins[-1], t_axis[0], t_axis[-1]]
    ax.imshow(hist, origin="lower", aspect="auto", interpolation="none", extent=extent)
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Demod value")
    ax.set_ylabel("Time (h)")
    ax.grid(c="grey", alpha=.2, linestyle="--")

ax1, ax2 = axs
ax1.set_title("R")
ax2.set_title("Theta")
fig.tight_layout()


# %%
max_t = hist_t[0].argmax(axis=1)
max_t = max_t - np.mean(max_t)

N = len(max_t)
dt = (t_axis[-1] - t_axis[0]) / N
fft_freqs = rfftfreq(N, dt*24)
print(f"Fréquence d'échantillonnage: {1/dt}")

amps = np.abs(rfft(max_t))
plt.figure(figsize=(5,4))
plt.plot(fft_freqs, amps)
plt.xlabel("Fréquence (1/h)")
plt.ylabel("Amplitude")
max_amp = fft_freqs[np.argmax(amps)]
plt.axvline(max_amp, c="red", ls=":", label=f"T={np.round(1/max_amp,2)} h")
plt.xlim(-.1, 2)
plt.legend()
plt.show()


# %%
n_slices, resolution = 100, 10
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

fig, axs = plots.hist_2d_with_thresholds(hists_init, bins_i_init, thresholds_init, thresholds)
fig, axs = plots.hist_2d_with_thresholds(hists, bins_i_init, thresholds_init, thresholds)


# %%
# %matplotlib inline
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)
t_to_s = (~singlets_init & singlets).sum(0) / (~singlets_init).sum(0)

fig, axs = plt.subplots(1, 2, sharey=True, width_ratios=[4, 1], figsize=(6, 4))
im = axs[0].imshow(s_to_s*100, 
# vmin=0, vmax=100,
interpolation="none", aspect="auto", extent=mk_extent(wait_times*1e-3, ramp_times), origin="lower")
fig.colorbar(im, label="Singlet probability", ax=axs[0])
axs[0].set(
    title="Init Singlet -> Readout Singlet",
    xlabel="Wait time (us)",
    ylabel="Ramp time (ns)"
)

axs[1].plot(s_to_s.mean(axis=1) * 100, ramp_times)
axs[1].set_xlim(0, 100)
axs[1].set(
    xlabel="Singlet prob (%)"
)
plt.tight_layout()

fig, axs = plt.subplots(1, 2, sharey=True, width_ratios=[4, 1], figsize=(6, 4))
im = axs[0].imshow(t_to_s*100, 
# vmin=0, vmax=100,
interpolation="none", aspect="auto", extent=mk_extent(wait_times*1e-3, ramp_times*1e-3), origin="lower")
fig.colorbar(im, label="Singlet probability", ax=axs[0])
axs[0].set(
    title="Init Triplet -> Readout Singlet",
    xlabel="Wait time (us)",
    ylabel="Ramp time (us)"
)

axs[1].plot(t_to_s.mean(axis=1) * 100, ramp_times*1e-3)
axs[1].set_xlim(0, 100)
axs[1].set(
    xlabel="Singlet prob (%)"
)
plt.tight_layout()

# %% [markdown]
# ### On regarde si la décroissance à des rampes très rapides n'est pas causée par une erreur de readout

# %%
data_tmp_init = data_i_init.transpose([1, 0, 2]).reshape((data_i.shape[1], -1))
data_tmp = data_i.transpose([1, 0, 2]).reshape((data_i.shape[1], -1))

hist_along_ramp_time_init = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins=bins_i_init)[0],
    1, data_tmp_init
)
hist_along_ramp_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins=bins_i_init)[0],
    1, data_tmp
)

plt.imshow(hist_along_ramp_time_init, aspect="auto", interpolation="none", extent=mk_extent(wait_times*1e-3, ramp_times), origin="lower")
plt.colorbar()

plt.figure()
plt.imshow(hist_along_ramp_time, aspect="auto", interpolation="none", extent=mk_extent(wait_times*1e-3, ramp_times), origin="lower")
plt.colorbar()

# %% [markdown]
# # Sweep 1d: temps de load (ex. osc. Rabi)
# Oscillation d'échange:
# caractériser le temps d'attente dans la zone (3, 1)

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 500

min_wait_time = 60
max_wait_time = 100_000
wait_time_increment = 400

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

wait_list = np.arange(min_wait_time, max_wait_time, wait_time_increment)

# PSB readout program
with program() as prgrm:
    wait_time = declare(int)
    n = declare(int)

    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):
            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="load_deep", duration=2_000, ramp_duration=2000)
            align("RF-SET1", *gates)

            sequence.add_step(voltage_point_name="readout")
            i_init, q_init = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )

            sequence.add_step(voltage_point_name="init", ramp_duration=16, duration=1000)
            sequence.add_step(voltage_point_name="load_deep", ramp_duration=100, duration=wait_time)
            
            align("RF-SET1", *gates)
            sequence.add_step(voltage_point_name="readout")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )

            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
            sequence.ramp_to_zero(duration=1)
            save(i, i_st)
            save(q, q_st)
            save(i_init, i_st_init)
            save(q_init, q_st_init)

    with stream_processing():
        i_st.buffer(len(wait_list)).save_all("I")
        q_st.buffer(len(wait_list)).save_all("Q")
        i_st_init.buffer(len(wait_list)).save_all("I_init")
        q_st_init.buffer(len(wait_list)).save_all("Q_init")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

filename = expand_filename(path()+"%T_deep_load_wait.hdf5"); print(filename)

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "wait_time"],
    ax_values=[n_avg, wait_list],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
# filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260311\20260311-183431_deep_load_wait.hdf5"
# filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260313\20260313-131204_deep_load_wait.hdf5"
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    wait_times = file["data/wait_time"][:]
    start = file["meta"].attrs["CREATION_TIME"]
    stop = file["meta"].attrs["LAST_CALL_TIME"]

n_avg = data_i.shape[0]
meas_time = np.linspace(0, stop-start, n_avg)
wait_times_us = 1e-3 * wait_times

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 50, 50
thresholds_init = fit_slices(data_i_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_i, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])


# %%
# %matplotlib inline
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init)
extent = [*bins_i_init_c[[0, -1]], 0, len(hists)]
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=extent)
plt.plot(thresholds_init, np.arange(n_avg), c="y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), c="r", label="Threshold readout")
plt.xlabel("Histogram I")
plt.ylabel("Index")
plt.legend()
cb = plt.colorbar(label="Count")

plt.figure()
hists2 = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i)
plt.imshow(hists2, aspect="auto", origin="lower", interpolation="none", extent=extent)
cb = plt.colorbar(label="Count")


# %%
# Calcul de la proportion de singulets après la mesure
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)

# # Fit des oscillations de Rabi
# popt, _ = curve_fit(oscillations_rabi, wait_times_us, s_to_s, p0=(1, 10, 1/5, 0))
# Ps0, T2, f0, ramp_time_correction = popt
# rabi_period = 1 / f0

# fitted_rabi = oscillations_rabi(wait_times_us, *popt)
# half_pi_pulse_time = rabi_period / 4
# pi_pulse_time = rabi_period / 2
# half_pi_pulse_time_first = half_pi_pulse_time - ramp_time_correction
# pi_pulse_time_first = pi_pulse_time - ramp_time_correction

# print("Résultat du fit:")
# print(f"  Ps0: {popt[0]*100:.1f} %")
# print(f"  T2: {popt[1]:.2f} us")
# print(f"  f0: {popt[2]:.2f} MHz  ({rabi_period:.2f} us)")
# print(f"  ramp_time_correction: {popt[3]*1e3:.2f} ns")

# print()
# print("Temps des pulses:")
# print(f"  pi/2: {half_pi_pulse_time*1e3:.0f} ns")
# print(f"  pi: {pi_pulse_time*1e3:.0f} ns")

plt.figure(figsize=(10,2))
plt.plot(wait_times_us, s_to_s, label="Data")
# plt.plot(wait_times_us, fitted_rabi, label=f"Fit Rabi\nT2 = {popt[1]:.2f} us")
plt.xlabel("Load wait time (us)")
plt.ylabel("S to S probability")
plt.legend(loc="upper right")
plt.grid(alpha=0.5, ls="--")

# %% [markdown]
# ## Analyse de la stabilité du Rabi dans le temps

# %%
n_times_averaged = 500
init_idx = np.arange(0, n_avg-n_times_averaged, n_times_averaged)

# Calcul de la proportion de singulets après la mesure
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = np.array([(singlets_init[i:i+n_times_averaged] & singlets[i:i+n_times_averaged]).sum(0) / singlets_init[i:i+n_times_averaged].sum(0) for i in init_idx])

popt = np.array([curve_fit(oscillations_rabi, wait_times_us, data, p0=(1, 10, 1/5, 0))[0] for data in s_to_s])


# %%
# %matplotlib inline

n_times_averaged = 500
init_idx = np.arange(0, n_avg-n_times_averaged, n_times_averaged)
meas_time_axis = meas_time[init_idx + n_times_averaged // 2]

# Calcul de la proportion de singulets après la mesure
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = np.array([(singlets_init[i:i+n_times_averaged] & singlets[i:i+n_times_averaged]).sum(0) / singlets_init[i:i+n_times_averaged].sum(0) for i in init_idx])

# Fit des oscillations de Rabi
popt = np.empty((len(s_to_s), 4))
for i, data in enumerate(s_to_s):
    try:
        p0 = (0.75, 10, 1/4, 0.1)
        pmin = (0, 0, 0, 0)
        pmax = (1, np.inf, np.inf, 0.5)
        popt[i] = np.array(curve_fit(oscillations_rabi, wait_times_us, data, p0=p0, bounds=(pmin, pmax))[0])
    except:
        popt[i] = [np.nan] * 4

Ps0, T2, f0, ramp_time_correction = popt.T
rabi_period = 1 / f0

first_min = rabi_period / 2 - ramp_time_correction

fitted_rabi = np.array([oscillations_rabi(wait_times_us, *p) for p in popt])
half_pi_pulse_time = rabi_period / 4
pi_pulse_time = rabi_period / 2
half_pi_pulse_time_first = half_pi_pulse_time - ramp_time_correction
pi_pulse_time_first = pi_pulse_time - ramp_time_correction

fig, axs = plt.subplots(1, 5, sharey=True, figsize=(10, 4), width_ratios=[4, 1, 1, 1, 1])
im0 = axs[0].imshow(s_to_s*100, aspect="auto", extent=[*wait_times_us[[0, -1]], *meas_time[[0, init_idx[-1]+500]]], origin="lower", interpolation="none")
axs[0].plot(first_min, meas_time_axis, c="r", label="Pi-pulse wait time")
axs[0].set_xlabel("Load wait time (us)")
axs[0].set_ylabel("Time of the measurement (s)")
axs[0].set_title("Rabi oscillations over time")
axs[0].legend()
fig.colorbar(im0, ax=axs[0], label="S to S Probability (%)")

axs[1].plot(Ps0*100, meas_time_axis)
axs[1].set_xlabel("Ps0 (%)")

axs[2].plot(T2, meas_time_axis)
axs[2].set_xlabel("T2 (us)")

axs[3].plot(rabi_period, meas_time_axis)
axs[3].set_xlabel("Rabi period (us)")

axs[4].plot(ramp_time_correction*1e3, meas_time_axis)
axs[4].set_xlabel("Initial time\ncorrection (ns)")
fig.tight_layout()


fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
im0 = axs[0].imshow(s_to_s*100, aspect="auto", extent=[*wait_times_us[[0, -1]], *meas_time[[0, init_idx[-1]+500]]], origin="lower", interpolation="none")
axs[0].set_xlabel("Load wait time (us)")
axs[0].set_ylabel("Time of the measurement (s)")
axs[0].set_title("Rabi oscillations over time")
fig.colorbar(im0, ax=axs[0], label="S to S Probability (%)")

im1 = axs[1].imshow(fitted_rabi*100, aspect="auto", extent=[*wait_times_us[[0, -1]], *meas_time[[0, init_idx[-1]+500]]], origin="lower", interpolation="none")
axs[1].set_xlabel("Load wait time (us)")
axs[1].set_ylabel("Time of the measurement (s)")
axs[1].set_title("Reconstruction from the fits")
fig.colorbar(im1, ax=axs[1], label="S to S Probability (%)")

fig.tight_layout()

# %% [markdown]
# # Sweep 1d: temps de load 2 (ramsey)
# Les oscillations de Rabi (mesure d'avant) donne le temps d'attente optimale pour préparer un état cohérent superposé.
# Ici, on essai des oscillations de ramsey, pour trouver le temps de décohérence de cette état.

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 200

wait_time_for_50_50 = 1_252

min_wait_time = 60
max_wait_time = 20_000
wait_time_increment = 60

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

wait_list = np.arange(min_wait_time, max_wait_time, wait_time_increment)

# PSB readout program
with program() as prgrm:
    ramp_time = declare(int)
    wait_time = declare(int)
    n = declare(int)

    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):


        with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):

            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="readout")
            i_init, q_init = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)


            sequence.add_step(voltage_point_name="init", duration=100_000)
            sequence.add_step(level=pi_over_2_level, ramp_duration=200, duration=pi_over_2_duration)
            sequence.add_step(voltage_point_name="load", ramp_duration=16, duration=wait_time)
            sequence.add_step(level=pi_over_2_level, ramp_duration=16, duration=pi_over_2_duration)
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="readout")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)

            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
            sequence.ramp_to_zero(duration=1)
            save(i, i_st)
            save(q, q_st)
            save(i_init, i_st_init)
            save(q_init, q_st_init)

    with stream_processing():
        i_st.buffer(len(wait_list)).save_all("I")
        q_st.buffer(len(wait_list)).save_all("Q")
        i_st_init.buffer(len(wait_list)).save_all("I_init")
        q_st_init.buffer(len(wait_list)).save_all("Q_init")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

filename = expand_filename(path()+"%T_Ramsey.hdf5"); print(filename)

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "wait_time"],
    ax_values=[n_avg, wait_list],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    wait_times = file["data/wait_time"][:]

n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 50, 10
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

# %%
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
extent = [*bins_i_init_c[[0, -1]], 0, len(hists)]
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=extent)
plt.plot(thresholds_init, np.arange(n_avg), c="y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), c="r", label="Threshold readout")
plt.xlabel("Histogram I")
plt.ylabel("Index")
plt.legend()
cb = plt.colorbar(label="Count")

# %%
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)

# plt.figure(figsize=(10,2))
# plt.imshow(singlets, origin="lower", aspect="auto", interpolation="none", extent=[*wait_times[[0,-1]], 0, len(singlets)])
# plt.colorbar()
# plt.xlabel("Load wait time"); plt.ylabel("count")

plt.figure(figsize=(10,2))
plt.plot(wait_times*1e-3, s_to_s)
plt.xlabel("Load wait time (us)")
plt.ylabel("S to S probability")
plt.grid(alpha=0.5, ls="--")
plt.title("Après attente au point d'échange (ramsey)")
# plt.xlim(xmin=0, xmax=2)

# %% [markdown]
# # Sweep 2d: tps de load / pt de load (rabi 2d)

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 1000

min_wait_time = 60
max_wait_time = 25_000
wait_time_increment = 120

p2_list = np.linspace(-0.000, -0.009, 101)
p3_list = -p2_list

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)
wait_list = np.arange(min_wait_time, max_wait_time, wait_time_increment)

with program() as prgrm:
    wait_time = declare(int)
    n = declare(int)
    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()
    p2, p3 = declare(fixed), declare(fixed)
    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')

    with for_each_((p2, p3), (p2_list*gain, p3_list*gain)):
        with for_(n, 0, n<n_avg, n+1):
            with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):

                reset_if_phase("RF-SET1")

                sequence.add_step(voltage_point_name="readout")
                i_init, q_init = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)


                sequence.add_step(voltage_point_name="init", duration=100_000)
                sequence.add_step(level=[p2, p3], ramp_duration=200, duration=wait_time)
                align("RF-SET1", *gates)
                
                sequence.add_step(voltage_point_name="readout")
                i, q = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)

                sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
                sequence.ramp_to_zero(duration=1)
                save(i, i_st)
                save(q, q_st)
                save(i_init, i_st_init)
                save(q_init, q_st_init)

    with stream_processing():
        streams = [i_st, q_st, i_st_init, q_st_init]
        save_names = ["I", "Q", "I_init", "Q_init"]
        [stream
            .buffer(len(wait_list))
            # .buffer(len(p2_list))
            .buffer(n_avg)
            .save_all(name) for stream, name in zip(streams, save_names)
        ]

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

filename = expand_filename(path()+"%T_load_point_wait_time.hdf5"); print(filename)
start_time = time()
with sweep_file(
    filename,
    ax_names=["detuning_points_idx", "avg", "wait_times"],
    ax_values=[len(p2_list), n_avg, wait_list],
    out_names=save_names,
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file), p2_list=p2_list, p3_list=p3_list,
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
detuning_ax = 0
avg_ax = 1
wait_ax = 2
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    wait_times = file["data/wait_times"][:]
    p2_list = file["meta"].attrs["p2_list"]
    p3_list = file["meta"].attrs["p3_list"]

n_points = len(p2_list)
n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 1, 1
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds, thresholds_init = [
    fit_slices(data, long_axis=0, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
    for data in (data_2d, data_2d_init)
]

# %%
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=mk_extent(bins_i_init_c, len(hists)))
plt.plot(thresholds_init, np.arange(n_avg), "y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), "r", label="Threshold readout")
plt.xlabel("Histogram I"); plt.ylabel("Index")
plt.legend(); plt.colorbar(label="Count")

# %%
# %matplotlib inline
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(avg_ax) / singlets_init.sum(avg_ax)
plt.figure()
plt.imshow(s_to_s, interpolation="none", aspect="auto", extent=mk_extent(wait_times, p2_list, center=True))
plt.colorbar()
plt.gca().set(
    title="Init Singlet -> Readout Singlet (Rabi)",
    xlabel="Wait time (ns)",
    ylabel="Detuning load point (V)")
# plt.gca().set_yticks(wait_times[::2]);

# %% [markdown]
# # Sweep 2d: vitesse de rampe / temps de load

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 500

min_wait_time = 60
max_wait_time = 2_000
wait_time_increment = 120

min_ramp_time = 100 # min 52
max_ramp_time = 150
ramp_time_increment = 1

wait_level = [-0.004*gain, 0.004*gain]

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

wait_times = np.arange(min_wait_time, max_wait_time, wait_time_increment)
ramp_times = np.arange(min_ramp_time, max_ramp_time, ramp_time_increment)

with program() as prgrm:
    wait_time = declare(int)
    ramp_time = declare(int)
    n = declare(int)
    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()
    p2, p3 = declare(fixed), declare(fixed)
    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):

        with for_(ramp_time, min_ramp_time, ramp_time<max_ramp_time, ramp_time+ramp_time_increment):
            
            with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):

                reset_if_phase("RF-SET1")

                sequence.add_step(voltage_point_name="readout")
                i_init, q_init = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)


                sequence.add_step(voltage_point_name="init", duration=100_000)
                sequence.add_step(level=wait_level, ramp_duration=ramp_time, duration=wait_time)
                align("RF-SET1", *gates)
                
                sequence.add_step(voltage_point_name="readout")
                i, q = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)

                sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
                sequence.ramp_to_zero(duration=1)
                save(i, i_st)
                save(q, q_st)
                save(i_init, i_st_init)
                save(q_init, q_st_init)

    with stream_processing():
        i_st.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("I")
        q_st.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("Q")
        i_st_init.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("I_init")
        q_st_init.buffer(len(wait_times)).buffer(len(ramp_times)).save_all("Q_init")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

print(filename:=expand_filename(path()+"%T_ramp_time_wait_time.hdf5"))
start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "ramp_times", "wait_times"],
    ax_values=[n_avg, ramp_times, wait_times],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file),
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# simulation_config = SimulationConfig(duration=10_000)  # In clock cycles
# job = qmm.simulate(config, prgrm, simulation_config)

# %%
samples = job.get_simulated_samples()
waveform_report = job.get_simulated_waveform_report()
# Cast the waveform report to a python dictionary
waveform_dict = waveform_report.to_dict()
# Visualize and save the waveform report
waveform_report.create_plot(samples, plot=True)

# %%
#filename = r"D:/Intel_Tunel_Falls_12QD_01/data\20260312\20260312-120720_ramp_time_wait_time.hdf5"
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    ramp_times = file["data/ramp_times"][:]
    wait_times = file["data/wait_times"][:]

n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 10, 10
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

# %%
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=mk_extent(bins_i_init_c, len(hists)))
plt.plot(thresholds_init, np.arange(n_avg), "y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), "r", label="Threshold readout")
plt.xlabel("Histogram I"); plt.ylabel("Index")
plt.legend(); plt.colorbar(label="Count")

# %%
import matplotlib.ticker as ticker
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)
plt.figure()
plt.imshow(s_to_s, origin="lower", interpolation="none", aspect="auto", extent=mk_extent(wait_times, ramp_times, center=True))
plt.colorbar()
plt.gca().set(
    title="Init Singlet -> Readout Singlet",
    xlabel="Wait time at load (ns)",
    ylabel="Ramp time to load (V)")
plt.gca().set_yticks(ramp_times[::2]);
#plt.xlim(60,4000)
#plt.ylim(top=200)

# %% [markdown]
# # Sweep 1d: temps de load 2 (ramsey)
# Les oscillations de Rabi (mesure d'avant) donne le temps d'attente optimale pour préparer un état cohérent superposé.
# Ici, on essai des oscillations de ramsey, pour trouver le temps de décohérence de cette état.

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 200

wait_time_for_50_50 = 1_252

min_wait_time = 60
max_wait_time = 20_000
wait_time_increment = 60

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

wait_list = np.arange(min_wait_time, max_wait_time, wait_time_increment)

# PSB readout program
with program() as prgrm:
    ramp_time = declare(int)
    wait_time = declare(int)
    n = declare(int)

    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):


        with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):

            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="readout")
            i_init, q_init = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)


            sequence.add_step(voltage_point_name="init", duration=100_000)
            sequence.add_step(level=pi_over_2_level, ramp_duration=200, duration=pi_over_2_duration)
            sequence.add_step(voltage_point_name="load", ramp_duration=16, duration=wait_time)
            sequence.add_step(level=pi_over_2_level, ramp_duration=16, duration=pi_over_2_duration)
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="readout")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)

            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
            sequence.ramp_to_zero(duration=1)
            save(i, i_st)
            save(q, q_st)
            save(i_init, i_st_init)
            save(q_init, q_st_init)

    with stream_processing():
        i_st.buffer(len(wait_list)).save_all("I")
        q_st.buffer(len(wait_list)).save_all("Q")
        i_st_init.buffer(len(wait_list)).save_all("I_init")
        q_st_init.buffer(len(wait_list)).save_all("Q_init")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

filename = expand_filename(path()+"%T_Ramsey.hdf5"); print(filename)

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "wait_time"],
    ax_values=[n_avg, wait_list],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    wait_times = file["data/wait_time"][:]

n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 50, 10
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

# %%
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
extent = [*bins_i_init_c[[0, -1]], 0, len(hists)]
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=extent)
plt.plot(thresholds_init, np.arange(n_avg), c="y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), c="r", label="Threshold readout")
plt.xlabel("Histogram I")
plt.ylabel("Index")
plt.legend()
cb = plt.colorbar(label="Count")

# %%
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)

# plt.figure(figsize=(10,2))
# plt.imshow(singlets, origin="lower", aspect="auto", interpolation="none", extent=[*wait_times[[0,-1]], 0, len(singlets)])
# plt.colorbar()
# plt.xlabel("Load wait time"); plt.ylabel("count")

plt.figure(figsize=(10,2))
plt.plot(wait_times*1e-3, s_to_s)
plt.xlabel("Load wait time (us)")
plt.ylabel("S to S probability")
plt.grid(alpha=0.5, ls="--")
plt.title("Après attente au point d'échange (ramsey)")
# plt.xlim(xmin=0, xmax=2)

# %% [markdown]
# # Sweep 2d: tps d'échange / pt d'échange (ramsey 2d)

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 500

min_wait_time = 52
max_wait_time = 1_500
wait_time_increment = 8

p2_list = np.linspace(-0.0015, 0, 201)
p3_list = -p2_list

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)
wait_list = np.arange(min_wait_time, max_wait_time, wait_time_increment)

with program() as prgrm:
    wait_time = declare(int)
    n = declare(int)
    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()
    p2, p3 = declare(fixed), declare(fixed)
    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_each_((p2, p3), (p2_list*gain, p3_list*gain)):
        with for_(n, 0, n<n_avg, n+1):

            with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):

                reset_if_phase("RF-SET1")

                sequence.add_step(voltage_point_name="readout")
                i_init, q_init = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)
                
                sequence.add_step(voltage_point_name="init", duration=100_000)
                sequence.add_step(level=pi_over_2_level, ramp_duration=200, duration=52)
                sequence.add_step(level=[p2, p3], ramp_duration=16, duration=wait_time)
                sequence.add_step(level=pi_over_2_level, ramp_duration=16, duration=52)
                align("RF-SET1", *gates)
                
                sequence.add_step(voltage_point_name="readout")
                i, q = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out",
                    amplitude=cw_amp*10,
                )
                align("RF-SET1", *gates)

                sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
                sequence.ramp_to_zero(duration=1)
                save(i, i_st)
                save(q, q_st)
                save(i_init, i_st_init)
                save(q_init, q_st_init)

    with stream_processing():
        streams = [i_st, q_st, i_st_init, q_st_init]
        save_names = ["I", "Q", "I_init", "Q_init"]
        [stream
            .buffer(len(wait_list))
            # .buffer(len(p2_list))
            .buffer(n_avg)
            .save_all(name) for stream, name in zip(streams, save_names)]

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

filename = expand_filename(path()+"%T_exchange_point_wait_time.hdf5"); print(filename)
start_time = time()
with sweep_file(
    filename,
    ax_names=["exchange_points_idx", "avg", "wait_times"],
    ax_values=[len(p2_list), n_avg, wait_list],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file), p2_list=p2_list, p3_list=p3_list,
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    wait_times = file["data/wait_times"][:]
    p2_list = file["meta"].attrs["p2_list"]
    p3_list = file["meta"].attrs["p3_list"]
    avg_ax = list(file["data"].attrs["sweeped_ax_names"]).index("avg")

n_points = len(p2_list)
n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 1, 1
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

# %%
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=mk_extent(bins_i_init_c, len(hists)))
plt.plot(thresholds_init, np.arange(n_avg), "y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), "r", label="Threshold readout")
plt.xlabel("Histogram I"); plt.ylabel("Index")
plt.legend(); plt.colorbar(label="Count")

# %%
# %matplotlib inline
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(avg_ax) / singlets_init.sum(avg_ax)
plt.figure()
plt.imshow(s_to_s, origin="lower", interpolation="none", aspect="auto", extent=mk_extent(wait_times, p2_list, center=True))
plt.colorbar()
plt.gca().set(
    title="Init Singlet -> Readout Singlet (Ramsey)",
    xlabel="Wait time (ns)",
    ylabel="Detuning load point (V)");

# %%
# %matplotlib inline
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(avg_ax) / singlets_init.sum(avg_ax)
plt.figure()
plt.imshow(s_to_s, origin="lower", interpolation="none", aspect="auto", extent=mk_extent(wait_times, p2_list, center=True))
plt.colorbar()
plt.gca().set(
    title="Init Singlet -> Readout Singlet (Ramsey)",
    xlabel="Wait time (ns)",
    ylabel="Detuning load point (V)");

# %%
# %matplotlib inline
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(avg_ax) / singlets_init.sum(avg_ax)
plt.figure()
plt.imshow(s_to_s, origin="lower", interpolation="none", aspect="auto", extent=mk_extent(wait_times, p2_list, center=True))
plt.colorbar()
plt.gca().set(
    title="Init Singlet -> Readout Singlet (Ramsey)",
    xlabel="Wait time (ns)",
    ylabel="Detuning load point (V)");

# %% [markdown]
# # Séquence spin echo

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 500

half_pi_pulse_time = 1536
pi_pulse_time = 3072

min_wait_time = 60
max_wait_time = 10_000
wait_time_increment = 60

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

wait_list = np.arange(min_wait_time, max_wait_time, wait_time_increment)

# PSB readout program
with program() as prgrm:
    ramp_time = declare(int)
    wait_time = declare(int)
    n = declare(int)

    i_st_init = declare_stream()
    q_st_init = declare_stream()
    i_st = declare_stream()
    q_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):


        with for_(wait_time, min_wait_time, wait_time<max_wait_time, wait_time+wait_time_increment):

            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="readout")
            i_init, q_init = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)


            sequence.add_step(voltage_point_name="init", duration=100_000)
            sequence.add_step(voltage_point_name="load_deep", ramp_duration=200, duration=half_pi_pulse_time)
            sequence.add_step(voltage_point_name="load", ramp_duration=16, duration=wait_time)
            sequence.add_step(voltage_point_name="load_deep", ramp_duration=16, duration=pi_pulse_time)
            sequence.add_step(voltage_point_name="load", ramp_duration=16, duration=wait_time)
            sequence.add_step(voltage_point_name="load_deep", ramp_duration=16, duration=half_pi_pulse_time)
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="readout")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)

            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
            sequence.ramp_to_zero(duration=1)
            save(i, i_st)
            save(q, q_st)
            save(i_init, i_st_init)
            save(q_init, q_st_init)

    with stream_processing():
        i_st.buffer(len(wait_list)).save_all("I")
        q_st.buffer(len(wait_list)).save_all("Q")
        i_st_init.buffer(len(wait_list)).save_all("I_init")
        q_st_init.buffer(len(wait_list)).save_all("Q_init")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(prgrm)

filename = expand_filename(path()+"%T_spin_echo.hdf5"); print(filename)

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "wait_time"],
    ax_values=[n_avg, wait_list],
    out_names=["I", "Q", "I_init", "Q_init"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    data_q = file["data/Q"][:]
    data_i_init = file["data/I_init"][:]
    data_q_init = file["data/Q_init"][:]
    wait_times = file["data/wait_time"][:]

n_avg = data_i.shape[0]

bins_i_init = np.histogram(data_i_init.flatten(), bins=101)[1]
bins_i_init_c = (bins_i_init[1:] + bins_i_init[:-1]) / 2

# %%
n_slices, resolution = 50, 10
data_2d_init = data_i_init.reshape((n_avg, -1))
data_2d = data_i.reshape((n_avg, -1))
thresholds_init = fit_slices(data_2d_init, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])
thresholds = fit_slices(data_2d, n_slices=n_slices, resolution=resolution, bins=bins_i_init, arg_names=["threshold"])

# %%
hists = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins_i_init)[0], 1, data_i_init.reshape((data_i_init.shape[0], -1)))
extent = [*bins_i_init_c[[0, -1]], 0, len(hists)]
plt.imshow(hists, aspect="auto", origin="lower", interpolation="none", extent=extent)
plt.plot(thresholds_init, np.arange(n_avg), c="y", label="Threshold init")
plt.plot(thresholds, np.arange(n_avg), c="r", label="Threshold readout")
plt.xlabel("Histogram I")
plt.ylabel("Index")
plt.legend()
cb = plt.colorbar(label="Count")

# %%
singlets_init = (data_i_init.T < thresholds_init.squeeze()).T
singlets =  (data_i.T < thresholds_init.squeeze()).T
s_to_s = (singlets_init & singlets).sum(0) / singlets_init.sum(0)

# plt.figure(figsize=(10,2))
# plt.imshow(singlets, origin="lower", aspect="auto", interpolation="none", extent=[*wait_times[[0,-1]], 0, len(singlets)])
# plt.colorbar()
# plt.xlabel("Load wait time"); plt.ylabel("count")

plt.figure(figsize=(10,2))
plt.plot(wait_times*1e-3, s_to_s)
plt.xlabel("Load wait time (us)")
plt.ylabel("S to S probability")
plt.grid(alpha=0.5, ls="--")
plt.title("Après attente au point d'échange - pulse pi/2 - attente au point d'échange")

# %% [markdown]
# # Mesures Complémentaires

# %% [markdown]
# ## Mesure d'initialisation par double readout

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 3600 * 1000 * 13
#n_avg = 25_000 * 1000 * 10
buffer_size = 1000

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_readout_time:
    n = declare(int)

    i_st = declare_stream()
    q_st = declare_stream()
    i2_st = declare_stream()
    q2_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        reset_if_phase("RF-SET1")

        sequence.add_step(voltage_point_name="readout")
        i, q = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )
        align("RF-SET1", *gates)

        # sequence.add_step(voltage_point_name="init", ramp_duration=1_440_000//2)
        sequence.add_step(voltage_point_name="init", duration=25_000)
        # sequence.add_step(voltage_point_name="zero_dc", ramp_duration=1_500, duration=0)
        # sequence.add_step(voltage_point_name="load_deep", ramp_duration=2_667_000, duration=0)
        sequence.add_step(voltage_point_name="load_deep", ramp_duration=200, duration=16)
        align("RF-SET1", *gates)
        
        sequence.add_step(voltage_point_name="readout")
        i2, q2 = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )
        align("RF-SET1", *gates)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
        sequence.ramp_to_zero(duration=1)
        save(i, i_st)
        save(q, q_st)
        save(i2, i2_st)
        save(q2, q2_st)
        
        wait(1_000_000 * u.ns)
    
    with stream_processing():
        i_st.buffer(buffer_size).save_all("Iinit")
        q_st.buffer(buffer_size).save_all("Qinit")
        i2_st.buffer(buffer_size).save_all("I")
        q2_st.buffer(buffer_size).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_time)

# filename = expand_filename(path()+"%T_ramp_init.hdf5"); print(filename)
filename = expand_filename(path()+"%T_readout_drift_nuit.hdf5"); print(filename)

from time import time

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "time"],
    ax_values=[n_avg //buffer_size, buffer_size],
    out_names=["I", "Q", "Iinit", "Qinit"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
import matplotlib.dates as mdates
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260226\20260226-194909_readout_drift_nuit.hdf5"
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260227\20260227-161907_readout_drift.hdf5"
# filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260307/20260307-183452_readout_drift.hdf5"
# filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260309/20260309-170250_derive.hdf5"
with h5py.File(filename, 'r') as file:
    data_i_with_nans = file["data/I"][:]#.flatten()
    data_q_with_nans = file["data/Q"][:]
    start_time = file["meta"].attrs["CREATION_TIME"]
    stop_time = file["meta"].attrs["LAST_CALL_TIME"]

    start_dt = datetime.datetime.fromtimestamp(start_time)
    stop_dt  = datetime.datetime.fromtimestamp(stop_time)

valid = ~np.isnan(data_i_with_nans).any(axis=1)
data_i = data_i_with_nans[valid].flatten()
data_q = data_q_with_nans[valid].flatten()
data_r = np.sqrt(data_i**2 + data_q**2)
data_t = np.arctan2(data_q, data_i)

times = np.linspace(
    mdates.date2num(start_dt),
    mdates.date2num(stop_dt),
    len(data_i)
)
arrays = (data_r, data_t, data_i, data_q, times)

# %%
# %matplotlib inline
fig, ax = plt.subplots()
ax.plot_date(times, data_i, ls="none", marker=".")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel("Temps de mesure")
plt.ylabel("I")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# %%
# reshape et calcul des histogrammes
r, t, i, q, time = map(lambda arr: np.reshape(arr, (-1, 1000)), arrays)
def fn(array):
    _, bins = np.histogram(array, bins=100)
    return np.apply_along_axis(lambda line: np.histogram(line, bins=bins)[0], 1, array), bins
hist_r, hist_t, hist_i, hist_q = map(fn, (r, t, i, q))
t_axis = time[:,0] - time[:,0][0]

# %%
fig, axs  = plt.subplots(1, 2, figsize=(10, 5))
for (hist, bins), ax in zip((hist_r, hist_t), axs):
    extent = [bins[0], bins[-1], t_axis[0], t_axis[-1]]
    ax.imshow(hist, origin="lower", aspect="auto", interpolation="none", extent=extent)
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Demod value")
    ax.set_ylabel("Time (h)")
    ax.grid(c="grey", alpha=.2, linestyle="--")

ax1, ax2 = axs
ax1.set_title("R")
ax2.set_title("Theta")
fig.tight_layout()


# %%
max_t = hist_t[0].argmax(axis=1)
max_t = max_t - np.mean(max_t)

N = len(max_t)
dt = (t_axis[-1] - t_axis[0]) / N
fft_freqs = rfftfreq(N, dt*24)
print(f"Fréquence d'échantillonnage: {1/dt}")

amps = np.abs(rfft(max_t))
plt.figure(figsize=(5,4))
plt.plot(fft_freqs, amps)
plt.xlabel("Fréquence (1/h)")
plt.ylabel("Amplitude")
max_amp = fft_freqs[np.argmax(amps)]
plt.axvline(max_amp, c="red", ls=":", label=f"T={np.round(1/max_amp,2)} h")
plt.xlim(-.1, 2)
plt.legend()
plt.show()


# %% [markdown]
# ## Mesure d'initialisation par double readout et init

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 3600 * 1000 * 13
#n_avg = 25_000 * 1000 * 10
buffer_size = 1000

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_readout_time:
    n = declare(int)

    i_st = declare_stream()
    q_st = declare_stream()
    i2_st = declare_stream()
    q2_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        reset_if_phase("RF-SET1")

        sequence.add_step(voltage_point_name="readout")
        i, q = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )

        sequence.add_step(voltage_point_name="init")
        sequence.add_step(voltage_point_name="zero_dc", ramp_duration=1_500, duration=0)
        sequence.add_step(voltage_point_name="load_deep", duration=5000)
        align("RF-SET1", *gates)
        
        sequence.add_step(voltage_point_name="readout")
        i2, q2 = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )
        align("RF-SET1", *gates)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
        sequence.ramp_to_zero(duration=1)
        save(i, i_st)
        save(q, q_st)
        save(i2, i2_st)
        save(q2, q2_st)
        
        wait(1_000_000 * u.ns)
    
    with stream_processing():
        i_st.buffer(buffer_size).save_all("Iinit")
        q_st.buffer(buffer_size).save_all("Qinit")
        i2_st.buffer(buffer_size).save_all("I")
        q2_st.buffer(buffer_size).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_time)

# filename = expand_filename(path()+"%T_ramp_init.hdf5"); print(filename)
filename = expand_filename(path()+"%T_readout_drift_nuit.hdf5"); print(filename)

from time import time

start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "time"],
    ax_values=[n_avg //buffer_size, buffer_size],
    out_names=["I", "Q", "Iinit", "Qinit"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
import matplotlib.dates as mdates
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260226\20260226-194909_readout_drift_nuit.hdf5"
#filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260227\20260227-161907_readout_drift.hdf5"
# filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260307/20260307-183452_readout_drift.hdf5"
# filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260309/20260309-170250_derive.hdf5"
with h5py.File(filename, 'r') as file:
    data_i_with_nans = file["data/I"][:]#.flatten()
    data_q_with_nans = file["data/Q"][:]
    start_time = file["meta"].attrs["CREATION_TIME"]
    stop_time = file["meta"].attrs["LAST_CALL_TIME"]

    start_dt = datetime.datetime.fromtimestamp(start_time)
    stop_dt  = datetime.datetime.fromtimestamp(stop_time)

valid = ~np.isnan(data_i_with_nans).any(axis=1)
data_i = data_i_with_nans[valid].flatten()
data_q = data_q_with_nans[valid].flatten()
data_r = np.sqrt(data_i**2 + data_q**2)
data_t = np.arctan2(data_q, data_i)

times = np.linspace(
    mdates.date2num(start_dt),
    mdates.date2num(stop_dt),
    len(data_i)
)
arrays = (data_r, data_t, data_i, data_q, times)

# %%
# %matplotlib inline
fig, ax = plt.subplots()
ax.plot_date(times, data_i, ls="none", marker=".")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel("Temps de mesure")
plt.ylabel("I")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# %%
# reshape et calcul des histogrammes
r, t, i, q, time = map(lambda arr: np.reshape(arr, (-1, 1000)), arrays)
def fn(array):
    _, bins = np.histogram(array, bins=100)
    return np.apply_along_axis(lambda line: np.histogram(line, bins=bins)[0], 1, array), bins
hist_r, hist_t, hist_i, hist_q = map(fn, (r, t, i, q))
t_axis = time[:,0] - time[:,0][0]

# %%
fig, axs  = plt.subplots(1, 2, figsize=(10, 5))
for (hist, bins), ax in zip((hist_r, hist_t), axs):
    extent = [bins[0], bins[-1], t_axis[0], t_axis[-1]]
    ax.imshow(hist, origin="lower", aspect="auto", interpolation="none", extent=extent)
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Demod value")
    ax.set_ylabel("Time (h)")
    ax.grid(c="grey", alpha=.2, linestyle="--")

ax1, ax2 = axs
ax1.set_title("R")
ax2.set_title("Theta")
fig.tight_layout()


# %%
max_t = hist_t[0].argmax(axis=1)
max_t = max_t - np.mean(max_t)

N = len(max_t)
dt = (t_axis[-1] - t_axis[0]) / N
fft_freqs = rfftfreq(N, dt*24)
print(f"Fréquence d'échantillonnage: {1/dt}")

amps = np.abs(rfft(max_t))
plt.figure(figsize=(5,4))
plt.plot(fft_freqs, amps)
plt.xlabel("Fréquence (1/h)")
plt.ylabel("Amplitude")
max_amp = fft_freqs[np.argmax(amps)]
plt.axvline(max_amp, c="red", ls=":", label=f"T={np.round(1/max_amp,2)} h")
plt.xlim(-.1, 2)
plt.legend()
plt.show()


# %% [markdown]
# ## Mesure avec post-selection

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 3600 * 1000 * 16
#n_avg = 25_000 * 1000 * 10
# n_avg = 1000 * 10
buffer_size = 1000

############################
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_readout_time:
    n = declare(int)

    i_st = declare_stream()
    q_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        reset_if_phase("RF-SET1")

        # sequence.add_step(voltage_point_name="init", ramp_duration=1_440_000//2)
        sequence.add_step(voltage_point_name="init", duration=25_000)
        # sequence.add_step(voltage_point_name="zero_dc", ramp_duration=1_500, duration=0)
        # sequence.add_step(voltage_point_name="load_deep", ramp_duration=2_667_000, duration=0)
        sequence.add_step(voltage_point_name="load_deep", ramp_duration=200, duration=16)
        align("RF-SET1", *gates)
        
        sequence.add_step(voltage_point_name="readout")
        i, q = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out",
            amplitude=cw_amp*10,
        )
        align("RF-SET1", *gates)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*gain)
        sequence.ramp_to_zero(duration=1)
        save(i, i_st)
        save(q, q_st)
        wait(1_000_000*u.ns)

    with stream_processing():
        i_st.buffer(buffer_size).save_all("I")
        q_st.buffer(buffer_size).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_time)

filename = expand_filename(path()+"%T_derive.hdf5"); print(filename)
from time import time
start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "time"],
    ax_values=[n_avg //buffer_size, buffer_size],
    out_names=["I", "Q"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)
stop_time = time()

# %%
filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260309/20260309-162825_ramp_init.hdf5"

# %%
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:].flatten()
    data_q = file["data/Q"][:].flatten()
    data_iinit = file["data/Iinit"][:].flatten()
    data_qinit = file["data/Qinit"][:].flatten()
    start_time = file["meta"].attrs["CREATION_TIME"]
    stop_time = file["meta"].attrs["LAST_CALL_TIME"]

# %%
hist_init, bins = np.histogram(data_iinit, bins=101)
hist, _ = np.histogram(data_i, bins=bins)

bins_c = (bins[1:] + bins[:-1]) / 2

fit_init = DistributionSTFitResult.from_bins_hist(bins_c, hist_init, compute_visibility=True, verbosity=6)
fit = DistributionSTFitResult.from_bins_hist(bins_c, hist, compute_visibility=True, verbosity=6)

# %%
s_to_s, t_to_s = [], []
thr_list = np.linspace(bins[0], bins[-1], 101)

for thr in thr_list:
    data_singlet_init = (data_iinit <= thr).astype(int)
    data_triplet_init = 1 - data_singlet_init
    data_singlet = (data_i <= fit.threshold).astype(int)
    data_triplet = 1 - data_singlet
    ss = data_singlet_init & data_singlet
    tt = data_triplet_init & data_singlet

    prop_ss = ss.sum() / data_singlet_init.sum()
    prop_ts = tt.sum() / data_triplet_init.sum()
    t_to_s.append(prop_ts)
    s_to_s.append(prop_ss)

plt.scatter(thr_list, s_to_s, label="Singlet to singlet")
plt.scatter(thr_list, t_to_s, label="Triplet to singlet")
plt.axvline(fit_init.threshold, linestyle="--", c="red", label="Computed init thr")
plt.legend()
plt.grid(linestyle="--", alpha=.5)
plt.xlabel("Init threshold")
# plt.ylim(0.25, 0.60)

# %% [markdown]
# ## analyse readout dans le temps

# %%
filename = r"D:/Intel_Tunel_Falls_12QD_01/data/20260310/20260310-203630_readout_drift_nuit.hdf5"
with h5py.File(filename, 'r') as file:
    data_i_with_nans = file["data/I"][:]#.flatten()
    data_q_with_nans = file["data/Q"][:]
    start_time = file["meta"].attrs["CREATION_TIME"]
    stop_time = file["meta"].attrs["LAST_CALL_TIME"]

    start_dt = datetime.datetime.fromtimestamp(start_time)
    stop_dt  = datetime.datetime.fromtimestamp(stop_time)

valid = ~np.isnan(data_i_with_nans).any(axis=1)
data_i = data_i_with_nans[valid].flatten()
data_q = data_q_with_nans[valid].flatten()
data_r = np.sqrt(data_i**2 + data_q**2)
data_t = np.arctan2(data_q, data_i)

times = np.linspace(
    mdates.date2num(start_dt),
    mdates.date2num(stop_dt),
    len(data_i)
)
arrays = (data_r, data_t, data_i, data_q, times)

# %%
# %matplotlib inline
fig, ax = plt.subplots()
ax.plot_date(times, data_i, ls="none", marker=".")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel("Temps de mesure")
plt.ylabel("I")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
