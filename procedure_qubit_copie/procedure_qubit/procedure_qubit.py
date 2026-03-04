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
import numpy as np
from time import sleep
import datetime
import h5py
from copy import deepcopy
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.special import erf, erfc
from scipy import integrate
import importlib
import re
import os



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
)
from qm import QuantumMachinesManager

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
    sweep_file
)
from utils.future import (
    get_file_variables,
    get_file_code,
    fit_distribution_ST,
    fit_distribution_ST_batch,
)



# %%
data_path = "D:/Intel_Tunel_Falls_12QD_01/data"
path = make_path_fn(data_path)

# %% [markdown]
# # Video mode
# Mettre (0, 0) au point d'initialisation.

# %%
# %gui qt

import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config
from videomode_lib.videomode import VideoModeWindow, Sweep
from params import *


short_axis = Sweep.from_nbpts(4e-3, -4e-3, 101, "P3", attenuation_db, interlace=0)
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
# # Trouver le point de readout

# %% [markdown]
# ## Code de mesure

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Parameters
n_avg = 500
n_detuning = 201

p2_list = np.linspace(-1.5e-3, -0.e-3, n_detuning)
p3_list = np.linspace(1.5e-3, 0.e-3, n_detuning)

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
            sequence.add_step(voltage_point_name="init")
            sequence.add_step(voltage_point_name="zero_dc", duration=16)
            sequence.add_step(voltage_point_name="load", ramp_duration=10_000)
            align("RF-SET1", *gates)
            
            sequence.add_step(level=[p2, p3], duration=cw_len)
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
hist_i = hist.sum(axis=1)
hist_q = hist.sum(axis=0)

# Calcul du centre des bins
bins_ic = (bins_i[1:] + bins_i[:-1]) / 2
bins_qc = (bins_q[1:] + bins_q[:-1]) / 2

# %% [markdown]
# ### Trouver l'angle du plan IQ
# Ce calcul trouve l'angle tel que toute l'information de la mesure sur trouve sur la quadrature I, et la gaussienne avec la moyenne la plus faible correspond à la configuration (4, 0) / singulet.

# %%
# %matplotlib inline
pente, _ = np.polyfit(data_i.flatten(), data_q.flatten(), 1)
i_tri, q_tri = data_i[:, 0].mean(), data_q[:, 0].mean()
i_sin, q_sin = data_i[:, -1].mean(), data_q[:, -1].mean()
delta_i = i_tri - i_sin
delta_q = q_tri - q_sin
correction = np.arctan2(np.sign(delta_q) * pente, np.sign(delta_i))
theta = iq_phase + correction
print(f"Angle de rotation du plan IQ: {theta} rad (correction de {correction} rad)")

x_centre, y_centre = data_i.mean(), data_q.mean()
B = y_centre - pente * x_centre
plt.plot(bins_i, pente*bins_i+B, c="red", label="Droite de correction IQ")

plt.imshow(hist.T, extent=[*bins_i[[0, -1]], *bins_q[[0, -1]]], interpolation="none", aspect="equal", origin="lower")
plt.xlabel("Histogram I")
plt.ylabel("Histogram Q")
cb = plt.colorbar()
cb.set_label("Count")

plt.figure()
plt.plot(bins_ic, hist_i)
plt.xlabel("Histogram I")
plt.ylabel("Count")

# %% [markdown]
# ### Trouver le point de détuning optimal

# %%
detuning_idx = 10

hist_vs_detuning = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_i)[0],
    0, data_i
)

plt.figure(figsize=(6, 4))
plt.title("Loading point: (P2, P3)=(-2mV, 2mV)")
plt.imshow(hist_vs_detuning, extent=[0, n_detuning, *bins_i[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
plt.xlabel("Detuning index")
plt.ylabel("Histogram I")
cb = plt.colorbar(label="Count")
plt.tight_layout()
# plt.savefig("Readout_displacement_2mV.png", dpi=1200)

if detuning_idx:
    #plt.axvline(detuning_idx, c="red", ls=":")
    plt.text(0.5, 0.9, f"Temps d'intégration: {cw_len*1e-3}us", c="white", horizontalalignment='center',
     verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    print(f"Couple de détuning: (P2, P3) = ({round(p2_list[detuning_idx], 5)}, {round(p3_list[detuning_idx], 5)})")
    plt.savefig(rf"R:\Student\Alexis\mesures\2026-03-04-splitting_vs_temps_int\{int(cw_len*1e-3)}us_{n_avg}moy.svg",)

    plt.figure()
    plt.plot(bins_ic, hist_vs_detuning[:, detuning_idx])
    plt.xlabel("Detuning index")
    plt.ylabel("Histogram I count")

# %% [markdown]
# # Trouver le temps de readout

# %% [markdown]
# ## Code de mesure

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_slices = 100
n_cores = 2
n_avg = 100_000


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
        sequence.add_step(voltage_point_name="load")
        align(*rf_sets, *gates)
        
        sequence.add_step(voltage_point_name="readout", duration=n_slices * cw_len_short)
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
n_bins = 400

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
hist_cum, bins_cum = np.histogram(data_cum, bins=n_bins)
hist_mov, bins_mov = np.histogram(data_mov, bins=n_bins)

# Calcul du centre des bins
bins_cum_center = (bins_cum[1:] + bins_cum[:-1]) / 2
bins_mov_center = (bins_mov[1:] + bins_mov[:-1]) / 2

# Calcul des histogrammes en fonction du temps de readout
hist_cum_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_cum)[0],
    0, data_cum
)

hist_mov_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_mov)[0],
    0, data_mov
)

# %% [markdown]
# ### Trouver le temps d'intégration optimal

# %%
plt.imshow(hist_cum_vs_time.T, extent=[*bins_cum[[0, -1]], *time_list[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
# plt.imshow(np.log10(hist_I_vs_time.T), extent=[*bins_i_cum[[0, -1]], *time_list[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
plt.xlabel("Histogram I")
plt.ylabel("Integration time ($\mu$s)")
# plt.xlim(-0.0026, -0.0024)
cb = plt.colorbar(label="Count")

# %%
plt.imshow(hist_mov_vs_time.T, extent=[*bins_mov[[0, -1]], *time_list_mov[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
plt.xlabel("Histogram I")
plt.ylabel("Time of the measurement ($\mu$s)")
# plt.xlim(-0.00605, -0.00575)
cb = plt.colorbar(label="Count")

# %% [markdown]
# #### Tracking des paramètres des gaussiennes en fonction du temps

# %%
# %matplotlib inline
default_time = 30e-6  # Temps d'intégration sur lequel se baser pour trouver les paramètres initiaux de fit
default_time_idx = int(default_time // (dt * 1e-6))  # Index correspondant à ce temps d'intégration
tranche = hist_cum_vs_time[:, default_time_idx]  # Tranche d'histogramme à ce temps d'intégration

res = fit_distribution_ST(bins_cum_center, tranche, compute_visibility=1, p0=None, debug_plot=1)
print(res.visibility)

# %%
default_time = 75e-6  # Temps d'intégration sur lequel se baser pour trouver les paramètres initiaux de fit
default_time_idx = int(default_time // (dt * 1e-6))  # Index correspondant à ce temps d'intégration
tranche = hist_cum_vs_time[:, default_time_idx]  # Tranche d'histogramme à ce temps d'intégration

p0 = fit_distribution_ST(bins_cum_center, tranche).popt  # Trouver une seule fois les paramètres optimaux à utiliser pour les fit et les passer à la fonction de fit pour tous les temps d'intégration
all_fits = [fit_distribution_ST(bins_cum_center, tranche, compute_visibility=True, p0=p0) for tranche in hist_cum_vs_time.T]  # Fit des données pour tous les temps d'intégration

# Trouver le temps d'intégration avec la visibilité maximale
optimal_tm_index = max(range(len(all_fits)), key=lambda i: all_fits[i].visibility if all_fits[i].visibility is not None else 0)
optimal_tm = time_list[optimal_tm_index]
optimal_fit = all_fits[optimal_tm_index]

# Afficher les données
print(f"Temps d'intégration optimal: {optimal_tm} us")
print(f"Visibilité: {optimal_fit.visibility*100:.1f}%")
print(f"Threshold: {optimal_fit.threshold} (u.a.)")

plt.figure(figsize=(6, 4))
plt.plot(time_list, [fit.visibility  for fit in all_fits], label="Visibility")
plt.scatter(time_list[optimal_tm_index], optimal_fit.visibility, marker="*", c="red", label="Optimal visibility")
plt.xlabel("Integration time (us)")
plt.ylabel("Visibility (%)")
plt.legend()
plt.tight_layout()

# %% [markdown]
# # Trouver le taux tunnel
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
n_avg = 1000

min_ramp_time = 52  # Durée minimale de la rampe (ns)
max_ramp_time = 20_000  # Durée maximale de la rampe (ns)
ramp_time_increment = 200  # Incrément de la durée de la rampe (ns)

use_short_wait_times = False
# Activer ce paramètres pour utiliser des temps de rampe inférieurs à 52ns.
# Le temps de compilation peut monter à près d'une minute.

############################
if use_short_wait_times:
    fast_ramp_times = np.arange(4, 48+1, 4)
    slow_ramp_times = np.arange(min_ramp_time, max_ramp_time+1, ramp_time_increment)
    ramp_times = np.hstack([fast_ramp_times, slow_ramp_times])
else:
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
        # Utiliser des variables python pour des temps de rampe inférieurs à 52ns
        if use_short_wait_times:
            for ramp_time_py in fast_ramp_times:
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
                sequence.add_step(voltage_point_name="zero_dc", duration=200)
                # sequence.add_step(voltage_point_name="load", ramp_duration=ramp_time, duration=0)
                sequence.add_step(voltage_point_name="load", ramp_duration=6000, duration=0)
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

        # Utiliser des variables qua pour accélérer le temps de compilation
        with for_(ramp_time, min_ramp_time, ramp_time<max_ramp_time+1, ramp_time+ramp_time_increment):
            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="readout")
            init_i, _ = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="load", ramp_duration=ramp_time, duration=6000)
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
        i_st.buffer(n_ramp_times).save_all("I")
        q_st.buffer(n_ramp_times).save_all("Q")
        init_st.buffer(n_ramp_times).save_all("init_readout")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_ramp_time)

filename = expand_filename(path()+"%T_ramp_time.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "ramp_time_ns"],
    ax_values=[n_avg, ramp_times],
    out_names=["I", "Q", "init_readout"],
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
    data_init = file["data/init_readout"][:]
    ramp_time_list = file["data/ramp_time_ns"][:]

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


# %%
plt.pcolormesh(ramp_time_list, bins_ic, hist_vs_time_init)
#plt.axhline(threshold, label="threshold", c="r")
plt.xlabel("Ramp time (ns)")
plt.ylabel("Histogram I init")
cb = plt.colorbar(label="Count")

plt.figure()
plt.pcolormesh(ramp_time_list, bins_ic, hist_vs_time)
#plt.axhline(threshold, label="threshold", c="r")
plt.xlabel("Ramp time (ns)")
plt.ylabel("Histogram I")
cb = plt.colorbar(label="Count")


# %%
fit_init = fit_distribution_ST(bins_ic, hist_init, compute_visibility=True, debug_plot=True)

fit_res = fit_distribution_ST(bins_ic, hist, compute_visibility=True, debug_plot=True)
print(f"Threshold: {fit_res.threshold}")
print(f"Visibility: {fit_res.visibility}")

# %%
data_copy = data_i[:]
init_thr = fit_init.threshold
valid_idx = data_init < init_thr
data_copy[~valid_idx] = 0
prob_s = ((data_copy < fit_res.threshold) & (data_copy != 0)).sum(axis=0) / (data_copy != 0).sum(axis=0)

# bin_idx = np.argmax(bins_ic > fit_res.threshold)
# prob_s = hist_vs_time[:bin_idx].sum(axis=0) / hist_vs_time.sum(axis=0)

plt.figure(figsize=(6, 4))
plt.plot(ramp_time_list, prob_s*100, marker="o")
plt.xlabel("Ramp time (ns)")
plt.ylabel("Singlet initialisation probability (%)")
# plt.ylim(30)
plt.tight_layout()

# %% [markdown]
# # Varier le point de loading
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
    n_avg = 2000
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
                # sequence.add_step(voltage_point_name="zero_dc", duration=2000)
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
# # Varier le temps de loading

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 100
load_time_min, load_time_max = 2_200_000, 3_000_000
load_time_incr = 10000
load_times = np.arange(load_time_min, load_time_max+1, load_time_incr)[::-1]
# Gate space
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as psb_ramp_time:
    n = declare(int)
    load_time = declare(int)

    st_I_readout_1 = declare_stream()
    st_I_readout_2 = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        with for_(load_time, load_time_min, load_time<load_time_max+1, load_time+load_time_incr):
            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="readout")
            r1, _ = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="load", duration=load_time)
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="readout")
            r2, _ = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            
            save(r1, st_I_readout_1)
            save(r2, st_I_readout_2)
            
            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*10)
            sequence.ramp_to_zero(duration=1)

    with stream_processing():
        st_I_readout_1.buffer(len(load_times)).save_all("readout1")
        st_I_readout_2.buffer(len(load_times)).save_all("readout2")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_ramp_time)

filename = expand_filename(path()+"%T_load_time.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "load_time"],
    ax_values=[n_avg, load_times],
    out_names=["readout1", "readout2"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %%
n_bins = 100
# Chargement des données
with h5py.File(filename, 'r') as file:
    data_readout1 = file["data/readout1"][:]
    data_readout2 = file["data/readout2"][:]
    load_times = file["data/load_time"][:]

# Calcul des histogrammes
hist1, bins_i = np.histogram(data_readout1.flatten(), bins=n_bins)
hist2, _ = np.histogram(data_readout2.flatten(), bins=bins_i)

# Calcul du centre des bins
bins_ic = (bins_i[1:] + bins_i[:-1]) / 2

# Calcul des histogrammes en fonction du temps de readout
hist1_vs_time = np.apply_along_axis( lambda arr: np.histogram(arr, bins_i)[0],0, data_readout1)
hist2_vs_time = np.apply_along_axis(lambda arr: np.histogram(arr, bins_i)[0],0, data_readout2)

# %%
plt.pcolormesh(load_times, bins_ic, hist1_vs_time)
#plt.axhline(threshold, label="threshold", c="r")
plt.xlabel("Load time (ns)")
plt.ylabel("Histogram readout 1")
cb = plt.colorbar(label="Count")

plt.figure()
plt.pcolormesh(load_times, bins_ic, hist2_vs_time)
#plt.axhline(threshold, label="threshold", c="r")
plt.xlabel("Load time (ns)")
plt.ylabel("Histogram readout 2")
cb = plt.colorbar(label="Count")


# %%
fit_r1 = fit_distribution_ST(bins_ic, hist1, compute_visibility=True)
fit_r2 = fit_distribution_ST(bins_ic, hist2, compute_visibility=True)
fit_res_vec = [fit_distribution_ST(bins_ic, h, p0=fit_res.popt, find_threshold=True) for h in hist2_vs_time.T]

data_copy = data_readout2[:]
init_thr = fit_r1.threshold

valid_idx = data_readout1 < fit_r1.threshold
data_copy[~valid_idx] = 0
prob_s = ((data_copy < fit_r2.threshold) & (data_copy != 0)).sum(axis=0) / (data_copy != 0).sum(axis=0)

# bin_idx = np.argmax(bins_ic > fit_res.threshold)
# prob_s = hist_vs_time[:bin_idx].sum(axis=0) / hist_vs_time.sum(axis=0)

plt.figure(figsize=(6, 4))
plt.plot(load_times, prob_s*100, marker="o")
plt.xlabel("Ramp time (ns)")
plt.ylabel("Singlet initialisation probability (%)")
# plt.ylim(30)
plt.tight_layout()

# %% [markdown]
# # Caractériser le temps d'initialisation
# Cette expérience cherche à déterminer le temps de vie de l'état Triplet dans la zone d'initialisation (4, 0).
# Pour ce faire, on fait la séquence de pulse suivante:
# - Init pendant 10ms
# - Rampe de 50us jusqu'en (3, 1) pour préparer un état aléatoire
# - Retour rapide en initialisation
# - Attente de temps variable
# - Readout

# %%
from config import config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 200
ramp_time = 50_000  # Temps de rampe pour aller en (3, 1) [ns]

# Paramètres pour générer les temps d'attente dans la zone (4, 0)
# wait_times = np.arange(0, max_delay+0.1, delay_increment)
max_delay = 5_000_000  # ns
delay_increment = 500_000  # ns


############################
wait_times = np.arange(0, max_delay+4, delay_increment)
n_wait_times = len(wait_times)

# Gate space
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as init_wait_time:
    n = declare(int)
    wait_time = declare(int)
    singulet_bin = declare(bool)

    i_st = declare_stream()
    q_st = declare_stream()
    bin_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        with for_(wait_time, 0, wait_time<max_delay+1, wait_time+delay_increment):
            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="init")
            sequence.add_step(voltage_point_name="load", ramp_duration=ramp_time)
            sequence.add_step(voltage_point_name="init", duration=wait_time)
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="readout")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            save(i, i_st)
            save(q, q_st)
            assign(singulet_bin, i < threshold)
            save(singulet_bin, bin_st)
            
            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*10)
            sequence.ramp_to_zero(duration=1)

    with stream_processing():
        i_st.buffer(n_wait_times).save_all("I")
        q_st.buffer(n_wait_times).save_all("Q")
        bin_st.boolean_to_int().buffer(n_wait_times).average().save("singlet_probability")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(init_wait_time)

filename = expand_filename(path()+"%T_init_time.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "wait_time_ns"],
    ax_values=[n_avg, wait_times],
    out_names=["I", "Q"],
    avg_names=["singlet_probability"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    threshold = threshold,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %%
n_bins = 40

# Chargement des données
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    prob_s = file["data/singlet_probability"][:]
    wait_time_list = file["data/wait_time_ns"][:]

# Calcul des histogrammes
hist, bins_i = np.histogram(data_i.flatten(), bins=n_bins)

# Calcul du centre des bins
bins_ic = (bins_i[1:] + bins_i[:-1]) / 2

# Calcul des histogrammes en fonction du temps de readout
hist_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_i)[0],
    0, data_i
)

# %%
plt.figure(figsize=(6, 4))
plt.plot(wait_time_list*1e-3, prob_s*100)
plt.xlabel("Initialisation wait time (ns)")
plt.ylabel("Singlet initialisation probability (%)")
plt.tight_layout()

# %% [markdown]
# # Caractériser le temps d'attente dans la zone (3, 1)
# Cette expérience cherche à déterminer l'impact du temps d'attente dans la zone (3, 1).
# Pour ce faire, on fait la séquence de pulse suivante:
# - Init pendant 10ms
# - Pulse instantanné en (3, 1) pour préparer un état Singulet, puis attente de temps variable
# - Readout

# %%
from config import config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 20

# Temps d'attente dans la zone (3, 1)
# wait_times = np.arange(0, max_delay+0.1, delay_increment)
max_delay = 200  # ns
delay_increment = 4  # ns


############################
wait_times = np.arange(0, max_delay+4, delay_increment)
n_wait_times = len(wait_times)

# Gate space
sequence = make_gate_sequence(config, gates, operation_points, gain)

# PSB readout program
with program() as init_wait_time:
    n = declare(int)
    wait_time = declare(int)
    singulet_bin = declare(bool)

    i_st = declare_stream()
    q_st = declare_stream()
    bin_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)

    sequence.ramp_to_zero()
    save(cw_amp, 'cw_amp')
    with for_(n, 0, n<n_avg, n+1):
        with for_(wait_time, 0, wait_time<max_delay+1, wait_time+delay_increment):
            reset_if_phase("RF-SET1")

            sequence.add_step(voltage_point_name="init")
            sequence.add_step(voltage_point_name="load", duration=wait_time)
            align("RF-SET1", *gates)
            
            sequence.add_step(voltage_point_name="readout")
            i, q = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out",
                amplitude=cw_amp*10,
            )
            align("RF-SET1", *gates)
            save(i, i_st)
            save(q, q_st)
            assign(singulet_bin, i < threshold)
            save(singulet_bin, bin_st)
            
            sequence.add_compensation_pulse(max_amplitude=max_compensation_amp*10)
            sequence.ramp_to_zero(duration=1)

    with stream_processing():
        i_st.buffer(n_wait_times).save_all("I")
        q_st.buffer(n_wait_times).save_all("Q")
        bin_st.boolean_to_int().buffer(n_wait_times).average().save("singlet_probability")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(init_wait_time)

filename = expand_filename(path()+"%T_loading_wait_time.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "wait_time_ns"],
    ax_values=[n_avg, wait_times],
    out_names=["I", "Q"],
    avg_names=["singlet_probability"],
    # -- meta:
    cell = get_cell_content(), 
    config = config_copy,
    cw_amp = cw_amp,
    threshold = threshold,
    params = get_file_code(params_file)
) as f:
    while not f.flush_data(job.result_handles):
        sleep(.1)

# %%
n_bins = 40

# Chargement des données
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:]
    prob_s = file["data/singlet_probability"][:]
    wait_time_list = file["data/wait_time_ns"][:]

# Calcul des histogrammes
hist, bins_i = np.histogram(data_i.flatten(), bins=n_bins)

# Calcul du centre des bins
bins_ic = (bins_i[1:] + bins_i[:-1]) / 2

# Calcul des histogrammes en fonction du temps de readout
hist_vs_time = np.apply_along_axis(
    lambda arr: np.histogram(arr, bins_i)[0],
    0, data_i
)

# %%
plt.imshow(hist_vs_time, extent=[*wait_time_list[[0, -1]], *bins_i[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
plt.xlabel("(3, 1) wait time (ns)")
plt.ylabel("Histogram I")
cb = plt.colorbar(label="Count")

# %% [markdown]
# # Mesures Complémentaires

# %% [markdown]
# ## Dérive de la quadrature I

# %%
import config as cfg_file
importlib.reload(cfg_file)
from config import qop_ip, cluster_name, config, config_copy
from params import *
import params as params_file

# Paramètres
n_avg = 3600 * 1000 * 56
# n_avg = 1 * 1000 * 10


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

        # sequence.add_step(voltage_point_name="init", duration=100)
        sequence.add_step(voltage_point_name="load")
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
    with stream_processing():
        i_st.buffer(1000).save_all("I")
        q_st.buffer(1000).save_all("Q")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_readout_time)

filename = expand_filename(path()+"%T_readout_drift.hdf5"); print(filename)
from time import time
start_time = time()
with sweep_file(
    filename,
    ax_names=["count", "time_us"],
    ax_values=[n_avg //1000, 1000],
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
import matplotlib.dates as mdates
filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260226\20260226-194909_readout_drift_nuit.hdf5"
filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260227\20260227-161907_readout_drift.hdf5"
with h5py.File(filename, 'r') as file:
    data_i = file["data/I"][:].flatten()
    start_time = file["meta"].attrs["CREATION_TIME"]
    stop_time = file["meta"].attrs["LAST_CALL_TIME"]

    start_dt = datetime.datetime.fromtimestamp(start_time)
    stop_dt  = datetime.datetime.fromtimestamp(stop_time)
    
times = np.linspace(
    mdates.date2num(start_dt),
    mdates.date2num(stop_dt),
    len(data_i)
)

# %%
print(mdates.date2num(start_dt))#, stop_dt)
print(start_dt)

# %%
# %matplotlib inline
# exec: 37s
fig, ax = plt.subplots()
ax.plot_date(times, data_i, ls="none", marker=".")
# ax.plot_date(times, data_i, '-')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel("Temps de mesure")
plt.ylabel("I")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# %%
# Compute all the fits (Long exec: ~1m30)
reshape_size = (-1, 1000)
hist, bins = np.histogram(data_i, bins=50)
bins_c = (bins[1:] + bins[:-1]) / 2
hist_vs_time = np.apply_along_axis(
    lambda data: np.histogram(data, bins=bins)[0],
    1, data_i.reshape(reshape_size)
)
time_axis = times.reshape(reshape_size)[:,0]



# %%
# %matplotlib inline
fig, ax = plt.subplots()
plt.imshow(hist_vs_time, extent=[*bins[[0, -1]], *times[[0, -1]]], interpolation="none", aspect="auto", origin="lower")
ax.yaxis_date()
ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.yaxis.set_major_locator(mdates.AutoDateLocator())
plt.xlabel("Histogram I")
plt.ylabel("Measure time")
cb = plt.colorbar(label="count")


# %%

p0 = (.9, 1, -0.443, -0.433, .01)
# all_fits = [
#    fit_distribution_ST(bins_c, tranche, p0=p0) 
#    for tranche in hist_vs_time
# ]
all_fits = fit_distribution_ST_batch(bins_c, hist_vs_time, p0=p0, verbose=1)


# %%
# %matplotlib inline
def extract(fit):
    if fit.popt is None:
        return (np.nan, np.nan)
    #if fit.popt.mut > -0.425 or fit.popt.mut < -0.45:
    #    return (np.nan, np.nan)
    return (fit.popt.mus, fit.popt.mut)
vec_mus, vec_mut = zip(*(extract(fit) for fit in all_fits))
vec_mus, vec_mut = np.array(vec_mus), np.array(vec_mut)
plt.plot(vec_mus-vec_mut)
_ = plt.hist(vec_mut, bins=100)
print(f"Fit compté: {np.sum(~np.isnan(vec_mus))}")
"""
fig, ax = plt.subplots()
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.scatter(time_axis, vec_mus, label="Centre gaussienne singulet")
plt.scatter(time_axis, vec_mut, label="Centre gaussienne triplet")
plt.xlabel("Measure time")
plt.ylabel("Histogram I")
plt.legend()
"""

# %%
# %matplotlib inline
N = len(vec_mus)

def prepare_for_fft(vec):
    vec = np.array(vec)
    not_nan = ~np.isnan(vec)
    vec_mod = np.interp(time_axis, time_axis[not_nan], vec[not_nan])
    vec_mod = vec_mod - np.mean(vec_mod)
    return vec_mod
vec_mus_fft, vec_mut_fft = prepare_for_fft(vec_mus), prepare_for_fft(vec_mut)
if 0:
    plt.plot(vec_mus)
    plt.plot(vec_mus_fft)

dt = (time_axis[-1] - time_axis[0]) / N
fft_freqs = rfftfreq(N, dt)
print(f"Fréquence d'échantillonnage: {1/dt}")

singlet_fft_amps = rfft(vec_mus_fft)
triplet_fft_amps = rfft(vec_mut_fft)
maxi = fft_freqs[np.argmax(np.abs(singlet_fft_amps))]
plt.plot(fft_freqs, np.abs(singlet_fft_amps), label="FFT singulets")
plt.plot(fft_freqs, np.abs(triplet_fft_amps), label="FFT triplts")

plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.axvline(maxi, c="red", ls=":")
print(f"Ligne rouge: {maxi}")
plt.xlim(-10, 100)
plt.legend()
plt.show()

