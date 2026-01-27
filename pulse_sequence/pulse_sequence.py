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

# %%
# Paramètres:
data_path = "D:/Intel_Tunel_Falls_12QD_01/data"
data_path = r"D:\test_opx"

# %%
# Librairies et fonctions
# %gui qt
import matplotlib.pyplot as plt
import numpy as np
import json
from time import sleep
from IPython import get_ipython
import os, datetime
import h5py
from matplotlib import pyplot as plt
from copy import deepcopy

import sys
sys.path.append("..")
# Functions for OPX use
from qm.qua import (
    program,
    assign,
    fixed,
    declare_stream,
    measure,
    wait, align,
    stream_processing,
    demod,
    dual_demod,
    declare,
    save,
    for_, for_each_, while_,
    update_frequency,
    play, pause, amp, 
    ramp_to_zero, ramp,
    FUNCTIONS
)
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua.lib import Math as mth
from qualang_tools.voltage_gates import VoltageGateSequence 
from qualang_tools.loops import (
    from_array,
)
from qualang_tools.config.helper_tools import QuaConfig

### UTIL
from utils.qua_custom import (
    readout_demod_macro,
    readout_demod_slice_macro,
    save_qua_array2stream,
    close_everything
)
from utils.file_saving import (
    make_path_fn,
    expand_filename,
    get_cell_content,
    sweep_file
)
path = make_path_fn(data_path)



# %% [markdown]
# # Programme QUA

# %%
## sweep 1d (juste avg)

# %%
# Pulse avec readout unique: r,t * n_avg
import config_100mV as cfg_file
from config_100mV import qop_ip, cluster_name, u, config

config_copy = deepcopy(config)  # Copier le dict de config

# Parameters
n_avg = 100
cw_readout_freq = 50 * u.MHz
cw_amp = 10 * u.mV
max_compensation_amp = 0.05

gain = 10**((attenuation_db:=20)/20)

# Gate space
gates = ["P2", "P3"]
points = [
    ["empty", -0.005, -0.005, 5000],
    ["init",   0.000,  0.004, 5000],
    ["1,1",    0.005,  0.005, 5000],
    ["read",    0.0,   0.0, cfg_file.cw_len]
]
sequence = VoltageGateSequence(config, gates)
for pts_name, x, y, time in points:
    x, y = gain*x, gain*y
    print(x,y)
    sequence.add_points(pts_name, [x, y], time)

# PSB readout program
with program() as psb_meas:
    n = declare(int)
    ramp_time = declare(int)

    r_st = declare_stream()
    t_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)
    sequence.ramp_to_zero()

    with for_(n, 0, n<n_avg, n+1):
        sequence.add_step(voltage_point_name="empty")
        sequence.add_step(voltage_point_name="init")
        sequence.add_step(voltage_point_name="1,1")
        align("RF-SET1", gates[0], gates[1])

        sequence.add_step(voltage_point_name="read")
        r, t = readout_demod_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out1",
            amplitude=cw_amp*10,
            mode="rt",
        )
        save(r, r_st)
        save(t, t_st)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp)
        sequence.ramp_to_zero()
        wait(1*u.us)

    with stream_processing():
        r_st.save_all("R")
        t_st.save_all("Theta")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_meas)

filename = expand_filename(path()+"%T_pulse.hdf5")
with sweep_file(
    filename,
    ax_names=["count"],
    ax_values=[np.arange(0, n_avg, 1)],
    out_names=["R", "Theta"],
    # -- meta:
    cell=get_cell_content(), 
    config=config.copy()
) as f:
    
    while not f.flush_data(job.result_handles):
        sleep(.1)



# %%
# Pulse avec readout unique: r,t * n_avg
import config_100mV as cfg_file
from config_100mV import qop_ip, cluster_name, u, config

config_copy = deepcopy(config)  # Copier le dict de config

# Parameters
n_avg = 100
cw_readout_freq = 500 * u.MHz
cw_amp = 10 * u.mV
chunk_time = 1000
gain = 10**((attenuation_db:=20)/20)

# Gate space
max_compensation_amp = .025
gates = ["P2", "P3"]
points = [
    ["empty", -0.005, -0.005, 5000],
    ["init",   0.000,  0.004, 5000],
    ["1,1",    0.005,  0.005, 5000],
    ["read",    0.0,   0.0, cfg_file.cw_len]
]
sequence = VoltageGateSequence(config, gates)
for pts_name, x, y, time in points:
    x, y = gain*x, gain*y
    print(x,y)
    sequence.add_points(pts_name, [x, y], time)

# PSB readout program
with program() as psb_meas:
    n = declare(int)
    ramp_time = declare(int)

    r_st = declare_stream()
    t_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)
    sequence.ramp_to_zero()

    with for_(n, 0, n<n_avg, n+1):
        sequence.add_step(voltage_point_name="empty")
        sequence.add_step(voltage_point_name="init")
        sequence.add_step(voltage_point_name="1,1")
        align("RF-SET1", gates[0], gates[1])

        sequence.add_step(voltage_point_name="read")
        r, t = readout_demod_slice_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out1",
            amplitude=cw_amp*10,
            mode="rt",
            chunk_time=chunk_time,
            cw_len_ns=cfg_file.cw_len,
            quaconfig=QuaConfig(config)
        )
        n_chunk = t.declaration_statement.size

        save_qua_array2stream(r, r_st)
        save_qua_array2stream(t, t_st)

        sequence.add_compensation_pulse(max_amplitude=max_compensation_amp)
        sequence.ramp_to_zero()
        wait(1*u.us)

    with stream_processing():
        r_st.buffer(n_chunk).save_all("R")
        t_st.buffer(n_chunk).save_all("Theta")

# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(psb_meas)

filename = expand_filename(path()+"%T_pulse.hdf5")
with sweep_file(
    filename,
    ax_names=["count", "temps (ms)"],
    ax_values=[np.arange(0, n_avg, 1), np.linspace(0, chunk_time*n_chunk, n_chunk)*1e-6],
    out_names=["R", "Theta"],
    # -- meta:
    cell=get_cell_content(), 
    config=config.copy()
) as f:
    
    while not f.flush_data(job.result_handles):
        sleep(.1)



# %% [markdown]
# affichage

# %%
with h5py.File(filename, 'r') as file:
        
    ax_names = file["data"].attrs["sweeped_ax_names"]
    sweep_names = file["data"].attrs["result_data_names"]

    x = file[f"data/{ax_names[0]}"]
    y = file[f"data/{ax_names[1]}"]
    z = [(file[f"data/{name}"], name) for name in sweep_names]

    for data, name in z:
        plt.pcolormesh(x, y, data)
        plt.xlabel(ax_names[0])
        plt.ylabel(ax_names[1])
        plt.title(name)
        plt.colorbar()
        plt.show()
        
