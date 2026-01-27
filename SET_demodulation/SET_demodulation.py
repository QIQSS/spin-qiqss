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
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import h5py

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
    reset_phase
)
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua.lib import Math as mth
from qualang_tools.voltage_gates import VoltageGateSequence 
from qualang_tools.loops import (
    from_array,
)


### UTIL
from utils.qua_custom import readout_demod_macro
from utils.file_saving import make_path_fn, get_cell_content
path = make_path_fn(data_path)

from utils.file_saving import (
    make_path_fn,
    expand_filename,
    sweep_file
)
path = make_path_fn(data_path)


# %%
from config import qop_ip, cluster_name, u, config

frequencies = np.arange(0e6, 1000e6+1, .01e6)
cw_amp = 1

with program() as sweep_rf:

    f = declare(int)  # QUA variable for the readout frequency
    r_st = declare_stream()
    t_st = declare_stream()

    with for_(*from_array(f, frequencies)):

        update_frequency("RF-SET1", f)
        r, t = readout_demod_macro(
                element="RF-SET1",
                operation="readout",
                element_output="out1",
                amplitude=cw_amp,
                mode="rt",)

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
    cell=get_cell_content(), 
    config=config.copy()
) as f:

    while not f.flush_data(job.result_handles):
        sleep(2)

# %% [markdown]
# Affichage

# %%
with h5py.File(filename, 'r') as f:
    ax_names = f["data"].attrs["sweeped_ax_names"]

    # chargement
    x = f["data"][ax_names[0]][:]
    r = f["data"]["R"][:]
    t = f["data"]["Theta"][:]
    ###
    # traitement (déroule, dépente)
    t = np.unwrap(t)
    t = t - (t[-1]-t[0])/(x[-1]-x[0]) * x
    ###
    # trace
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(x, r)
    ax[1].plot(x, t)

    ax[0].set_ylabel("R")
    ax[0].grid(True)
    ax[0].set_title(filename)

    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Theta")
    ax[1].grid(True)

    fig.tight_layout()
    plt.show()
    ###
