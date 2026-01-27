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
# # Setup

# %%
# Paramètres:
data_path = "D:/Intel_Tunel_Falls_12QD_01/data"
data_path = r"D:\test_opx"

# %%
# Librairies et fonctions
# %gui qt
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import json
from time import monotonic, sleep
from functools import wraps
from pathlib import Path
from IPython import get_ipython
import os, datetime
import h5py
from matplotlib import pyplot as plt

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

### UTIL
### UTIL
from utils.qua_custom import readout_demod_macro

from utils.file_saving import (
    make_path_fn,
    expand_filename,
    sweep_file
)
path = make_path_fn(data_path)



# %% [markdown]
# # Main program

# %%
from config import qop_ip, cluster_name, u, config, cw_len
from videomode_lib.videomode import VideoModeWindow, Sweep

short_axis = Sweep.from_nbpts(-20e-3, 20e-3, 31, "P1", 21, interlace=0)
long_axis = Sweep.from_nbpts(-10e-3, 10e-3, 8, "P2", 21, interlace=0)

cw_readout_freq = 300 * u.MHz
cw_amp = 0.5

cw_readout_len = cw_len
before_wait = 1000 * u.ns
short_duration = cw_readout_len + before_wait
long_duration = short_duration * short_axis.nbpts

# Def config
with program() as videomode:
    update_frequency("RF-SET1", cw_readout_freq)
    n, m = declare(fixed), declare(fixed)
    r_st = declare_stream() # R
    t_st = declare_stream() # Theta
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
                r, t = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out1",
                    amplitude=cw_amp,
                    mode="rt",)
                save(r, r_st)
                save(t, t_st)
                wait(2*before_wait * u.ns, "oscillo")
                measure("raw", "oscillo", adc_stream=adc_st)
            ramp_to_zero(short_axis.element)

        ramp_to_zero(long_axis.element)
    
    with stream_processing():
        (adc_st
            .input1()
            .map(FUNCTIONS.average())
            .buffer(short_axis.nbpts)
            .buffer(long_axis.nbpts)
            .save('raw1')
        )
        r_st.buffer(short_axis.nbpts).buffer(long_axis.nbpts).save("R")
        t_st.buffer(short_axis.nbpts).buffer(long_axis.nbpts).save("Theta")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qmm.close_all_qms()
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(videomode)


vm = VideoModeWindow.from_job(
   job, save_path = path(),
   out_name = "R",
   short_axis = short_axis,
   long_axis = long_axis,
   play = 1
)
# vm2 = VideoModeWindow.from_job(
#    job, save_path = path(),
#    out_name="Theta",
#    short_axis=short_axis,
#    long_axis=long_axis,
# )

# %%
handle = job.result_handles.get("raw1")
job.resume()


# %%
data = handle.fetch_all()
print(data)

# %%
out_name = "raw1"
def get_map(job):
    # Play opx
    # Wait for pause
    # Get result
    handle = job.result_handles.get(out_name)
    if handle is None: raise KeyError(f"{out_name} probably not right")
    job.resume()
    while not job.is_paused() and len(handle) != 0:
        sleep(0.001)
    try:
        res = handle.fetch_all()
        res = u.raw2volts(res)
    except KeyError:
        return get_map(job)

    # if short_axis.is_interlaced:
    # res = interlace_array(res.T).T
    # if short_axis.step < 0:
    #     res = res.T[::-1].T
    # if long_axis.is_interlaced:
    #     res = interlace_array(res)
    # if long_axis.step < 0:
    #     res = res[::-1]
    
    return res
get_map(job)

# %%
vm.show()

# %%
vm

# %%
handle = job.result_handles.get(out_name)
handle.fetch_all().shape

# %%
res = get_map(job)

# %%
plt.imshow(res)


# %% [markdown]
# Acquisition et trace sans fenêtre continue:

# %%
# Get and plot one map
def plot_map(fig, ax, data, title='', label=""):
    extent = (long_axis.start, long_axis.stop, short_axis.start, short_axis.stop)
    im = ax.imshow(data.T, extent=extent, aspect='auto', origin='lower', interpolation='none')
    ax.set_xlabel(f"delta {long_axis.element} (V)")
    ax.set_ylabel(f"delta {short_axis.element} (V)")
    cb = fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        location='top',
    )
    cb.set_label(label)
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')

    return fig, ax, cb

job.resume()
while not job.is_paused():
    sleep(0.001)

r = job.result_handles.get("R").fetch_all()
t = job.result_handles.get("Theta").fetch_all()

fig, [ax1, ax2] = plt.subplots(1, 2)
plot_map(fig, ax1, r, 'Amplitude', label="Amplitude")
plot_map(fig, ax2, t, 'Phase', label="Phase")
fig.tight_layout()
plt.show()

# %%
