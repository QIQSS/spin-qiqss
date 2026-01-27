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
from time import sleep as slp

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
    reset_global_phase,
)
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua.lib import Math as mth
from qualang_tools.voltage_gates import VoltageGateSequence 
from qualang_tools.loops import (
    from_array,
)

### UTIL
from utils.qua_custom import readout_demod_macro
from utils.file_saving import (
    make_path_fn,
    expand_filename,
    sweep_file
)
path = make_path_fn(data_path)

# %%
from config import qop_ip, cluster_name, u, config, time_of_flight

cw_amp = 1.50e-3

with program() as tof:
    adc_st = declare_stream(adc_trace = True)
    reset_global_phase()
    update_frequency("RF-SET1", 322.4e6)

    measure("readout"*amp(cw_amp), "RF-SET1", adc_stream=adc_st)

    with stream_processing():
        adc_st.input1().save_all("trace")
        adc_st.input1().timestamps().save_all("time")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(tof)


# %% [markdown]
# Affichage

# %%
# %matplotlib qt
res_handles = job.result_handles
res_handles.wait_for_all_values()
tm = res_handles.get("time").fetch_all()["value"][0]
trace = res_handles.get("trace").fetch_all()["value"][0]
trace = u.raw2volts(-trace)

idx = np.where(np.abs(trace) > 0.01)[0][0]
tof_calculated = 4*(np.round((tm[idx]-tm[0])/4)) + time_of_flight
print(f"Temps de vol: {tof_calculated} ns")

plt.plot(tm, trace)
plt.axvline(tm[0] + tof_calculated - time_of_flight)
plt.show()
