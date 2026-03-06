# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # calcul du taux tunnel.
# - moyennage d'un readout après un pulse marche carré.
# - fit de la décroissance exponentielle.
#

# %%
# Paramètres:
data_path = "D:/Intel_Tunel_Falls_12QD_01/data"
data_path = r"D:\test_opx"

# %%
# Librairies et fonctions
# %gui qt
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
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
    save_qua_array2stream
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
from config_taux_tunnel import qop_ip, cluster_name, u, config, cw_len
quaconfig = QuaConfig(config)
# Parameters
n_avg = 500
cw_readout_freq = 50 * u.MHz
cw_amp = 5 * u.mV

## pulse:
gate = "P2"
amplitude = .005

chunk_time = 10_000

gain = 10**((attenuation_db:=21)/20)

# PSB readout program
with program() as tunnel_rate:
    n = declare(int)
    ramp_time = declare(int)

    r_st = declare_stream()
    t_st = declare_stream()

    update_frequency("RF-SET1", cw_readout_freq)
    ramp_to_zero(gate)

    with for_(n, 0, n<n_avg, n+1):
        # in
        play("step"*amp(10*amplitude*gain), gate, duration=1000*u.ns)
        align("RF-SET1", gate)
        
        # out
        play("step"*amp(-10*amplitude*gain), gate, duration=cw_len*u.ns)

        r, t = readout_demod_slice_macro(
            element="RF-SET1",
            operation="readout",
            element_output="out1",
            amplitude=cw_amp*10,
            mode="rt",
            chunk_time=chunk_time, cw_len_ns=cw_len, quaconfig=quaconfig
        )
        n_chunk = t.declaration_statement.size

        save_qua_array2stream(r, r_st)
        save_qua_array2stream(t, t_st)

        ramp_to_zero(gate, duration=1000*u.ns)
        wait(500*u.us)
    
    with stream_processing():
        r_st.buffer(n_chunk).save_all("R")
        t_st.buffer(n_chunk).save_all("Theta")


# Exécuter le programme
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(tunnel_rate)

filename = expand_filename(path()+"%T_taux_tunnel.hdf5")
with sweep_file(
    filename,
    ax_names=["avg", "temps (ms)"],
    ax_values=[np.arange(0, n_avg, 1), np.linspace(0, chunk_time*n_chunk, n_chunk)*1e-6],
    out_names=["R", "Theta"],
    # -- meta:
    cell=get_cell_content(), 
    config=config.copy()
) as f:
    
    while not f.flush_data(job.result_handles):
        sleep(.1)



# %%
r = job.result_handles.get("R")
r.count_so_far()
#r.count_so_far()
res = r.fetch_all()["value"]
res.shape

# %% [markdown]
# affichage

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C

with h5py.File(filename, 'r') as file:
    temps = file["data/temps (ms)"][:]
    traces = file["data/R"][:].T

mean_trace = np.mean(traces, axis=1)

p0 = (mean_trace.max() - mean_trace.min(), 
      (temps[-1] - temps[0]) / 2, 
      mean_trace.min())

params, cov = curve_fit(exp_decay, temps, mean_trace, p0=p0)
A, tau, C = params

print(f"A = {A:.3g}, tau = {tau:.3g} ms, C = {C:.3g}")


tau_s = tau * 1e-3
freq = 1 / (2 * np.pi * tau_s)

# plot
plt.figure()
plt.plot(temps, mean_trace, label=f"Après step: moyenné {len(traces)}")
plt.plot(temps, exp_decay(temps, *params), "--", label=f"Exp fit (tau={tau:.2f} ms, f={freq:.3g} Hz)")
plt.xlabel("temps (ms)")
plt.ylabel("R")
plt.title(filename)
plt.legend()
plt.show()


