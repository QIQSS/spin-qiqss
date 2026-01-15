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

def path():
    date = datetime.datetime.now().strftime("%Y%m%d")
    path = os.path.join(data_path, date)

    if not os.path.exists(path):
        os.mkdir(path)
    return path + os.path.sep
    
def get_cell_content(cell_number=-1):
    return get_ipython().user_ns['In'][cell_number]


### ###

### QUA

def readout_demod_macro(
    element: str,
    operation: str,
    element_output: str,
    amplitude: float,
    adc_stream = None,
    mode: str = "xy",  # rt
):
    """
    Macro pour démoduler la sortie d'un élément.

    Args:
        element (str): Nom de l'élément à lire
        operation (str): Opération définie dans le dictionnaire de config
        element_output (str): Sortie sur laquelle démoduler (définie dans la config)
        amplitude (float): Facteur d'amplitude sur le pulse qui est envoyé sur l'élément
        mode (str): Extraire les paramètres X-Y (`mode='xy'`) ou R-Theta (`mode='rt'`) en démodulant
    
    Returns:
        Tuple de variables QUA représentant le résultat de la démodulation

    Raises:
        ValueError: Si `mode` n'est pas `'xy'` ou `'rt'`.
    """
    i_res = declare(fixed)  # 'I' quadrature
    q_res = declare(fixed)  # 'Q' quadrature

    if adc_stream is None:
        outputs = (
            demod.full(iw="cos", target=i_res, element_output=element_output),
            demod.full(iw="sin", target=q_res, element_output=element_output),
        )
        measure(operation*amp(amplitude), element, *outputs)
    else:
        measure(operation*amp(amplitude), element, adc_stream=adc_stream)
        return

    if mode == "xy":
        return i_res, q_res

    elif mode == "rt":
        r = declare(fixed)
        theta = declare(fixed)
        assign(r, mth.sqrt(mth.pow(mth.abs(i_res), 2.0) + mth.pow(mth.abs(q_res), 2.0)))
        assign(theta, mth.atan2(q_res, i_res))
        return r, theta

    else:
        raise(ValueError, f"{mode} is not recognized. Try 'xy' or 'rt'.")

### ###

### SWEEP FILE

def h5_dump_dict(grp:h5py.File, **dict_):
    """
    Ajoute des dict en tant qu'attributs dans le group "grp" d'un fichier.
    La fonction essaie d'enregistrer les métadonnées directement. Si l'enregistrement
    échoue, les données sont sérialisées.

    Args:
        grp: h5py.File ou h5py.Group
        dict_: dictionnaire
    """
    for key, val in dict_.items():
        try:
            grp.attrs[key] = val
        except:
            grp.attrs[key] = json.dumps(val, indent=2)

    grp.file.flush() # Note: File.file is file so this work even if grp is a File

def _check_ax_args(ax_names, ax_values):
    if ax_names != []:
        if len(ax_values) != len(ax_names):
            raise(ValueError, f"axs (size {len(ax_values)}) and ax_names (size {len(ax_names)}) must be of same length.")
    else:
        ax_names = [f"ax{idx}" for idx in range(len(ax_values))]
    return ax_names, ax_values

def flush_from_res_handle(file, res_handles, print_progress=True):
    """
    Parcours les variables
    - file/data/<out_var>
    Mets à jour ces variables en allant les chercher par le même nom dans res_handles:
    - file/data/<out_var> = res_handles.get(out_var).fetch()

    La modification est faite seulement si de nouvelles données sont présentes.

    Retourne:
        False si les donneés ne sont pas entièrement remplies
        True sinon
    """

    is_complete = not res_handles.is_processing()

    memory_dict = file.memory_dict

    # Extraire et écrire les données
    out_names = file["data"].attrs["result_data_names"]
    for out_name in out_names:
        handle = res_handles.get(out_name)

        last_n = memory_dict.get(out_name, 0)
        n_available = handle.count_so_far()

        if last_n != n_available:
            slc = slice(last_n, n_available)
            new_data = handle.fetch(slc)["value"]
            memory_dict[out_name] = n_available

            file["data"][out_name][slc] = new_data

    if print_progress:     
        name_0, data_0 = out_names[0], file["data"][out_names[0]][:]

        total_points = len(data_0.flatten())
        current_point = file.memory_dict.get(name_0, 0)
        
        print(f"Avancement: {current_point}/{total_points}     ", end="\r")

    file.flush()

    return is_complete

def expand_filename(filename):
    filename = filename.replace("%T", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return filename

def sweep_file(
    filename: str,
    ax_names: list[str] = [],
    ax_values: list[np.ndarray] = [],
    out_names: list[str] = [],
    print_progress_on_flush: bool = True,
    **metadata
):
    """ 
    Crée et retourne le fichier hdf5.
    - out_names: nom des variables de res_handles à sauvegarder.
    - définie une fonction `flush_data(res_handles)` pour ajouter les nouvelles données disponibles.

    structure du fichier
    data:
        attrs: 
            "sweeped_ax_names": ["x", "y", ...]
            "result_data_names": ["out1", "out2", ...]
        x: array
        y: array
        ...
        out1: array  ->  se rempli avec .flush_data(qmjob.res_handles)
        out2: array
        ...
    meta:
        attrs: **metadata


    Clés réservées dans metadata:
    - VERSION
    """
    ax_names, ax_values = _check_ax_args(ax_names, ax_values)

    f = h5py.File(filename, "w")
    # meta
    f.create_group("meta")
    metadata["VERSION"] = 0.1
    h5_dump_dict(f["meta"], **metadata)
    # data
    f.create_group("data")
    data_grp = f["data"]
    data_grp.attrs["sweeped_ax_names"] = ax_names
    data_grp.attrs["result_data_names"] = out_names
    for idx, (ax, name) in enumerate(zip(ax_values, ax_names)):
        dset = data_grp.create_dataset(name, data=ax)
        dset.attrs["ax_no"] = idx
    for name in out_names:
        data_grp.create_dataset(
            name,
            shape=map(len, reversed(ax_values)),
            dtype="f",
            fillvalue=None,
        )
    
    setattr(f, "memory_dict", {})
    setattr(f, "flush_data", lambda res_handles: flush_from_res_handle(f, res_handles, print_progress=print_progress_on_flush))

    f.flush()

    return f



# %%
from config import qop_ip, cluster_name, u, config, cw_len
from videomode_lib.videomode import VideoModeWindow, Sweep

long_axis = Sweep.from_step(-15e-3, 15e-3, .2e-3, "P2")
short_axis = Sweep.from_step(-20e-3, 20e-3, .2e-3, "P1")

cw_readout_freq = 300 * u.MHz
cw_amp = 1.5

cw_readout_len = cw_len
wait_before_meas = 1000 * u.ns
cw_step_len = cw_readout_len + 2*wait_before_meas
# seconds -> clock cycle:
wait_duration = int(wait_before_meas / 4)
short_duration = int(cw_step_len/4)
long_duration = short_duration * short_axis.nbpts + 4*wait_duration

# Def config
with program() as videomode:
    update_frequency("RF-SET1", cw_readout_freq)
    n, m = declare(fixed), declare(fixed)    
    r_st = declare_stream() # R
    t_st = declare_stream() # Theta
    #adc_st = declare_stream(adc_trace=True)
    with while_(True):
        pause()
        with for_each_(n, long_axis.stickysteps):
            play("step"*amp(n), long_axis.element, duration=long_duration)
            with for_each_(m, short_axis.stickysteps):
                play("step"*amp(m), short_axis.element, duration=short_duration)
                r, t = readout_demod_macro(
                    element="RF-SET1",
                    operation="readout",
                    element_output="out1",
                    amplitude=cw_amp,
                    mode="rt",)
                save(r, r_st)
                save(t, t_st)
                #measure("raw", "oscillo", adc_stream=adc_st)
            ramp_to_zero(short_axis.element)

        ramp_to_zero(long_axis.element)

    with stream_processing():
        # (adc_st
        #     .input1()
        #     .real()
        #     .map(FUNCTIONS.average())
        #     .buffer(short_axis.nbpts)
        #     .buffer(long_axis.nbpts)
        #     .save('raw1')
        # )
        r_st.buffer(short_axis.nbpts).buffer(long_axis.nbpts).save("R")
        t_st.buffer(short_axis.nbpts).buffer(long_axis.nbpts).save("Theta")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)
job = qm.execute(videomode)

vm = VideoModeWindow.from_job(
   job,
   out_name="R",
   short_axis=short_axis,
   long_axis=long_axis
)
vm2 = VideoModeWindow.from_job(
   job,
   out_name="Theta",
   short_axis=short_axis,
   long_axis=long_axis
)

# %%
import pyqtgraph as pg
exp = pg.exporters.ImageExporter(vm2.graph.scene())
p = exp.parameters()
p["antialias"] = True
from PyQt5.QtGui import QColor
color = QColor(255, 255, 255)
p["background"] = color
print([(c.name(), c.value()) for c in p.children()])
exp.export(copy=True)

# %%
exp.export("test.svg")


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
