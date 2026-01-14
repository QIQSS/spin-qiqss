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
from dataclasses import dataclass
from typing import List
import json
from time import monotonic, sleep
from functools import wraps
from pathlib import Path
from IPython import get_ipython
import os, datetime
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

    outputs = (
        demod.full(iw="cos", target=i_res, element_output=element_output),
        demod.full(iw="sin", target=q_res, element_output=element_output),
    )
    measure(operation*amp(amplitude), element, *outputs)

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

    grp.file.flush() # Note: File.file is file so this work.

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
        False si les donneés ne sont pas complètes
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
        out1: array  ->  se rempli avec .flush_data()
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

filename = expand_filename(path()+"%T_test_opx.hdf5")
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
