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
from typing import List
import json
from time import monotonic, sleep
from functools import wraps
from pathlib import Path
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
        Tuple de 2 valeurs: variables QUA représentant le résultat de la démodulation.

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

def readout_demod_slice_macro(
    element: str,
    operation: str,
    element_output: str,
    amplitude: float,
    mode: str = "xy",  # rt
    chunk_time: int = 1, 
    
    cw_len_ns=1, quaconfig=None
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
        Tuple de 2 valeurs: variables QUA représentant le résultat de la démodulation.

    Raises:
        ValueError: Si `mode` n'est pas `'xy'` ou `'rt'`.
    """
    #pulse_name = config["elements"][element]["operations"][operation]
    #cw_len_ns = config["pulses"][pulse_name]["length"]

    if chunk_time % 4 != 0:
        raise ValueError(f"'chunk_time' must be a multiple of 4. Got {chunk_time}.")
    
    n_chunk = int(cw_len_ns / chunk_time)
    i_res = declare(fixed, size=n_chunk)
    q_res = declare(fixed, size=n_chunk)
    
    outputs = (
        demod.sliced(
            iw="cos",
            target=i_res,
            samples_per_chunk=chunk_time//4,
            element_output=element_output),
        demod.sliced(
            iw="sin",
            target=q_res,
            samples_per_chunk=chunk_time//4,
            element_output=element_output),
    )
    
    
    quaconfig.update_integration_weight(
        element=element, operation_name=operation,
        iw_op_name="cos",
        iw_cos=[(1, n_chunk * chunk_time)],
        iw_sin=[(0, n_chunk * chunk_time)]
    )
    quaconfig.update_integration_weight(
        element=element, operation_name=operation,
        iw_op_name="sin",
        iw_cos=[(0, n_chunk * chunk_time)],
        iw_sin=[(1, n_chunk * chunk_time)]
    )

    measure(operation*amp(amplitude), element, *outputs)

    if mode == "xy":
        return i_res, q_res

    elif mode == "rt":
        idx = declare(int, value=0)
        r = declare(fixed, size=n_chunk)
        theta = declare(fixed, size=n_chunk)
        
        with for_(idx, 0, idx<n_chunk, idx+1):
            r_value = mth.sqrt(mth.pow(mth.abs(i_res[idx]), 2.0) + mth.pow(mth.abs(q_res[idx]), 2.0))
            theta_value = mth.atan2(q_res[idx], i_res[idx])
            assign(r[idx], r_value)
            assign(theta[idx], theta_value)
        
        return r, theta

    else:
        raise(ValueError, f"{mode} is not recognized. Try 'xy' or 'rt'.")

def save_qua_array2stream(arr, st):
    i = declare(int)
    with for_(i, 0, i<arr.declaration_statement.size, i+1):
        save(arr[i], st)

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

        total_points = len(data_0.T[0])
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

    f = h5py.File(filename, "w", libver="latest")
    f.swmr_mode = True
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
            shape=list(map(len, ax_values)),
            dtype="f",
            fillvalue=np.nan,
        )
    
    setattr(f, "memory_dict", {})
    setattr(f, "flush_data", lambda res_handles: flush_from_res_handle(f, res_handles, print_progress=print_progress_on_flush))

    f.flush()

    return f



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
with program() as tunnel_rate:
    var = declare(fixed)
    vec = declare(fixed, size=10)
    print(dir(var))
    print(dir(vec))
    print(vec.declaration_statement)
    print(vec.declaration_statement.size)


# %%
r = job.result_handles.get("R")
r.count_so_far()
#r.count_so_far()
res = r.fetch_all()["value"]
res.shape

# %%
f["data"]["R"][:].flatten()

# %%
f = h5py.File(filename, "r")
f["data"]['R'][:]
f.close()

# %%
r.fetch(slice(0,1,None)).shape

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

