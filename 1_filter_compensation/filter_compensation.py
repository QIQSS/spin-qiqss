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
# # Trouver les paramètres à utiliser pour les filtres digitaux
#
# Ce notebook permet d'extraire la constante de temps d'une ligne RF ce qui permet par la suite de la spécifier dans le fichier de config et d'avoir des pulses avec la forme voulue.
#
# Pour déterminer la constante de temps d'une ligne RF, suivre la procédure suivante:
#  - Connecter un port RF d'une carte de l'OPX dans l'entrée 1 de la même carte.
#  - Dans le fichier de config, spécifier le numéro de la carte et du port testés (ligne 10-11 environ).
#  - Exécuter les cellules de ce notebook.
#  - Recommencer la procédure pour chaque port à caractériser.
#
# Une fois la constante de temps déterminée, elle peut être ajoutée dans le fichier de config avec:
#
# ```python
# "analog_outputs": {
#     port_num: {
#         ...
#         "filter": {
#             "exponential": [(100, constante_de_temps)],
#         },
#     },
# },
# ```
#
# **Note**:
# Ce code permet seulement de trouver des paramètres de compensation pour des filtres passe-haut d'ordre 1.

# %% [markdown]
# # Importation des librairies

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from copy import deepcopy
from qm.qua import program, declare_stream, measure, wait, stream_processing
from qm import QuantumMachinesManager

# %% [markdown]
# # Exécution du code

# %%
import filter_compensation_config as cfg_file
from filter_compensation_config import qop_ip, cluster_name, u, config

wait_time = 50e3
config["pulses"]["step_pulse"]["length"] = wait_time

# Copier le dictionnaire de config pour tester les paramètres de fit
config2 = deepcopy(config)

# Programme pour appliquer une marche et observer l'effet du filtre
with program() as test_prog:
    adc_st = declare_stream(adc_trace=True)
    measure("step", "filtered_line", adc_stream=adc_st)
    wait(2*u.us)
    
    with stream_processing():
        adc_st.input1().save_all("adc")
        adc_st.input1().timestamps().save_all("time")

# Exécuter le programme et récupérer les données
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config, close_other_machines=True)

job = qm.execute(test_prog)
res_handles = job.result_handles
res_handles.wait_for_all_values()
adc_res = res_handles.get("adc").fetch_all()["value"][0]
adc_res = u.raw2volts(-adc_res)
time_res = res_handles.get("time").fetch_all()["value"][0]
time_res = time_res-time_res[0]

# Extraire les paramètres du filtre
def decay(x, A, tau, offset):
    return A * np.exp(-x/tau) + offset

p0 = [adc_res[0], 10000, 0]
popt, _ = optimize.curve_fit(decay, time_res, adc_res, p0=p0)
_, tau_opt, _ = popt

# Exécuter le programme avec filtre maintenant et récupérer les données
config2["controllers"][cfg_file.con]["fems"][cfg_file.lf_num]["analog_outputs"][cfg_file.port_num]["filter"] = {"exponential": [(100, tau_opt)]}

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qm = qmm.open_qm(config2, close_other_machines=True)

job = qm.execute(test_prog)
res_handles = job.result_handles
res_handles.wait_for_all_values()
adc_res2 = res_handles.get("adc").fetch_all()["value"][0]
adc_res2 = u.raw2volts(-adc_res2)
time_res2 = res_handles.get("time").fetch_all()["value"][0]
time_res2 = time_res2-time_res2[0]

# Afficher les résultats
plt.plot(time_res, adc_res, label="Marche sans compensation")
plt.plot(time_res2, adc_res2, label="Marche avec compensation")
plt.plot(time_res, decay(time_res, *popt), label=f"$\\tau$ = {int(tau_opt)}ns")
plt.legend()
plt.xlabel("Temps (ns)")
plt.ylabel("Tension (V)")
plt.show()
