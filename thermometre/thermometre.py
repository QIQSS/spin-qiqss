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
# Header for setting pyHegel in a jupyter notebook
from pyHegel.scipy_fortran_fix import fix_problem_new
fix_problem_new()
from pyHegel.commands import *
_init_pyHegel_globals()
# %gui qt

# %%
import datetime
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# %%
def path():
    date = datetime.datetime.now().strftime("%Y%m%d")
    path = os.path.join("D:"+os.path.sep, "Intel_Tunel_Falls_12QD_01", "data", date)

    if not os.path.exists(path):
        os.mkdir(path)
    return path + os.path.sep

def try_load(instruments, add_to_global=True):
    results = {}
    g = globals()
    for name, constructor in instruments.items():
        try:
            inst = g[name]
            print(f"{name}: already loaded")
        except Exception as e:
            print(f"{name}: loading", end="")
            inst = constructor()
            sys.stdout.flush()
            print(f", ok")
            if add_to_global: g[name] = inst  # add to globals
            continue
        results[name] = inst
    return results

instruments_to_load = {
    "dmm": lambda: instruments.agilent_multi_34410A("USB0::0x2A8D::0x0101::MY60096682::0::INSTR"),
}

loaded_instruments = try_load(instruments_to_load)

# %%
filename = path()+"%T_temperature_over_time.txt"

dmm_therm = instruments.ScalingDevice(dmm.readval, scale_factor=1e0, only_val=True, invert_trans=True)
dmm.nplc.set(100)

therm = _Snap()
therm(dmm.readval, filename)

# %%
while True:
    therm()

# %%
filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260310\temperature_over_time.txt"
filename = r"d:\Intel_Tunel_Falls_12QD_01\data\20260311\20260311-160457_temperature_over_time.txt"

# %%
data = readfile(filename)
time, volt = data
plt.plot(time/(3600*24), volt)
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%D %H:%M"))
ax.tick_params(axis="x", labelrotation=45)
