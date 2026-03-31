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
mx = instruments.AmericanMagnetics_model430("tcpip::B4027_X-AX.lan::7180::socket")
my = instruments.AmericanMagnetics_model430("tcpip::B4030_Y-AX.lan::7180::socket")
mz = instruments.AmericanMagnetics_model430("tcpip::B4032_Z-AX.lan::7180::socket")

# %%
iprint(mz, True)

# %%
set(mx.persistent_switch_en, True)
set(my.persistent_switch_en, True)
set(mz.persistent_switch_en, True)

# %%
set(mz.field_target_T, 0.)
mz.set_state("ramp")

# %%
spy(mz.field_T)
# spy(mz.state)

# %%
print(get(mz.state))
print(get(mz.led_state))

# %%
set(mx.persistent_switch_en, False)
set(my.persistent_switch_en, False)
set(mz.persistent_switch_en, False)

# %%
spy(mz.state)

# %%
iprint(mz, True)
