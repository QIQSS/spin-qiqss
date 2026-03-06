from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

# Démodulation
cw_len = 50_000
cw_len_short = 1_000

cw_readout_freq = 540.35 * u.MHz
cw_amp = 15e-3

# Compensation du montage
max_compensation_amp = 0.007
attenuation_db = 21
gain = 10**(attenuation_db / 20)

# Pulses
gates = ["P2", "P3"]
operation_points = [
    ["zero_dc", 0, 0, 0],
    ["init", 0.007, -0.007, 1_000],
    ["load", -0.004, 0.004, 0],
    ["load_deep", -0.007, 0.007, 0],
    ["readout", *(-0.0003, 0.0003), cw_len],
]

# Autres
iq_phase = 0.8
threshold = -0.301
