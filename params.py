from qualang_tools.units import unit
u = unit(coerce_to_integer=True)

# Démodulation
cw_len = 50_000
cw_len_short = 1_000
# cw_len = 500

cw_readout_freq = 540.35 * u.MHz
cw_amp = 15e-3

# Compensation du montage
max_compensation_amp = 0.011
attenuation_db = 21
gain = 10**(attenuation_db / 20)

# Pulses
gates = ["P2", "P3", "B2"]
operation_points = [
    ["zero_dc", 0, 0, 0, 0],
    ["init", 0.004, -0.004, 0, 100_000],
    ["load", -0.002, 0.002, 0, 16],
    
    ["load_deep", -0.005, 0.005, 0.000, 16],
    ["readout", *(-0.00038, 0.00038), 0, cw_len],
]

# gates = ["P2", "P3", "B2"]
# operation_points = [
#     ["zero_dc", 0, 0, 0, 0],
#     ["init", -0.004, 0.004, 0, 100_000],
#     ["load", 0.002, -0.002, 0, 16],
#     ["load_deep", 0.005, -0.005, 0.000, 16],
#     ["readout", *(0.0005, -0.0005), -0.000, cw_len],
# ]

pi_over_2_level = [-0.004, 0.004]
pi_over_2_duration = 1200

# Autres
iq_phase = -2.668957687985658
threshold = -0.301