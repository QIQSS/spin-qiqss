import numpy as np
from qualang_tools.units import unit
from qualang_tools.config.waveform_tools import flattop_cosine_waveform


######################
# Network parameters #
######################
qop_ip = "192.168.137.169"  # Write the QM router IP address
cluster_name = "Cluster_1"  # Write your cluster_name if version >= QOP220
con = "con1"

lf_num = 1

RF_SET1_port = 8
P1_port = 1
P2_port = 4

sampling_rate_gate = int(1e9)  # or, int(1e9)
sampling_rate_rf = int(2e9)
u = unit(coerce_to_integer=True)

# Time of flight
time_of_flight = 192

cw_len = 1_000
flattop_cosine = flattop_cosine_waveform(1, 500, 100) # 100 + 500 + 100


#############################################
#                  Config                   #
#############################################
config = {
    "controllers": {
        con: {
            "type": "opx1000",
            "fems": {
                lf_num: {
                    "type": "LF",
                    "analog_outputs": {
                        RF_SET1_port: {
                            "offset": 0.068,  # Amplified +- 2.5
                            "sampling_rate": sampling_rate_rf,
                            "output_mode": "amplified",
                        },
                        P1_port: {
                            "offset": 0.008,  # Direct +- .5
                            # "offset": 0.045,  # Amplified
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "direct",
                            "filter": {
                                #"exponential": [(100, 16967)],
                            },
                        },
                        P2_port: {
                            "offset": 0.001,  # Direct
                            # "offset": 0.017,  # Amplified
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "direct",
                            "filter": {
                                #"exponential": [(100, 17703)],
                            },
                        },
                    },
                    "analog_inputs": {
                        1: {
                            "offset": 0,
                            "sampling_rate": sampling_rate_rf,
                            "gain_db": 0,
                        }, 
                        2: {
                            "offset": 0.0,
                            "sampling_rate": sampling_rate_rf,
                            "gain_db": 0,
                        }, 
                    },
                    "digital_outputs": {
                        1: {},
                    },
                },
            },
        }
    },
    "elements": {
        "RF-SET1": {
            "singleInput": {
                "port": (con, lf_num, RF_SET1_port)
            },
            "outputs": {
                "out1": (con, lf_num, 1),
                "out2": (con, lf_num, 2)
            },
            "time_of_flight": time_of_flight,
            "intermediate_frequency": 5e5,
            "operations": {
                "readout": "meas_pulse",
            }
        },
        "P1": {
            "singleInput": {
                "port": (con, lf_num, P1_port)
            },
            "intermediate_frequency": 0,
            "operations": {
                "step": "control_pulse",
            },
            "sticky": {
                "analog": True,
                "duration": 1000 # ramp duration
            }
        },
        "P2": {
            "singleInput": {
                "port": (con, lf_num, P2_port)
            },
            "intermediate_frequency": 0,
            "operations": {
                "step": "control_pulse",
            },
            "sticky": {
                "analog": True,
                "duration": 1000 # ramp duration
            }
        },
        "oscillo": {
            "outputs": {
                "raw": (con, lf_num, 1),
            },
            "time_of_flight": time_of_flight,
            "intermediate_frequency": 0,
            "operations": {
                "raw": "raw_pulse",
            }
        },
    },
    "pulses": {
        "meas_pulse": {
            "operation": "measurement",
            "length": cw_len,
            "waveforms": {
                "single": "cst_wf",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
        },
        "control_pulse": {
            "operation": "control",
            "length": cw_len,
            "waveforms": {
                "single": "cst_wf",
            },
            "digital_marker": "ON",
        },
        "raw_pulse": {
            "operation": "measurement",
            "length": 100,
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "cst_wf": {"type": "constant", "sample": 1.0},
        "test_wf": {"type": "arbitrary", "samples": flattop_cosine}
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },

    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, cw_len)],
            "sine": [(0.0, cw_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, cw_len)],
            "sine": [(1.0, cw_len)],
        },
    },
}
