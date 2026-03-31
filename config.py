from params import *
import numpy as np
from qualang_tools.config.helper_tools import QuaConfig
from copy import deepcopy



######################
# Network parameters #
######################
qop_ip = "192.168.0.11"  # Write the QM router IP address
cluster_name = "Cluster_1"  # Write your cluster_name if version >= QOP220
con = "con1"

lf_num = 1

RF_SET1_port = 8
RF_SET1_in = 1
B2_port = 5
P2_port = 2
P3_port = 4
Trigger_port = 6

sampling_rate_gate = int(1e9)  # or, int(1e9)
sampling_rate_rf = int(2e9)

# Time of flight
time_of_flight = 540


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
                            "offset": 0.012,  # direct
                            # "offset": 0.0677,  # amplified
                            "sampling_rate": sampling_rate_rf,
                            "output_mode": "direct",
                        },
                        P2_port: {
                            # "offset": 0.0436,  # amplified
                            "offset": 0.0075,
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "direct",
                            "filter": {
                                # "exponential": [(100, 625_000_000)],
                            },
                        },
                        P3_port: {
                            # "offset": 0.0166,  # amplified
                            "offset": 0.0015,
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "direct",
                            "filter": {
                                # "exponential": [(100, 625_000_000)],
                            },
                        },
                        B2_port: {
                            "offset": 0.0075,
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "direct",
                            "filter": {
                                # "exponential": [(100, 625_000_000)],
                            },
                        },
                        Trigger_port: {
                            "offset": 0.000,  # direct
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "direct",
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
                "out": (con, lf_num, RF_SET1_in),
            },
            "time_of_flight": time_of_flight,
            "intermediate_frequency": 322.4e6,
            "operations": {
                "readout": "meas_pulse",
                "readout_short": "meas_pulse_short",
            },
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
                "duration": 1000 # ramp to zero duration
            }
        },
        "P3": {
            "singleInput": {
                "port": (con, lf_num, P3_port)
            },
            "intermediate_frequency": 0,
            "operations": {
                "step": "control_pulse",
            },
            "sticky": {
                "analog": True,
                "duration": 1000 # ramp to zero duration
            }
        },
        "B2": {
            "singleInput": {
                "port": (con, lf_num, B2_port)
            },
            "intermediate_frequency": 0,
            "operations": {
                "step": "control_pulse",
            },
            "sticky": {
                "analog": True,
                "duration": 1000 # ramp to zero duration
            }
        },
        "oscillo": {
            "outputs": {
                "raw": (con, lf_num, 1),
            },
            "digitalInputs": {
                "trig": {
                    "port": (con, lf_num, 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "time_of_flight": time_of_flight,
            "intermediate_frequency": 0,
            "operations": {
                "raw": "raw_pulse",
            }
        },
        "Trigger": {
            "singleInput": {
                "port": (con, lf_num, Trigger_port)
            },
            "intermediate_frequency": 0,
            "operations": {
                "step": "control_pulse",
            },
            "sticky": {
                "analog": True,
                "duration": 1000 # ramp to zero duration
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
        "meas_pulse_short": {
            "operation": "measurement",
            "length": cw_len_short,
            "waveforms": {
                "single": "cst_wf",
            },
            "digital_marker": "ON",
            "integration_weights": {
                "cos": "cosine_weights_short",
                "sin": "sine_weights_short",
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
        "cst_wf": {"type": "constant", "sample": 0.1},
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
        "cosine_weights_short": {
            "cosine": [(1.0, cw_len_short)],
            "sine": [(0.0, cw_len_short)],
        },
        "sine_weights_short": {
            "cosine": [(0.0, cw_len_short)],
            "sine": [(1.0, cw_len_short)],
        },
    },
}

config_copy = deepcopy(config)
config = QuaConfig(config)
config.update_integration_weight("RF-SET1", "readout", "cos", [(np.cos(iq_phase), cw_len)], [(np.sin(iq_phase), cw_len)])
config.update_integration_weight("RF-SET1", "readout", "sin", [(np.cos(iq_phase+np.pi/2), cw_len)], [(np.sin(iq_phase+np.pi/2), cw_len)])
config.update_integration_weight("RF-SET1", "readout_short", "cos", [(np.cos(iq_phase), cw_len_short)], [(np.sin(iq_phase), cw_len_short)])
config.update_integration_weight("RF-SET1", "readout_short", "sin", [(np.cos(iq_phase+np.pi/2), cw_len_short)], [(np.sin(iq_phase+np.pi/2), cw_len_short)])
