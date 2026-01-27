from qualang_tools.units import unit

######################
# Network parameters #
######################
qop_ip = "192.168.137.169"  # Write the QM router IP address
cluster_name = "Cluster_1"  # Write your cluster_name if version >= QOP220
con = "con1"

lf_num = 1
port_num = 4

sampling_rate_gate = int(1e9)
sampling_rate_rf = int(1e9)
u = unit(coerce_to_integer=True)

time_of_flight = 1000


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
                        port_num: {
                            "sampling_rate": sampling_rate_gate,
                            "output_mode": "amplified",
                        },
                    },
                    "analog_inputs": {
                        1: {"sampling_rate": sampling_rate_rf,},
                    },
                },
            },
        }
    },
    "elements": {
        "filtered_line": {
            "singleInput": {
                "port": (con, lf_num, port_num),
            },
            "outputs": {
                "filtered_signal": (con, lf_num, 1),
            },
            "time_of_flight": time_of_flight,
            "intermediate_frequency": 0,
            "operations": {
                "step": "step_pulse",
            },
            "sticky": {
                "analog": True,
                "duration": 1000,
            },
        },
    },
    "pulses": {
        "step_pulse": {
            "operation": "measurement",
            "length": 0,
            "digital_marker": "HIGH",
            "waveforms": {
                "single": "cst_wf",
            },
        },
    },
    "waveforms": {
        "cst_wf": {"type": "constant", "sample": 0.1},
    },
    "digital_waveforms": {
        "HIGH": {"samples": [(1, 0)]},
    },
}
