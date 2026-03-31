from copy import deepcopy
from qm.qua import (
    assign,
    fixed,
    measure,
    demod,
    declare,
    amp,
    for_,
    save,
    wait,
)
from qm.qua.lib import Math as mth
from qualang_tools.config.helper_tools import QuaConfig
from qualang_tools.units import unit
from qualang_tools.voltage_gates import VoltageGateSequence 
u = unit(coerce_to_integer=True)

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

def readout_demod_sliced_macro(
    element: str,
    operation: str,
    element_output: str,
    amplitude: float,
    n_slices: int,
    cw_len: int,
    i_st,
    q_st,
    n_cores: int=2,
):
    """
    Macro pour démoduler la sortie d'un élément en slice.

    Args:
        element (str): Nom de l'élément à lire
        operation (str): Opération définie dans le dictionnaire de config
        element_output (str): Sortie sur laquelle démoduler (définie dans la config)
        amplitude (float): Facteur d'amplitude sur le pulse qui est envoyé sur l'élément
        cw_len (int): Durée d'intégration pour chaque point en ns
        i_st, q_st: Stream dans lesquels enregistrer les données
        n_cores (int): Nombre de cores à utiliser. Le produit n_cores * cw_len devrait être supérieur à 650ns pour éviter les gap entre les démodulations.
    
    Exemple d'utilisation:
    ```
    elements = duplicate_element(config, element_name, n_cores)
    with program() as sliced_demod:
        i_st, q_st = declare_stream(), declare_stream()
        for ele in element:
            reset_if_phase(ele)
        
        align(*gates, *elements)
        readout_demod_sliced_macro(
            element= element_name,
            operation= operation,
            element_output= out,
            amplitude= 10*amp,
            n_slices= n_slices,
            cw_len= cw_len,
            i_st=i_st, q_st=q_st,
            n_cores= n_cores,
        )

        with stream_processing()
            i_st.buffer(n_slices).save_all("I")
            q_st.buffer(n_slices).save_all("Q")
    ```
    """
    if n_cores > 16:
        raise(ValueError, f"The maximum number of available cores is 16. Got {n_cores}.")

    required_size = [(n_slices // n_cores) + (n_slices % n_cores > i) for i in range(n_cores)]
    x_var = [declare(fixed) for i in range(n_cores)]
    y_var = [declare(fixed) for i in range(n_cores)]
    idx = [declare(int) for i in range(n_cores)]
    
    for core in range(1, n_cores+1):
        element_name = element
        if core != 1:
            element_name += f"_core{core}"
            wait(cw_len*(core-1)*u.ns, element_name)
        
        with for_(idx[core-1], 0, idx[core-1]<required_size[core-1], idx[core-1]+1):
            outputs = [
                demod.full(iw="cos", target=x_var[core-1], element_output=element_output),
                demod.full(iw="sin", target=y_var[core-1], element_output=element_output)
            ]
            measure(operation*amp(amplitude), element_name, *outputs)
            save(x_var[core-1], i_st)
            save(y_var[core-1], q_st)
            wait(cw_len*(n_cores-1)*u.ns, element_name)
    
    return x_var, y_var

def save_qua_array2stream(arr, st):
    i = declare(int)
    with for_(i, 0, i<arr.declaration_statement.size, i+1):
        save(arr[i], st)


def duplicate_element(config, name, nb: int = 2):
    """
    """
    new_elements = [name] + [f"{name}_core{i}" for i in range(2, nb+1)]
    for i, ele in enumerate(new_elements):
        config["elements"][ele] = deepcopy(config["elements"][name])
    return [name] + new_elements

def make_gate_sequence(config:dict, gates:list[str], points, gain:float) -> VoltageGateSequence:
    """
    gates = ["P2", "P3", "B2]
    points = [
        ["name", V_p2, V_p3, V_b2, default_duration]
        ["name2", V_p2, V_p3, V_b2, default_duration]
    ]
    """
    sequence = VoltageGateSequence(config, gates)
    for point in points:
        sequence.add_points(
            name = point[0],
            coordinates = [voltage*gain for voltage in point[1:-1]],
            duration = point[-1]
        )
    return sequence


def trypass(fn):
    try:
        fn()
    except Exception as e:
        print(e)
        pass

def close_everything(qmm, qm):
    try:
        trypass(qm.clear_queue)
        trypass(qm.close)
        trypass(qmm.reset_data_processing)
        trypass(qmm.clear_all_job_results)
        trypass(qmm.close_all_qms)
        trypass(qmm.close_all_quantum_machines)
    except:
        pass
