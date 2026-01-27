
from qm.qua import (
    assign,
    fixed,
    measure,
    demod,
    declare,
    amp,
    for_,
    save,
)
from qm.qua.lib import Math as mth
from qualang_tools.config.helper_tools import QuaConfig

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

def readout_demod_slice_macro(
    element: str,
    operation: str,
    element_output: str,
    amplitude: float,
    mode: str = "xy",  # rt
    chunk_time: int = 1, 
    cw_len_ns=1, quaconfig:QuaConfig=None
):
    """
    Macro pour démoduler la sortie d'un élément en slice.
    TODO: documenter.
    !! doit passe quaconfig=QuaConfig(config) pour changer la longueur des poids d'intégration.

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
