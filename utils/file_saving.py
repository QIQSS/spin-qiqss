import datetime
import os
from IPython import get_ipython
import h5py
import json
import numpy as np
import time as timelib

from typing import Callable

def make_path_fn(data_path) -> Callable[[], str]:

    def path_fn():
        date = datetime.datetime.now().strftime("%Y%m%d")
        path = os.path.join(data_path, date)

        if not os.path.exists(path):
            os.mkdir(path)
        return path + os.path.sep
    
    return path_fn
    
def expand_filename(filename) -> str:
    filename = filename.replace("%T", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return filename

def get_cell_content(cell_number=-1):
    return get_ipython().user_ns['In'][cell_number]

def get_file_variables(file, var_to_exclude=list()):
    """
    Retourne l'ensemble des variables d'un fichier importé à l'exception des variables
    commençant par "__" et des variables inclues dans `var_to_exclude`.

    Args:
        file: Fichier importé
        var_to_exclude (list[str]): Liste contenant le nom des variables à ne pas récolter
    
    Returns:
        dict: Dictionnaire contenant les variables du fichier
    """
    variables = dir(file)
    variables = filter(lambda var: not var.startswith("__"), variables)
    variables = filter(lambda var: var not in var_to_exclude, variables)
   
    return {var: file.__dict__[var] for var in variables}

def get_file_code(file):
    """
    Retourne le code d'un fichier importé.
    Args:
        file: Fichier importé
    
    Returns:
        str: Code du fichier
    """
    path = file.__file__
    with open(path, "r", encoding="utf-8") as f:
        txt = "".join(f.readlines())
    return txt


def h5_dump_dict(grp:h5py.File, **dict_) -> None:
    """
    Ajoute des dict en tant qu'attributs dans le group "grp" d'un fichier.
    La fonction essaie d'enregistrer les métadonnées directement. Si l'enregistrement
    échoue, les données sont sérialisées.

    Args:
        grp: h5py.File ou h5py.Group
        dict_: dictionnaire
    """
    for key, val in dict_.items():
        try:
            grp.attrs[key] = val
        except TypeError as e:
            grp.attrs[key] = json.dumps(val, indent=2)

    grp.file.flush() # Note: File.file is file so this work even if grp is a File

def _check_ax_args(ax_names, ax_values):
    """
    S'assure que les longueurs correspondent
    Génère les noms si ax_names est la liste vide.
    Génère un arange(0, val, 1) si un des ax_value est un entier
    """
    if ax_names != []:
        for i, ax_value in enumerate(ax_values):
            if isinstance(ax_value, int):
                ax_values[i] = np.arange(0, ax_value, 1)
        if len(ax_values) != len(ax_names):
            raise(ValueError, f"axs (size {len(ax_values)}) and ax_names (size {len(ax_names)}) must be of same length.")
    else:
        ax_names = [f"ax{idx}" for idx in range(len(ax_values))]
    return ax_names, ax_values


def _flush_from_res_handles(file, res_handles, print_progress=True) -> bool:
    """
    Flush des données d'un job qua en cours vers le fichier
    Parcours les variables
    - file/data/<out_var>
    Mets à jour ces variables en allant les chercher par le même nom dans res_handles:
    - file/data/<out_var> = res_handles.get(out_var).fetch()

    La modification est faite seulement si de nouvelles données sont présentes.

    Retourne:
        False si les donneés ne sont pas entièrement remplies
        True sinon
    """

    is_complete = not res_handles.is_processing()

    memory_dict = file.memory_dict

    # Extraire et écrire les données
    out_names = file["data"].attrs["result_data_names"]
    avg_names = file["data"].attrs["average_data_names"]
    for out_name in out_names:
        handle = res_handles.get(out_name)

        last_n = memory_dict.get(out_name, 0)
        n_available = handle.count_so_far()

        if last_n != n_available:
            slc = slice(last_n, n_available)
            new_data = handle.fetch(slc)["value"]
            memory_dict[out_name] = n_available
            assert file["data"][out_name][slc].shape == new_data.shape, \
            "Problème de tailles. Dans le programme qua, on doit avoir stream.buffer().save_all() pour d=2, stream.save_all() pour d=1"
            file["data"][out_name][slc] = new_data

    for avg_name in avg_names:
        handle = res_handles.get(avg_name)
        if handle.count_so_far() != 0:
            new_data = handle.fetch_all()
            file["data"][avg_name][:] = new_data

    file["meta"].attrs["LAST_CALL_TIME"] = current_time = timelib.time()
    creation_time = file["meta"].attrs["CREATION_TIME"]    
    remaining_time_str = "∞"

    if print_progress:     
        name_0, data_0 = out_names[0], file["data"][out_names[0]][:]

        total_points = data_0.shape[0]
        current_point = file.memory_dict.get(name_0, 0)

        if current_point != 0:
            time_per_point = (current_time - creation_time) / current_point
            remaining_time = (total_points - current_point) * time_per_point
            nb_days = int(remaining_time // (3600*24))
            remaining_time_str = timelib.strftime(r"%Hh%Mm%Ss", timelib.gmtime(remaining_time))
            if nb_days != 0:
                remaining_time_str = f"{nb_days}j{remaining_time_str}"

        print(f"Avancement: {current_point}/{total_points}     |    ETA: {remaining_time_str}     ", end="\r")
  
    file.flush()

    return is_complete


def sweep_file(
    filename: str,
    ax_names: list[str] = [],
    ax_values: list[np.ndarray] = [],
    out_names: list[str] = [],
    avg_names: list[str] = [],
    print_progress_on_flush: bool = True,
    **metadata
) -> h5py.File:
    """ 
    Crée et retourne le fichier hdf5.
    - ax_names: nom des axes de sweep
    - ax_values: vecteurs de valeurs des axe. Peut aussi être un int pour dire: np.arrange(0, int, 1)
    - out_names: nom des variables de res_handles à sauvegarder.
    - avg_names: nom des variables de res_handles qui sont moyennées sur l'ensemble de l'expérience.
    - définie une fonction `flush_data(res_handles)` pour ajouter les nouvelles données disponibles.

    structure du fichier
    data:
        attrs: 
            "sweeped_ax_names": ["x", "y", ...]
            "result_data_names": ["out1", "out2", ...]
            "average_data_names": ["avg1", "avg2", ...]
        x: array
        y: array
        ...
        out1: array  ->  se rempli avec .flush_data(qmjob.res_handles)
        out2: array
        ...
    meta:
        attrs: **metadata


    Clés réservées dans metadata:
    - VERSION
    - CREATION_TIME (0.3+)
    - LAST_CALL_TIME (0.3+)
    """
    ax_names, ax_values = _check_ax_args(ax_names, ax_values)

    f = h5py.File(filename, "w", libver="latest")
    f.swmr_mode = True
    # meta
    f.create_group("meta")
    ## reserved keys
    metadata["VERSION"] = 0.3
    metadata["CREATION_TIME"] = timelib.time()
    metadata["LAST_CALL_TIME"] = timelib.time()
    h5_dump_dict(f["meta"], **metadata)
    ##
    # data
    f.create_group("data")
    data_grp = f["data"]
    data_grp.attrs["sweeped_ax_names"] = ax_names
    data_grp.attrs["result_data_names"] = out_names
    data_grp.attrs["average_data_names"] = avg_names
    for idx, (ax, name) in enumerate(zip(ax_values, ax_names)):
        dset = data_grp.create_dataset(name, data=ax)
        dset.attrs["ax_no"] = idx
    for name in out_names:
        data_grp.create_dataset(
            name,
            shape=list(map(len, ax_values)),
            dtype="f",
            fillvalue=np.nan,
        )
    for name in avg_names:
        data_grp.create_dataset(
            name,
            shape=list(map(len, ax_values[1:])),
            dtype="f",
            fillvalue=np.nan,
        )
    
    setattr(f, "memory_dict", {})
    setattr(f, "flush_data", lambda res_handles: _flush_from_res_handles(f, res_handles, print_progress=print_progress_on_flush))

    f.flush()

    return f

