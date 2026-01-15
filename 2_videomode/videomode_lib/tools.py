import os
import time as stdtime
from dataclasses import dataclass
from typing import Any, Dict, Tuple, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Metadata: TypeAlias = Dict[Any, Any]
Array: TypeAlias = npt.ArrayLike


@dataclass
class _DataType:
    pass


@dataclass
class Unknown(_DataType):
    pass


@dataclass
class Array1d(_DataType):
    x_axis: Array #= np.empty(0)
    x_label: str = "x"
    y_label: str = "y"
    title: str = ""


@dataclass
class Array2d(_DataType):
    x_label: str = "x"
    y_label: str = "y"
    x_range: Tuple[float, float] = (0.0, 1.0)
    y_range: Tuple[float, float] = (0.0, 1.0)
    title: str = ""

    def get_extent(self) -> Tuple[float, float, float, float]:
        return self.x_range + self.y_range


def datetime(format: str = "%Y%m%d", with_time: bool = False) -> str:
    if with_time:
        format += "-%H%M%S"
    return stdtime.strftime(format)


def make_date_directory(path: str) -> str:
    date_str = datetime(with_time=False)
    full_path = os.path.join(path, date_str)
    os.makedirs(full_path, exist_ok=True)
    return full_path + "/"


def make_sweep_filename(path: str = "D:/", filename: str = "sweep"):
    date_dir = make_date_directory(path)
    time = stdtime.strftime("%H%M%S")
    filename = f"{date_dir}{time}-{filename}.txt"
    return filename


def save_to_npz(
    path: str,
    filename: str,
    array: Array,
    metadata: Metadata = {},
    verbose: int = 1,
    type: _DataType = Unknown(),
) -> str:
    if path != "" and not path.endswith(("/", "\\")):
        path += "/"
    fullname = f"{path}{datetime(with_time=True)}_{filename}"

    # embed metadata dict to a numpy array:
    metadata_array = np.array(metadata, dtype=object)
    type_array = np.array(type, dtype=object)

    np.savez_compressed(
        fullname, array=array, metadata_array=metadata_array, type_array=type_array
    )
    if verbose:
        print("Saved file to: " + fullname + ".npz")

    return fullname + ".npz"


def load_npz(file: str) -> Tuple[Array, Metadata, _DataType]:
    load = np.load(file, allow_pickle=True)
    array: Array = load.get("array")
    metadata: Metadata = load.get("metadata_array").item()
    type: _DataType = load.get("type_array").item()
    return array, metadata, type


def make_fig_ax(
    x_label: str = "x", y_label: str = "y", title: str = ""
) -> Tuple[Figure, Axes]:
    fig: Figure = plt.figure()
    ax = cast(Axes, fig.add_axes([0, 0, 1, 1]))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig, ax


def plot_npz(file: str) -> Tuple[Figure, Axes] | Exception:
    array, metadata, type = load_npz(file)

    match type:
        case Array1d(x_label=xl, y_label=yl, title=t, x_axis=xa):
            fig, ax = make_fig_ax(xl, yl, t)
            ax.plot(xa, array)
            return fig, ax
        case Array2d(x_label=xl, y_label=yl, title=t):
            fig, ax = make_fig_ax(xl, yl, t)
            im = ax.imshow(
                array,
                extent=type.get_extent(),
                aspect="auto",
                origin="lower",
                interpolation="none",
            )
            fig.colorbar(im)
            return fig, ax
        case _:
            return TypeError()


# contains: str|List[str] = '',
def files_in(path: str, full_path: bool = True):
    """List all files in the directory"""

    res = list(
        filter(os.path.isfile, map(lambda x: os.path.join(path, x), os.listdir(path)))
    )
    if full_path:
        return res

    return list(map(os.path.basename, res))


def svg2png(svg_path: str):
    import cairosvg

    cairosvg.svg2png(url="input.svg", write_to="output.png", output_width=1920)
