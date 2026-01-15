import time
import traceback
from time import sleep
from typing import Any, Callable, Dict, List
from dataclasses import dataclass
import os 

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import pyqtgraph.exporters
from pandas.core.window.expanding import Literal
from pandas.io.formats.style_render import Sequence
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QWidget,
)

from . import tools


# from Utils import files as uf
from .ScientificSpinBox import PyScientificSpinBox

@dataclass
class Sweep:
    start: float
    stop: float
    step: float
    nbpts: int
    points: List[int]
    element: str
    stickysteps: List[int]

    @staticmethod
    def _make_stickysteps(start, nbpts, step, element=''):
        stickysteps = np.ones(nbpts, dtype=int)*step
        stickysteps[0] = start
        if step < .155e-3:
            print(f"Step {step} probably too low for {element} axis")
        return stickysteps

    @staticmethod
    def from_nbpts(start, stop, nbpts, element):
        points = np.linspace(start, stop, nbpts)
        step = (stop-start)/nbpts
        stickysteps = Sweep._make_stickysteps(start, nbpts, step, element)
        return Sweep(start, stop, step, nbpts, points, element, stickysteps)

    @staticmethod
    def from_step(start, stop, step, element):
        points = np.arange(start, stop, step)
        nbpts = len(points)
        stickysteps = Sweep._make_stickysteps(start, nbpts, step, element)
        return Sweep(start, stop, step, nbpts, points, element, stickysteps)

class VideoModeWindow(QMainWindow):

    @staticmethod
    def from_job(
        job, # qm job
        long_axis: Sweep, 
        short_axis: Sweep,
        out_name: str,
        save_path: str = None,
        **kwargs):
        """
        Make a window from an opx qua job.
        Generate a function for getting out_name in job.results_handle
        The result must be a 2d array.
        Job is must have a pause().
        """
        def get_map(job):
            # Play opx
            # Wait for pause
            # Get result
            handle = job.result_handles.get(out_name)
            job.resume()
            while not job.is_paused() and len(handle) != 0:
                sleep(0.001)
            try:
                res = handle.fetch_all()
            except KeyError:
                return get_map(job)
            return res

        def saveto(data_2d):
            filename = save_path+"%T_"+out_name+".hdf5"
            filename = expand_filename(filename)
            with sweep_file(
                filename,
                [long_axis.element, short_axis.element],
                [long_axis.points, short_axis.points],
                [out_name],
            ) as file:
                file["data"][out_name][...] = data_2d.T
                print(data_2d.shape)

                file.flush()
            print(f"Vm data saved: {filename}")

        return VideoModeWindow(
            dim = 2,
            fn_get = lambda: get_map(job),
            xlabel = long_axis.element,
            ylabel = short_axis.element,
            axes_dict = {
                "x": [long_axis.start, long_axis.stop],
                "y": [short_axis.start, short_axis.stop],
            },
            win_title = f"vm {out_name}",
            saveto_function = saveto if save_path is not None else None,
            **kwargs
        )

    def __init__(
        self,
        fn_get: Callable[..., Any] | None = None,
        dim: Literal[1, 2] = 1,
        show: bool = True,
        play: bool = False,
        take_focus: bool = False,
        sec_between_frame: float = 0.01,
        axes_dict: Dict[Literal["x", "y"], List[float] | List[int]] = {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        },
        xlabel: str | None = None,
        ylabel: str | None = None,
        fn_xshift: Callable[..., Any] | None = None,
        fn_yshift: Callable[..., Any] | None = None,
        # fix_xaxis=False, fix_yaxis=False,
        wrap_at: int = 0,
        wrap_direction: Literal["h", "v"] = "h",
        pause_after_one: bool = False,
        ysweep: "SweepAxis|None" = None,
        xsweep: "SweepAxis|None" = None,
        window_size: Literal[False, "wide", "wider"] = False,
        make_app: bool = True,
        win_title: str = "Video mode",
        saveto_function = None,
    ):
        """
        Opens a window and start a thread that exec and show `fn_get`.

        Parameters
        ----------
        fn_get : FUNCTION
        dim : dim of the array returned by fn_get
        show : BOOL
            show the window by default
        play : BOOL
            pause by default
            if True and take_focus: will not return until pause is pressed
        sec_between_frame : float
            if fn_get is too fast, this should be >0 to not overload the display thread.
        axes_dict: dict
            of the form {'x': [start, stop]} for 1d
            {'x': [start, stop], 'y':[start, stop] for 2d
             OR
            of the form {'x': stop} and it will be interpreted as [0, stop]
        fn_xshift: function
            called when pressing x shift buttons, with arg step
        wrap_at: int
            when dim=1, you can choose to still display an image. 'wrap_at' is the second dimension of this image.
            the shape of get_fn() must be constant.
        wrap_direction: 'v' | 'h'
        pause_after_one: bool, pause after a map has been completed. (== after an image is in the buffer)
        ysweep: give a SweepAxis object and it will set the right ylabel, yaxis, fn_yshift and wrap_at (overridding kw args).
        xsweep: same as ysweep but wrap_direction is set to 'v'. Can only use one or the other.
        window_size: False:default | 'wide' | 'wider'
        Returns
        -------
        None.


        Known issue: pause_after_one is actually "auto_press_pause_after_one_frame"
                     which means it also take one trace of the second frame.
        """
        self.app = None
        if make_app:
            self.app = QApplication([])
        super().__init__()
        self.frame_count = 0
        self._wrap_mode = False
        self.pause_after_one = pause_after_one

        self.xlabel, self.ylabel = xlabel, ylabel
        self.dim = dim

        # sweep object
        if ysweep:
            wrap_at = len(ysweep)
            ylabel = ysweep.label
            fn_yshift = ysweep.shift
            axes_dict["y"] = ysweep.axis
        elif xsweep:
            wrap_at = len(xsweep)
            xlabel = xsweep.label
            fn_xshift = xsweep.shift
            axes_dict["x"] = xsweep.axis
            wrap_direction = "v"

        # VM
        # init a dummy vm
        self.navg: int = 1
        self.data_buffer: List[Any] = []
        self.avg_data = None  # store the current total avg image

        self.x_coord = axes_dict.get("x", [0, 1])
        if isinstance(self.x_coord, (int, float)):
            self.x_coord = [0, self.x_coord]
        self.y_coord = axes_dict.get("y", [0, 1] if wrap_at == 0 else [0, wrap_at])
        if isinstance(self.y_coord, (int, float)):
            self.y_coord = [0, self.y_coord]

        self.fn_xshift = fn_xshift
        self.fn_yshift = fn_yshift

        # for wrapping mode
        if dim == 1 and wrap_at > 0:
            self._wrap_single_img = None  # buffer to store a single full image
            self._wrap_counter = wrap_at
            self._wrap_direction = wrap_direction
            self._wrap_mode = True

        def get_fn_1d_example():
            return np.random.rand(100)

        def get_fn_2d_example():
            return np.random.rand(10, 10)

        if fn_get is None:
            fn_get = [get_fn_1d_example, get_fn_2d_example][dim - 1]
        self.continousGet(fn_get, dim=dim, sec_between_frame=sec_between_frame)
        self.vthread.start()  # start thread, but vm is paused.
        # setting
        self.pause_at_max_avg = False

        # UI
        self.setWindowTitle(win_title)

        splitter = QSplitter()
        self.graph: pg.PlotWidget = pg.PlotWidget()
        self.graph.plotItem.setLabel(axis="bottom", text=xlabel)
        self.graph.plotItem.setLabel(axis="left", text=ylabel)

        self.curve: pg.PlotDataItem = self.graph.plot()
        self.image: pg.ImageItem = pg.ImageItem()
        self.cm: pg.ColorMap = pg.colormap.get("viridis")
        self.image.setColorMap(self.cm)
        self.graph.addItem(self.image)

        self.left = QWidget()
        self.commands = QGridLayout()
        self.btnPlay = QPushButton("Play")
        self.btnPlay.clicked.connect(self.togglePlay)
        self.btnCopy = QPushButton("Copy to clipboard")
        self.btnCopy.clicked.connect(self.copyToClipboard)
        if saveto_function is not None:
            self.btnSave = QPushButton("Save") if saveto_function else None
            self.btnSave.clicked.connect(
                lambda: saveto_function(
                    self.image.image.T if self.dim == 2 else self.curve.getData[1])
                )
        self.spinAvg = QSpinBox()
        self.progress = QProgressBar()
        self.progress.setMaximum(self.navg)
        self.spinAvg.setMinimum(1)
        self.spinAvg.setMaximum(1000)
        self.spinAvg.setValue(1)
        self.spinAvg.valueChanged.connect(self.setNavg)
        self.lblFps = QLabel(". fps")
        n = 0
        self.commands.addWidget(self.btnPlay, n, 0); n+=1
        self.commands.addWidget(self.btnCopy, n, 0); n+=1
        if saveto_function: 
            self.commands.addWidget(self.btnSave, n, 0); n+=1
        self.commands.addWidget(self.spinAvg, n, 0); n+=1
        self.commands.addWidget(self.progress, n, 0); n+=1
        self.btnYminus = QPushButton("y-")
        self.btnYplus = QPushButton("y+")
        self.btnYminus.clicked.connect(lambda: self.yShift(direction=-1))
        self.btnYplus.clicked.connect(lambda: self.yShift(direction=+1))
        self.spinYstep = PyScientificSpinBox()
        self.spinYstep.setValue(0.005)
        if (dim == 2 or self._wrap_mode) and fn_yshift is not None:
            self.commands.addWidget(self.btnYplus, n, 0); n+=1
            self.commands.addWidget(self.spinYstep, n, 0); n+=1
            self.commands.addWidget(self.btnYminus, n, 0); n+=1

        self.commands.addWidget(self.lblFps, n, 0); n+=1
        self.left.setLayout(self.commands)

        self.right = QWidget()
        self.right_layout = QGridLayout()
        self.right_layout.addWidget(self.graph, 0, 0, 1, 3)
        self.btnXminus = QPushButton("x-")
        self.btnXplus = QPushButton("x+")
        self.btnXminus.clicked.connect(lambda: self.xShift(direction=-1))
        self.btnXplus.clicked.connect(lambda: self.xShift(direction=+1))
        self.spinXstep = PyScientificSpinBox()
        self.spinXstep.setValue(0.005)
        if fn_xshift is not None:
            self.right_layout.addWidget(self.btnXminus, 1, 0)
            self.right_layout.addWidget(self.spinXstep, 1, 1)
            self.right_layout.addWidget(self.btnXplus, 1, 2)
        self.right.setLayout(self.right_layout)

        splitter.addWidget(self.left)
        splitter.addWidget(self.right)

        self.setCentralWidget(splitter)

        if show:
            self.show()
        if play:
            self.play()

        if window_size == "wide":
            self.resize(1000, 500)
        elif window_size == "wider":
            self.resize(1000, 200)

        if take_focus:
            while not self.vthread.pause:
                # wait(1)
                sleep(1)
            return

    def closeEvent(self, event) -> None:
        self.stop()
        print("closed")
        self.close()
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        self.event = event
        modifier = event.modifiers()

        if key == QtCore.Qt.Key_S and modifier == QtCore.Qt.ShiftModifier:
            self.saveToNpz()

        elif key == QtCore.Qt.Key_Down or key == QtCore.Qt.Key_S:
            self.yShift(-1)
        elif key == QtCore.Qt.Key_Up or key == QtCore.Qt.Key_W:
            self.yShift(1)
        elif key == QtCore.Qt.Key_Left or key == QtCore.Qt.Key_A:
            self.xShift(-1)
        elif key == QtCore.Qt.Key_Right or key == QtCore.Qt.Key_D:
            self.xShift(1)

    def _doAvg(
        self, data: Sequence[Any], store_in_buffer: bool = True
    ) -> Sequence[Any]:
        # this is called by plot and imgplot.
        # so we average but also do some general stuff

        # buffer
        if store_in_buffer:
            # we get here after each completed frame.
            if self.pause_after_one:
                self.pause()

            # fps
            self.frame_count += 1
            if self.frame_count == 1:  # first frame
                self.t0 = time.time()
            else:
                self.lblFps.setText(
                    str(round(self.frame_count / (time.time() - self.t0), 1)) + " fps"
                )

            self.data_buffer.append(data)
            if len(self.data_buffer) > self.navg:
                self.data_buffer = self.data_buffer[1:]
            self.progress.setValue(len(self.data_buffer))

            avg_data = np.nanmean(self.data_buffer, axis=0)

        else:
            # if len(self.data_buffer) < 1:
            #    avg_data = data
            # else:
            if len(self.data_buffer) < 1:  # first map
                avg_data = data

            elif len(self.data_buffer) == self.navg:
                self.data_buffer[0] = np.where(
                    np.isnan(data), self.data_buffer[0], data
                )
                avg_data = np.nanmean(self.data_buffer, axis=0)

            else:
                avg_data = np.nanmean(
                    np.concatenate((self.data_buffer, [data])), axis=0
                )

        if len(self.data_buffer) == self.navg and self.pause_at_max_avg:
            self.stop()
        return avg_data

    def plot(self, data_1d: Sequence[Any]):
        x_axis = np.linspace(*self.x_coord, len(data_1d))
        self.curve.setData(x_axis, self._doAvg(data_1d))

    def plotToImg(self, data_1d: Sequence[Any]) -> None:
        """called in wrapping_mode instead of 'plot'"""

        # first call ever
        if self._wrap_single_img is None:
            self._wrap_single_img = np.empty((self._wrap_counter, len(data_1d)))
            self._wrap_single_img[:] = np.nan

        nb_line = len(self._wrap_single_img)
        i = self._wrap_counter % nb_line
        is_last_line = i == (nb_line - 1)
        self._wrap_counter = i + 1

        self._wrap_single_img[i] = data_1d
        if is_last_line:
            img = (
                self._wrap_single_img.copy().T
                if self._wrap_direction == "h"
                else self._wrap_single_img.copy()
            )
            self.imgplot(img, store_in_buffer=True)
            self._wrap_single_img = np.empty((self._wrap_counter, len(data_1d)))
            self._wrap_single_img[:] = np.nan
        else:
            img = (
                self._wrap_single_img.T
                if self._wrap_direction == "h"
                else self._wrap_single_img
            )
            self.imgplot(img, store_in_buffer=False)

    def imgplot(self, data_2d: Sequence[Any], store_in_buffer: bool = True) -> None:
        data_2d = self._doAvg(data_2d, store_in_buffer)
        self.image.setImage(data_2d)
        self.image.setRect(
            self.x_coord[0],
            self.y_coord[0],
            self.x_coord[1] - self.x_coord[0],
            self.y_coord[1] - self.y_coord[0],
        )  # x,y,w,h

    def continousGet(
        self,
        get_fn: Callable[..., Any],
        dim: Literal[1, 2],
        sec_between_frame: float = 1.0,
    ):
        """
        run a thread that periodically exec `get_fn`.
        use self.pause to pause.

        Parameters
        ----------
        get_fn : function
            a function that returns a 1d or 2d array, specified in `dim`.

        Returns
        -------
        None.

        """
        self.vthread = VideoThread(get_fn, wait_time=sec_between_frame)

        if self._wrap_mode:
            self.vthread.sig_frameDone.connect(self.plotToImg)
        elif dim == 1:
            self.vthread.sig_frameDone.connect(self.plot)
        elif dim == 2:
            self.vthread.sig_frameDone.connect(self.imgplot)
        self.vthread.start()

    def xShift(self, direction: int):
        # direction: +1|-1
        shift = direction * self.spinXstep.value()
        self.x_coord[0] = self.x_coord[0] + shift
        self.x_coord[1] = self.x_coord[1] + shift
        if self.fn_xshift is not None:
            self.fn_xshift(shift)

    def yShift(self, direction: int):
        # direction: +1|-1
        shift = direction * self.spinYstep.value()
        self.y_coord[0] = self.y_coord[0] + shift
        self.y_coord[1] = self.y_coord[1] + shift
        if self.fn_yshift is not None:
            self.fn_yshift(shift)

    def setNavg(self, val: int):
        self.data_buffer = self.data_buffer[len(self.data_buffer) - val :]
        self.navg = val
        self.progress.setMaximum(val)

    def play(self):
        self.vthread.pause = False
        self.btnPlay.setText("Pause")

    def pause(self):
        self.vthread.pause = True
        self.btnPlay.setText("Play")

    def togglePlay(self):
        self.play() if self.vthread.pause else self.pause()

    def stop(self):
        self.vthread.terminate()
        self.btnPlay.setText("Stopped")
        self.btnPlay.setDisabled(True)

    def copyToClipboard(self):
        if not self.app:
            return
        clipboard = self.app.clipboard()
        exp = pg.exporters.ImageExporter(self.graph.scene())
        p = exp.parameters()
        p["antialias"] = False
        #p['width'] = 1000
        #p['height'] = 1000
        from PyQt5.QtGui import QColor
        p['background'] = QColor(255,255,255)
        exp.export(copy=True)
        clipboard.setImage(exp.png)

    def saveToNpz(self):
        to_save: npt.ArrayLike
        if self.dim == 1:
            if self._wrap_mode:
                to_save = self.image.image.T
            else:
                to_save = self.curve.getData[1]
        else:
            to_save = self.image.image.T

        meta = dict(
            x_axis=self.x_coord,
            y_axis=self.y_coord,
            x_label=self.xlabel,
            y_label=self.ylabel,
        )
        path = "."
        filename = "vm"
        tools.save_to_npz(path, filename, to_save, meta)

    #     uf.saveToNpz(path, filename, to_save, make_date_folder=False, metadata=meta)


from typing import Generic, TypeVar

T = TypeVar("T")


class VideoThread(QThread, Generic[T]):
    sig_frameDone = pyqtSignal(np.ndarray)

    def __init__(self, fn_get_data: Callable[..., T], wait_time: float = 1.0):
        super().__init__()
        self.get = fn_get_data
        self.wait_time = wait_time

        self.pause = True

    def run(self):
        import time

        while True:
            if not self.pause:
                try:
                    data = self.get()
                    if data is not None:
                        self.sig_frameDone.emit(data)
                except Exception as exc:
                    print(traceback.format_exc())
                    print(exc)
                time.sleep(self.wait_time)
            else:
                time.sleep(1)


class SweepAxis:
    """we usually want to sweep an axis in video mode 1d wrapping.
    This class is to make it less verbose.
    args:
        val_list: list of values to sweep
        fn_sweep: function called with the 'val' arg from the val_list at each iteration

    example:

    ```

    p2_sweep = SweepAxis(np.linspace(1.052, 1.058, 101), fn_next=rhP2.set, label='rhP2 level')

    def vmget():
        p2_sweep.next()
        data = ... meas ...
        return data

    vm = VideoModeWindow(fn_get=vmget, dim=1, wrap_at=len(y_read_dict['list']),
                         axes_dict={'y':p2_sweep.axis},
                         ylabel=p2_sweep.label,
                         fn_yshift=p2_sweep.shift)

    ```
    """

    def __init__(
        self,
        val_list: List[float],
        fn_next: Callable[[float], None] = lambda val: print(val),
        label: str = "sweep",
        enable: bool = True,
    ):
        self.current_index = 0
        self.val_list = val_list
        self.axis = [min(val_list), max(val_list)]
        self.label = label
        self.fn_next = fn_next
        self.enable = enable
        if not enable:
            self.axis = [0, len(val_list)]
            self.label = "count"

    def next(self) -> None:
        if not self.enable:
            return
        self.fn_next(self.val_list[self.current_index])
        self.current_index = (self.current_index + 1) % len(self.val_list)

    def shift(self, step: float) -> None:
        if not self.enable:
            return
        self.val_list = np.array(self.val_list) + step

    def __len__(self) -> int:
        return len(self.val_list)


# COPIED FROM IPYNB.
import h5py
import json
import datetime

def h5_dump_dict(grp:h5py.File, **dict_):
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
        except:
            grp.attrs[key] = json.dumps(val, indent=2)

    grp.file.flush() # Note: File.file is file so this work even if grp is a File

def _check_ax_args(ax_names, ax_values):
    if ax_names != []:
        if len(ax_values) != len(ax_names):
            raise(ValueError, f"axs (size {len(ax_values)}) and ax_names (size {len(ax_names)}) must be of same length.")
    else:
        ax_names = [f"ax{idx}" for idx in range(len(ax_values))]
    return ax_names, ax_values

def expand_filename(filename):
    filename = filename.replace("%T", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return filename

def sweep_file(
    filename: str,
    ax_names: list[str] = [],
    ax_values: list[np.ndarray] = [],
    out_names: list[str] = [],
    print_progress_on_flush: bool = True,
    **metadata
):
    """ 
    Crée et retourne le fichier hdf5.
    - out_names: nom des variables de res_handles à sauvegarder.
    - définie une fonction `flush_data(res_handles)` pour ajouter les nouvelles données disponibles.

    structure du fichier
    data:
        attrs: 
            "sweeped_ax_names": ["x", "y", ...]
            "result_data_names": ["out1", "out2", ...]
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
    """
    ax_names, ax_values = _check_ax_args(ax_names, ax_values)
    print(filename)
    f = h5py.File(filename, "w")
    # meta
    f.create_group("meta")
    metadata["VERSION"] = 0.1
    h5_dump_dict(f["meta"], **metadata)
    # data
    f.create_group("data")
    data_grp = f["data"]
    data_grp.attrs["sweeped_ax_names"] = ax_names
    data_grp.attrs["result_data_names"] = out_names
    for idx, (ax, name) in enumerate(zip(ax_values, ax_names)):
        dset = data_grp.create_dataset(name, data=ax)
        dset.attrs["ax_no"] = idx
    for name in out_names:
        data_grp.create_dataset(
            name,
            shape=map(len, ax_values),
            dtype="f",
            fillvalue=None,
        )
    
    f.flush()

    return f


if __name__ == "__main__":
    vm = VideoModeWindow(dim=2, make_app=True)
    vm.app.exec_()
