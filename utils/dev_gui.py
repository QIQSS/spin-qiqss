from PyQt5.QtWidgets import (
    QWidget, 
    QVBoxLayout,
    QLabel, QPushButton,
    QGridLayout,
    QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal

from pyHegel.gui.ScientificSpinBox import PyScientificSpinBox


class DevsGui(QWidget):
    def __init__(self, name_to_device={}):
        super().__init__()
        self.devs = name_to_device

        main_layout = QVBoxLayout()
        grid_layout = QGridLayout() # for rows

        current_row = 0
        for name, dev in self.devs.items():
            name_label = QLabel(name)

            # Create the spin box for the value
            spinbox = PyScientificSpinBox(
                #TODO: value=get
                accelerated=False,
                singleStep=0.001,
            )
            spinbox.setReadOnly(True)

            # increments
            increment_spinbox = PyScientificSpinBox()
            increment_spinbox.setValue(.001)
            btn_plus = QPushButton('+')
            btn_minus = QPushButton('-')
            btn_plus.clicked.connect(lambda _, sb=spinbox, inc_sb=increment_spinbox: sb.setValue(sb.value() + inc_sb.value()))
            btn_minus.clicked.connect(lambda _, sb=spinbox, inc_sb=increment_spinbox: sb.setValue(sb.value() - inc_sb.value()))

            status = QLabel('')
            spinbox.valueChanged.connect(lambda val, dev=dev, status=status: device_set_value(dev, val, status))

            grid_layout.addWidget(name_label, current_row, 0)
            grid_layout.addWidget(spinbox, current_row, 1)
            grid_layout.addWidget(btn_minus, current_row, 2)
            grid_layout.addWidget(increment_spinbox, current_row, 3)
            grid_layout.addWidget(btn_plus, current_row, 4)
            grid_layout.addWidget(status, current_row, 5)

            current_row += 1

        main_layout.addLayout(grid_layout)
        main_layout.addStretch(1)

        copy_btn = QPushButton("Copy dict")
        copy_btn.clicked.connect(self.copy_to_dict)
        main_layout.addWidget(copy_btn)

        self.setLayout(main_layout)

    def copy_to_dict(self):
        d = {}
        for name, dev in self.devs.items():
            try:
                d[name] = device_get_value(dev)
            except Exception as e:
                d[name] = None
        text = repr(d)
        print(text)
        QApplication.clipboard().setText(text)


### UTILS

def device_get_value(device):
    # TODO: check dev is a device
    # on_success = lambda _: print('success ', _)
    # on_fail = lambda er: print('fail ', er)
    # thread = QuickThread.do_in_thread(device.get, on_success=on_success, on_fail=on_fail)
    return device.get()

def device_set_value(device, value, status_label):
    # TODO: check dev is a device
    #print(device)
    set_fn = lambda: device.set(value)
    on_start = lambda: status_label.setText('ramping')
    on_success = lambda: status_label.setText('at value')
    #on_interrupt = lambda: status_label.setText('ramp stopped')
    def on_fail(er):
        print('fail ', er)
        status_label.setText(er)
    thread = QuickThread.do_in_thread(set_fn, 
                                        on_start=on_start, 
                                        on_success=on_success, 
                                        on_fail=on_fail)


class QuickThread(QThread):
    """ A thread object that keeps track of its instances
    Start a thread with:
    QuickThread.do_in_thread(myfunction, on_success, on_fail)
    """ 
    start_signal = pyqtSignal()
    success_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    instance_list = []

    def __init__(self, function):
        super().__init__()
        QuickThread.instance_list.append(self)
        self.function = function

    def run(self):
        try:
            self.start_signal.emit()
            result = self.function()
            self.success_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))
    
    @classmethod
    def do_in_thread(cls, function, 
                    on_start = lambda *args: None,
                    on_success = lambda *args: None,
                    on_fail = lambda *args: None):
        """
        Run the given function in a thread.
        Calls `on_success` with the result if successful, or `on_fail` with the error message if failed.
        """
        thread = cls(function)
        thread.start_signal.connect(on_start)
        thread.success_signal.connect(on_success)
        thread.error_signal.connect(on_fail)
        # remove after finished:
        thread.finished.connect(lambda: QuickThread.instance_list.remove(thread))
        thread.finished.connect(lambda: thread.deleteLater())
        thread.start()


if __name__ == '__main__':
    class fakedev:
        def __init__(self):
            self.set = print

    devs = {"dev2": fakedev(), "dev1": fakedev()}
    gui = DevsGui(devs)

    gui.show()
   
