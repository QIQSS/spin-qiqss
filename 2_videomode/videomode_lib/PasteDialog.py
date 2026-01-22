from PyQt5.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton


class PasteDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paste Text")
        self.resize(400, 300)

        self.text_edit = QTextEdit(self)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        layout.addWidget(close_btn)

    def getText(self):
        return self.text_edit.toPlainText()
