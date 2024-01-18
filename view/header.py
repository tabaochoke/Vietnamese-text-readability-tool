from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import  QPalette, QColor
from style import PRIMARY_50, ALPHA, NEUTRAL_95
from view.logo import Logo

class Header(QWidget):
    def __init__(self, on_open_file_dialog):
        super().__init__()
        self.on_open_file_dialog = on_open_file_dialog

        self.setAutoFillBackground(True)
        header_palette = self.palette()
        header_color = QColor(PRIMARY_50)
        header_color.setAlphaF(ALPHA)
        header_palette.setColor(QPalette.Window, header_color)
        self.setPalette(header_palette)

        header_layout = QHBoxLayout(self)
        self.setLayout(header_layout)

        logo = Logo()
        logo.setContentsMargins(48, 0, 0, 0)

        name = QLabel("Toolism")
        name.setObjectName('heading')
        name.setContentsMargins(0, 0, 48, 0)
        name.setStyleSheet("""
            #heading {{
                color: {color};
                font-size: {font_size}px;
                font-style: italic;
            }}
        """.format(
            color=NEUTRAL_95,
            font_size=24
        ))

        open_file_button = QPushButton("Open File")
        open_file_button.setObjectName('button')
        open_file_button.setContentsMargins(0, 0, 0, 0)
        open_file_button.setStyleSheet("""
            #button {{
                height: 48px;
                padding-left: 16px;
                padding-right: 16px;
                margin-left: 24px;
                background-color: {background_color};
                color: {color};
                font-size: {font_size}px;
                border-radius: 24px;
            }}
        """.format(
            background_color=PRIMARY_50,
            color="#fff",
            font_size=18
        ))
        open_file_button.clicked.connect(self.on_open_file_dialog)

        header_layout.addWidget(logo)
        header_layout.addWidget(open_file_button)
        header_layout.addStretch()
        header_layout.addWidget(name)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt)")
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
        
            return content
