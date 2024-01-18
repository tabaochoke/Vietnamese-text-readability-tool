import os
import sys
from PyQt5.QtWidgets import QMainWindow, QGraphicsBlurEffect, QLabel, QVBoxLayout, QWidget, QGridLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPalette, QColor
from view.header import Header
from view.input import Input
from view.output import Output

class View(QMainWindow):
    def __init__(self, on_text_editor_text_changed, on_text_editor_highlight_text, on_open_file_dialog, on_press_with_highlight_callback):
        super().__init__()
        
        self.base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        self.setWindowTitle("Blur Example")
        self.setGeometry(100, 100, 400, 300)

        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(10)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Background layout
        path = os.path.join(self.base_path, 'assets', 'background.png')
        bg_image_pxmap = QPixmap(path)
        background_widget = QLabel()
        background_widget.setPixmap(bg_image_pxmap)

        # App layout
        app_widget = QWidget()
        app_widget.setAutoFillBackground(True)
        app_palette = app_widget.palette()
        app_palette_color = QColor('red')
        app_palette_color.setAlphaF(0)
        app_palette.setColor(QPalette.Window, app_palette_color)
        app_widget.setPalette(app_palette)
        app_layout = QVBoxLayout(app_widget)
        app_layout.setSpacing(0)
        app_layout.setContentsMargins(0, 0, 0, 0)
        app_widget.setLayout(app_layout)

        self.header_widget = Header(on_open_file_dialog)

        body_widget = QWidget()
        body_layout = QHBoxLayout(body_widget)
        body_layout.setSpacing(0)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_widget.setLayout(body_layout)

        self.input_widget = Input(on_text_editor_text_changed, on_text_editor_highlight_text, on_press_with_highlight_callback)
        self.output_widget = Output()
        body_layout.addWidget(self.input_widget, 10)
        body_layout.addWidget(self.output_widget, 3)

        app_layout.addWidget(self.header_widget, 1)
        app_layout.addWidget(body_widget, 13)

        main_layout.addWidget(background_widget, 0, 0)
        main_layout.addWidget(app_widget, 0, 0)

        self.showMaximized()
