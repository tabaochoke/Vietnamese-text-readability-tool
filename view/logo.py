import os
import sys
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap

class Logo(QLabel):
    def __init__(self):
        super().__init__()
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_path, 'assets', 'logo.png')

        bg_image_pxmap = QPixmap(path)
        self.setPixmap(bg_image_pxmap)
        self.setStyleSheet('padding: 0 10px')
