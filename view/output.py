from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QTextEdit, QLineEdit
from style import NEUTRAL_95, PRIMARY_50, ALPHA, NEUTRAL_10, BORDER_RADIUS
from utils import hex_to_rgba

class Output(QWidget):

    def __init__(self):
        super().__init__()

        self.setAutoFillBackground(True)
        output_layout = QVBoxLayout(self)
        self.setLayout(output_layout)
        self.setStyleSheet("""
            margin: 64px 24px 32px 24px;
        """)

        output_container1 = QWidget()
        output_container1.setObjectName('outputContainer1')
        output_container1.setStyleSheet("""
            #outputContainer1 {{
                border-radius: {border_radius}px;
                background-color: {background_color};
            }}
        """.format(
            border_radius=BORDER_RADIUS,
            background_color=hex_to_rgba(PRIMARY_50, ALPHA)
        ))
        output_container1.setAutoFillBackground(True)
        output_container_layout1 = QVBoxLayout(output_container1)
        output_container1.setLayout(output_container_layout1)

        heading1 = QLabel("Result")
        heading1.setObjectName('heading1')
        heading1.setStyleSheet("""
            #heading1 {{
                color: {color};
                font-size: {font_size}px;
                margin: 72px 32px 12px 32px;
            }}
        """.format(
            color=NEUTRAL_95,
            font_size=24
        ))

        self.line_editor1 = QLineEdit()
        self.line_editor1.setReadOnly(True)
        self.line_editor1.setObjectName('textEditor1')
        self.line_editor1.setStyleSheet("""
            #textEditor1 {{
                color: {color};
                font-size: {font_size}px;
                background-color: {background_color};
                margin: 0px 32px 48px 32px;
                padding: 16px 8px 16px 8px;
                border-radius: {border_radius}px
            }}
        """.format(
            color=NEUTRAL_10,
            font_size=20,
            background_color=hex_to_rgba(NEUTRAL_95, 1),
            border_radius=BORDER_RADIUS
        ))

        output_container2 = QWidget()
        output_container2.setObjectName('outputContainer2')
        output_container2.setStyleSheet("""
            #outputContainer2 {{
                border-radius: {border_radius}px;
                background-color: {background_color};
            }}
        """.format(
            border_radius=BORDER_RADIUS,
            background_color=hex_to_rgba(PRIMARY_50, ALPHA)
        ))
        output_container2.setAutoFillBackground(True)
        output_container_layout2 = QVBoxLayout(output_container2)
        output_container2.setLayout(output_container_layout2)

        heading2 = QLabel("Describe")
        heading2.setObjectName('heading2')
        heading2.setStyleSheet("""
            #heading2 {{
                color: {color};
                font-size: {font_size}px;
                margin: 72px 32px 12px 32px;
            }}
        """.format(
            color=NEUTRAL_95,
            font_size=24
        ))

        # output box
        self.text_editor2 = QTextEdit()
        self.text_editor2.setReadOnly(True)
        self.text_editor2.setObjectName('textEditor2')
        self.text_editor2.setStyleSheet("""
            #textEditor2 {{
                color: {color};
                font-size: {font_size}px;
                background-color: {background_color};
                margin: 0px 32px 48px 32px;
                padding: 24px 8px 8px 8px;
                border-radius: {border_radius}px
            }}
        """.format(
            color=NEUTRAL_10,
            font_size=20,
            background_color=hex_to_rgba(NEUTRAL_95, 1),
            border_radius=BORDER_RADIUS
        ))

        output_container_layout1.addWidget(heading1)
        output_container_layout1.addWidget(self.line_editor1)

        output_container_layout2.addWidget(heading2)
        output_container_layout2.addWidget(self.text_editor2)

        output_layout.addWidget(output_container1, 1)
        output_layout.addWidget(output_container2, 3)
        