from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QTextEdit
from style import NEUTRAL_95, PRIMARY_50, ALPHA, NEUTRAL_10, BORDER_RADIUS
from utils import hex_to_rgba

class ToolismTextEdit(QTextEdit):
    def __init__(self, on_highlight_callback, on_press_with_highlight_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_highlight_callback = on_highlight_callback
        self.on_press_with_highlight_callback = on_press_with_highlight_callback

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        selected_text = self.textCursor().selectedText()
        if selected_text:
            self.on_highlight_callback(selected_text)
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        self.on_press_with_highlight_callback()

class Input(QWidget):
    def __init__(self, on_text_editor_text_changed, on_text_editor_highlight_text, on_press_with_highlight_callback):
        super().__init__()

        self.setAutoFillBackground(True)
        input_layout = QHBoxLayout(self)
        self.setLayout(input_layout)
        self.setStyleSheet("""
           margin: 64px 24px 32px 24px;
        """)

        input_container = QWidget()
        input_container.setObjectName('inputContainer')
        input_container.setStyleSheet("""
            #inputContainer {{
                border-radius: {border_radius}px;
                background-color: {background_color};
            }}
        """.format(
            border_radius=BORDER_RADIUS,
            background_color=hex_to_rgba(PRIMARY_50, ALPHA)
        ))
        input_container.setAutoFillBackground(True)
        input_container_layout = QVBoxLayout(input_container)
        input_container.setLayout(input_container_layout)

        heading = QLabel("Document")
        heading.setObjectName('heading')
        heading.setStyleSheet("""
            #heading {{
                color: {color};
                font-size: {font_size}px;
                margin: 72px 32px 12px 32px;
            }}
        """.format(
            color=NEUTRAL_95,
            font_size=24
        ))

        # input box
        self.text_editor = ToolismTextEdit(on_text_editor_highlight_text, on_press_with_highlight_callback)
        self.text_editor.setObjectName('textEditor')
        self.text_editor.setObjectName('textEditor')
        self.text_editor.setStyleSheet("""
            #textEditor {{
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
        self.text_editor.textChanged.connect(on_text_editor_text_changed)

        input_container_layout.addWidget(heading, 1)
        input_container_layout.addWidget(self.text_editor, 10)

        input_layout.addWidget(input_container)
