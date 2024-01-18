from view.view import View
from utils import Describe
from model.model import ReadabilityClassifier

class Controller():
    def __init__(self):
        self.view = View(self.on_text_editor_text_changed, self.on_text_editor_highlight_text, self.on_open_file_dialog, self.on_press_with_highlight_callback)
        self.model = ReadabilityClassifier()
        self.model.predict("chào")
        self.describe = Describe()

    def run(self):
        self.view.show()

    def on_text_editor_text_changed(self):
        content = self.view.input_widget.text_editor.toPlainText()
        if len(content) == 0:
            self.view.output_widget.line_editor1.setText('')
            self.view.output_widget.text_editor2.setText('')

            return

        if len(content) <= 30:
            result = self.model.predict(content)
            self.view.output_widget.line_editor1.setText(self.describe.get_label('0'))
            self.view.output_widget.text_editor2.setText(self.describe.get_describe('0'))

            return

        result = self.model.predict(content)
        self.view.output_widget.line_editor1.setText(self.describe.get_label(result))
        self.view.output_widget.text_editor2.setText(self.describe.get_describe(result))

    def on_text_editor_highlight_text(self, selected_text):
        result = self.model.predict(selected_text)
        self.view.output_widget.line_editor1.setText(self.describe.get_label(result))
        self.view.output_widget.text_editor2.setText(self.describe.get_describe(result))

    def on_press_with_highlight_callback(self):
        self.on_text_editor_text_changed()


    def on_open_file_dialog(self):
        content = self.view.header_widget.open_file_dialog()
        self.view.input_widget.text_editor.setText(content)

        
