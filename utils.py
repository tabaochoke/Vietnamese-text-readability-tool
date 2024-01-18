import json
import os 
import sys

class Describe:
    def __init__(self):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_path, 'assets', 'describe.json')
        with open(path) as file:
            self.text = json.load(file)

    def get_label(self, index):
        return self.text[index]['label']

    def get_describe(self, index):
        return self.text[index]['describe']

### 
def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    return 'rgba({r}, {g}, {b}, {alpha})'.format(r=r, g=g, b=b, alpha=alpha)
