from controller import Controller

import sys
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app_controller = Controller()
    app_controller.run()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()