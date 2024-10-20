from pyqtgraph.Qt import QtGui, QtWidgets

class PlotWindow(QtWidgets.QWidget):
    def __init__(self, plot_func, *args, **kwargs):
        super().__init__()
        self.plot_func = plot_func
        self.args = args
        self.kwargs = kwargs
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        plot_widget = self.plot_func(*self.args, **self.kwargs)
        layout.addWidget(plot_widget)
        self.setLayout(layout)
        self.setWindowTitle('Plot')
        self.show()

def Plot_imagetool(*args):
    app = QtWidgets.QApplication.instance()
    if app is None:
        # if it does not exist then a QApplication is created
        app = QtWidgets.QApplication([])
    for arg in args:
        window = PlotWindow(arg.arpes.plot)
        app.exec_()
        window.close()