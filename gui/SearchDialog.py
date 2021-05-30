import glob
import os
from typing import Union

from PyQt5 import QtWidgets, Qt
from PyQt5.QtCore import QThread, pyqtSlot
from PyQt5.QtWidgets import QTableWidgetItem, QHeaderView

from controllers.SearchWorker import SearchWorker
from gui.designer.Ui_SearchDialog import Ui_SearchDialog


class SearchDialog(QtWidgets.QDialog, Ui_SearchDialog):
    def __init__(self, model_name: str, search_root: Union[str, os.PathLike], num_results: int,
                 target_path: Union[str, os.PathLike],
                 parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.fileTable.setRowCount(num_results)
        self.current_row = 0

        self.thread = QThread()
        self.worker = SearchWorker(model_name, search_root, num_results, target_path)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Signals
        self.worker.status.connect(self.update_status)
        self.worker.progress.connect(self.update_progbar)
        self.worker.files_retrieved.connect(self.add_files)

        self.thread.start()

        # Buttons
        self.finishButton.clicked.connect(self.quit_on_clicked_finish)

    @pyqtSlot()
    def quit_on_clicked_finish(self):
        self.close()

    @pyqtSlot(int)
    def update_progbar(self, value: int):
        self.progbar.setValue(value)

    @pyqtSlot(str)
    def update_status(self, text: str):
        self.statusMessage.setText(text)

    @pyqtSlot(str)
    def add_file(self, path: str):
        file_location = QTableWidgetItem(path)
        file_name = QTableWidgetItem(path.split("\\")[-1])
        self.fileTable.setItem(self.current_row, 0, file_name)
        self.fileTable.setItem(self.current_row, 1, file_location)
        self.current_row += 1

    @pyqtSlot(list, list, list)
    def add_files(self, paths: list, distances: list, codes: list):
        for path, distance, code in zip(paths, distances, codes):
            file_name = QTableWidgetItem(path.split("\\")[-1])
            file_location = QTableWidgetItem(path)
            distance_item = QTableWidgetItem(str(distance))
            code_item = QTableWidgetItem(code)

            self.fileTable.setItem(self.current_row, 0, file_name)
            self.fileTable.setItem(self.current_row, 1, file_location)
            self.fileTable.setItem(self.current_row, 2, distance_item)
            self.fileTable.setItem(self.current_row, 3, code_item)

            self.current_row += 1

        self.fileTable.resizeColumnsToContents()
        self.finishButton.setEnabled(True)
