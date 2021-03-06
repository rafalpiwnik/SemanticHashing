import abc

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QGraphicsOpacityEffect

LABEL_FONT_WEIGHT = 75
DEFAULT_PROMPT_OPACITY = 0.25


class EntityWidget:
    def __init__(self):
        super().__init__()
        self.label_font = QFont()
        self.label_font.setBold(False)
        self.label_font.setWeight(LABEL_FONT_WEIGHT)

        self.fieldStyle = 'QLabel { color: black; font: "Segoe UI"; font-size: 14px }'
        self.fieldErrorStyle = 'QLabel { color: red; font: bold "Segoe UI"; font-size: 14px }'
        self.fieldMatchStyle = 'QLabel { color: green; font: bold "Segoe UI"; font-size: 14px }'

        self.fields: list[QLabel] = []
        self.labels = []

        self.is_prompt = False

    @abc.abstractmethod
    def set_fields(self, mi):
        pass

    @abc.abstractmethod
    def set_default_icon(self):
        pass

    @abc.abstractmethod
    def set_inactive_icon(self):
        pass

    @abc.abstractmethod
    def set_bad_icon(self):
        pass

    @abc.abstractmethod
    def mark_mismatch_error(self, kind: str):
        pass

    @abc.abstractmethod
    def reset_state(self):
        pass

    def mark_field_unknown(self, field):
        field.setStyleSheet(self.fieldErrorStyle)
        field.setText("???")

    def mark_not_parsable(self):
        self.set_bad_icon()

        for f in self.fields:
            f.setText("???")
            f.setStyleSheet(self.fieldErrorStyle)

    # Intentional
    def make_prompt_preset(self, opacity=DEFAULT_PROMPT_OPACITY):
        self.is_prompt = True

        op = QGraphicsOpacityEffect(self)
        op.setOpacity(opacity)
        self.setGraphicsEffect(op)
        self.setAutoFillBackground(False)
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        self.set_inactive_icon()

        for f in self.fields:
            f.setText(". . . . . .")
