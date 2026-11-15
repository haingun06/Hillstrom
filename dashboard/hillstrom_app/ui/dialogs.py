"""Popup thông báo riêng của ứng dụng (thay QMessageBox mặc định của hệ điều hành).

Chỉ dùng cho thông báo hệ thống (thiếu file mô hình, CSV hỏng, lỗi khi xuất ảnh…) và
thông báo thành công. Lỗi nhập liệu hiển thị ngay tại ô nhập (viền đỏ + dòng chữ), không popup.

    show_error(parent, 'Không thể đọc CSV', 'Thiếu cột bắt buộc: segment')
    show_success(parent, 'Nạp thành công', 'Đã đọc 64.000 dòng.')
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from .motion import ElevatedFrame, HoverGlow
from .theme import MOTION, T, qss

# kind -> (màu nhấn, nền huy hiệu, ký hiệu)
KINDS = {
    'error':   ('#DC2626', '#FEE2E2', '!'),
    'warning': (T['gold600'], T['gold100'], '!'),
    'success': (T['brand600'], T['brand100'], '✓'),
    'info':    (T['indigo600'], T['indigo100'], 'i'),
}


class AppDialog(QDialog):
    """Hộp thoại nổi trên lớp nền mờ phủ cửa sổ chính."""
    CARD_WIDTH = 460

    def __init__(self, parent, kind, title, message, details=None, button_text='Đã hiểu'):
        host = parent.window() if parent is not None else None
        super().__init__(host)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setModal(True)
        self._host = host
        accent, tint, glyph = KINDS.get(kind, KINDS['info'])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.card = ElevatedFrame(radius=18)
        self.card.setObjectName('dialogCard')
        self.card.setFixedWidth(self.CARD_WIDTH)
        row.addWidget(self.card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(24, 22, 24, 20)
        lay.setSpacing(14)
        head = QHBoxLayout()
        head.setSpacing(14)
        badge = QLabel(glyph)
        badge.setObjectName('dialogBadge')
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(40, 40)
        badge.setStyleSheet(f'background: {tint}; color: {accent}; border-radius: 20px; '
                            f'font-size: 19px; font-weight: 900;')
        head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setObjectName('dialogTitle')
        title_lbl.setWordWrap(True)
        texts.addWidget(title_lbl)
        msg = QLabel(message)
        msg.setObjectName('dialogMessage')
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        texts.addWidget(msg)
        if details:
            det = QLabel(details)
            det.setObjectName('dialogDetails')
            det.setWordWrap(True)
            det.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            texts.addWidget(det)
        head.addLayout(texts, 1)
        lay.addLayout(head)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.ok = QPushButton(button_text)
        self.ok.setObjectName('dialogOk')
        self.ok.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok.setMinimumHeight(40)
        self.ok.setMinimumWidth(110)
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        HoverGlow(self.ok, accent, rest_alpha=40, hover_alpha=110)
        buttons.addWidget(self.ok)
        lay.addLayout(buttons)

        self.setStyleSheet(qss(f'''
            QWidget {{ font-family: "Segoe UI", Arial; color: @ink; }}
            QLabel#dialogTitle {{ font-size: 16px; font-weight: 800; color: @ink; }}
            QLabel#dialogMessage {{ font-size: 13px; color: @ink2; }}
            QLabel#dialogDetails {{ font-size: 11px; color: @muted; background: @surface2;
                                    border: 1px solid @line; border-radius: 8px; padding: 8px 10px; }}
            QPushButton#dialogOk {{ background: {accent}; color: white; border: none; border-radius: 11px;
                                    font-size: 13px; font-weight: 800; padding: 0 20px; }}
            QPushButton#dialogOk:hover {{ background: {QColor(accent).darker(115).name()}; }}
            QPushButton#dialogOk:pressed {{ background: {QColor(accent).darker(130).name()}; }}
            QPushButton#dialogOk:focus {{ border: 2px solid {QColor(accent).lighter(160).name()}; }}
        '''))

    # ------------------------------------------------------------- display
    def showEvent(self, e):
        if self._host is not None:
            # phủ đúng vùng cửa sổ chính
            top_left = self._host.mapToGlobal(QPoint(0, 0))
            self.setGeometry(top_left.x(), top_left.y(), self._host.width(), self._host.height())
        super().showEvent(e)
        self.layout().activate()          # cần vị trí thật của thẻ trước khi chạy hiệu ứng trượt
        self.setWindowOpacity(0.0)
        self._fade = QPropertyAnimation(self, b'windowOpacity', self)
        self._fade.setDuration(MOTION['base'])
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.start()
        end = self.card.pos()
        self._slide = QPropertyAnimation(self.card, b'pos', self)
        self._slide.setDuration(MOTION['slow'] - 100)
        self._slide.setStartValue(end + QPoint(0, 14))
        self._slide.setEndValue(end)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._slide.start()
        self.ok.setFocus()

    def paintEvent(self, e):
        p = QPainter(self)
        dim = QColor(4, 40, 29)
        dim.setAlphaF(0.38)
        p.fillRect(self.rect(), dim)
        p.end()


def show_dialog(parent: QWidget | None, kind: str, title: str, message: str, details: str | None = None):
    AppDialog(parent, kind, title, message, details).exec()


def show_error(parent, title, message, details=None):
    show_dialog(parent, 'error', title, message, details)


def show_success(parent, title, message, details=None):
    show_dialog(parent, 'success', title, message, details)


def show_info(parent, title, message, details=None):
    show_dialog(parent, 'info', title, message, details)
