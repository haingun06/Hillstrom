"""Các thành phần chuyển động / chiều sâu dùng chung cho giao diện.

Nguyên tắc để mượt:
- Bóng đổ của thẻ được tự vẽ (QPainter) thay vì QGraphicsDropShadowEffect:
  effect bắt Qt render lại toàn bộ thẻ ra ảnh tạm mỗi lần vẽ, gây giật khi cuộn
  hoặc khi số liệu đang chạy animation.
- Chuyển trang dùng ảnh chụp trang cũ mờ dần (chỉ 1 QLabel có effect), trang mới
  trượt nhẹ lên — không render lại biểu đồ matplotlib trong lúc animation.
- Mọi thời lượng / easing lấy từ ui_theme.MOTION.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import (
    QAbstractAnimation, QEasingCurve, QEvent, QObject, QPoint, QPropertyAnimation,
    QRect, QRectF, QSize, Qt, QTimer, QVariantAnimation,
)
from PyQt6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath
from PyQt6.QtWidgets import (
    QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox, QFrame,
    QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QLabel, QPushButton,
    QScrollBar, QStyledItemDelegate, QWidget,
)

from .theme import MOTION, T, icon


EASE_OUT = QEasingCurve.Type.OutCubic
EASE_IN_OUT = QEasingCurve.Type.InOutCubic


# ------------------------------------------------------------ elevation ---
class ElevatedFrame(QFrame):
    """Thẻ bo góc có bóng đổ mềm, vẽ tay; tùy chọn nổi lên khi rê chuột.

    Vùng bóng đổ nằm trong contentsMargins của widget nên layout bên trong
    không bị ảnh hưởng (chỉ cần giảm spacing giữa các thẻ tương ứng).
    """
    SHADOW = 8

    def __init__(self, parent=None, *, radius=16, fill=None, border=None,
                 gradient=None, hover_lift=False, decor=False, shadow=True):
        super().__init__(parent)
        self._radius = radius
        self._fill = QColor(fill or T['surface'])
        self._border = QColor(border or T['line']) if border is not False else None
        self._gradient = gradient          # [(pos, '#hex'), ...] theo chiều ngang
        self._decor = decor                # vòng tròn trang trí (banner)
        self._shadow = shadow
        self._lift = 0.0
        if shadow:
            s = self.SHADOW
            self.setContentsMargins(s, s - 2, s, s + 2)
        if hover_lift:
            self._anim = QVariantAnimation(self)
            self._anim.setDuration(MOTION['base'])
            self._anim.setEasingCurve(EASE_OUT)
            self._anim.valueChanged.connect(self._set_lift)
            self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        else:
            self._anim = None

    def _set_lift(self, v):
        self._lift = float(v)
        self.update()

    def _animate_lift(self, target):
        if self._anim is None:
            return
        self._anim.stop()
        self._anim.setStartValue(self._lift)
        self._anim.setEndValue(float(target))
        self._anim.start()

    def enterEvent(self, e):
        self._animate_lift(1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._animate_lift(0.0)
        super().leaveEvent(e)

    def body_rect(self) -> QRectF:
        m = self.contentsMargins()
        return QRectF(self.rect()).adjusted(m.left(), m.top(), -m.right(), -m.bottom())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = self.body_rect()
        r = self._radius
        if self._shadow:
            lift = self._lift
            n = self.SHADOW
            shade = QColor(6, 46, 30)
            base = 0.010 + 0.009 * lift          # đậm hơn khi nổi lên
            dy = 1.5 + 2.0 * lift
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(n, 0, -1):
                shade.setAlphaF(base)
                rect = body.translated(0, dy).adjusted(-i, -i + 1, i, i)
                p.setBrush(shade)
                p.drawRoundedRect(rect, r + i, r + i)
        path = QPainterPath()
        path.addRoundedRect(body, r, r)
        if self._gradient:
            g = QLinearGradient(body.topLeft(), body.topRight())
            for pos, col in self._gradient:
                g.setColorAt(pos, QColor(col))
            p.fillPath(path, g)
        else:
            p.fillPath(path, self._fill)
        if self._decor:
            p.save()
            p.setClipPath(path)
            p.setPen(Qt.PenStyle.NoPen)
            for cx, cy, rad, a in ((0.93, 0.1, 0.55, 0.07), (0.80, 1.05, 0.42, 0.06), (0.99, 0.95, 0.22, 0.09)):
                c = QColor(255, 255, 255)
                c.setAlphaF(a)
                p.setBrush(c)
                h = body.height()
                p.drawEllipse(QRectF(body.left() + body.width() * cx - h * rad,
                                     body.top() + h * cy - h * rad, 2 * h * rad, 2 * h * rad))
            p.restore()
        if self._border is not None:
            pen_col = QColor(self._border)
            if self._lift > 0 and not self._gradient:
                # viền ngả sang màu thương hiệu khi rê chuột
                mix = QColor(T['brand300'])
                t = self._lift * 0.55
                pen_col = QColor(round(pen_col.red() * (1 - t) + mix.red() * t),
                                 round(pen_col.green() * (1 - t) + mix.green() * t),
                                 round(pen_col.blue() * (1 - t) + mix.blue() * t))
            p.setPen(pen_col)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), r, r)
        p.end()


# ---------------------------------------------------------- smooth scroll ---
class SmoothScroller(QObject):
    """Cuộn mượt bằng bánh xe chuột cho các vùng cuộn đã đăng ký.

    Cài 1 lần ở cấp ứng dụng nên cũng bắt được bánh xe khi con trỏ nằm trên
    biểu đồ matplotlib / bảng (vốn "nuốt" sự kiện cuộn).
    Touchpad (pixelDelta) vốn đã mượt nên cuộn trực tiếp.
    """
    _instance = None
    NOTCH_PX = 110

    def __init__(self, app):
        super().__init__(app)
        self.areas = []
        self.anims = {}
        app.installEventFilter(self)

    @classmethod
    def register(cls, area):
        app = QApplication.instance()
        if app is None:
            return
        if cls._instance is None:
            cls._instance = cls(app)
        inst = cls._instance
        if area not in inst.areas:
            inst.areas.append(area)
            area.destroyed.connect(lambda *_: inst._forget(area))

    def _forget(self, area):
        self.areas = [a for a in self.areas if a is not area]

    def _area_for(self, w):
        while w is not None:
            for a in self.areas:
                if a is w:
                    return a
            w = w.parentWidget()
        return None

    def eventFilter(self, obj, ev):
        if ev.type() != QEvent.Type.Wheel or not isinstance(obj, QWidget):
            return False
        # Ô nhập / combobox / thanh trượt đang được chọn thì để chúng tự xử lý bánh xe.
        owner = obj.parentWidget() if isinstance(obj.parentWidget(), QAbstractSpinBox) else obj
        if (isinstance(owner, (QAbstractSpinBox, QComboBox, QAbstractSlider))
                and not isinstance(owner, QScrollBar) and owner.hasFocus()):
            return False
        area = self._area_for(obj)
        if area is None or not area.isVisible():
            return False
        horizontal = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        bar = area.horizontalScrollBar() if horizontal else area.verticalScrollBar()
        if bar.maximum() <= bar.minimum():
            return False
        pix, ang = ev.pixelDelta(), ev.angleDelta()
        if not pix.isNull():
            d = pix.x() if horizontal else pix.y()
            if horizontal and not d:
                d = pix.y()
            self._stop(bar)
            bar.setValue(bar.value() - d)
            return True
        d = ang.x() if horizontal else ang.y()
        if horizontal and not d:
            d = ang.y()
        if not d:
            return False
        self.scroll_by(bar, d / 120.0 * self.NOTCH_PX)
        return True

    def _stop(self, bar):
        anim = self.anims.get(id(bar))
        if anim is not None:
            anim.stop()

    def scroll_by(self, bar, delta_px):
        anim = self.anims.get(id(bar))
        if anim is None:
            anim = QPropertyAnimation(bar, b'value', bar)
            anim.setEasingCurve(EASE_OUT)
            self.anims[id(bar)] = anim
        running = anim.state() == QAbstractAnimation.State.Running
        target = anim.endValue() if running else bar.value()
        target = int(max(bar.minimum(), min(bar.maximum(), target - delta_px)))
        anim.stop()
        anim.setDuration(MOTION['slow'] - 100)
        anim.setStartValue(bar.value())
        anim.setEndValue(target)
        anim.start()


# --------------------------------------------------------- page transition ---
def crossfade(container: QWidget, switch, slide_target=None, distance=16):
    """Chuyển nội dung trong `container` mượt mà.

    switch()          -- hàm thực hiện việc đổi trang / làm mới dữ liệu.
    slide_target()    -- (tùy chọn) trả về widget sẽ trượt nhẹ từ dưới lên.
    """
    if not container.isVisible() or container.width() <= 0:
        switch()
        return
    snap = container.grab()
    switch()
    overlay = QLabel(container)
    overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    overlay.setPixmap(snap)
    overlay.setGeometry(container.rect())
    overlay.show()
    overlay.raise_()
    effect = QGraphicsOpacityEffect(overlay)
    overlay.setGraphicsEffect(effect)
    fade = QPropertyAnimation(effect, b'opacity', overlay)
    fade.setDuration(MOTION['base'])
    fade.setStartValue(1.0)
    fade.setEndValue(0.0)
    fade.setEasingCurve(EASE_IN_OUT)
    fade.finished.connect(overlay.deleteLater)
    fade.start()
    target = slide_target() if callable(slide_target) else slide_target
    if target is not None:
        end = target.pos()
        slide = QPropertyAnimation(target, b'pos', overlay)
        slide.setDuration(MOTION['slow'] - 60)
        slide.setEasingCurve(EASE_OUT)
        slide.setStartValue(end + QPoint(0, distance))
        slide.setEndValue(end)
        slide.start()


def fade_in(widget: QWidget, duration=None, start=0.15):
    """Làm widget hiện dần; gỡ effect khi xong để không tốn chi phí vẽ."""
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(start)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b'opacity', widget)
    anim.setDuration(duration or MOTION['slow'])
    anim.setStartValue(start)
    anim.setEndValue(1.0)
    anim.setEasingCurve(EASE_OUT)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start()
    widget._fade_anim = anim


# --------------------------------------------------------------- numbers ---
class CountUp:
    """Số liệu chạy từ giá trị cũ tới giá trị mới (ease-out)."""

    def __init__(self, label: QLabel, fmt, duration=None):
        self.label = label
        self.fmt = fmt
        self.value = 0.0
        self.anim = QVariantAnimation(label)
        self.anim.setDuration(duration or MOTION['count'])
        self.anim.setEasingCurve(EASE_OUT)
        self.anim.valueChanged.connect(lambda v: self.label.setText(self.fmt(float(v))))

    def to(self, value, start=None):
        value = float(value)
        self.anim.stop()
        self.anim.setStartValue(float(self.value if start is None else start))
        self.anim.setEndValue(value)
        self.value = value
        self.anim.start()

    def reset(self, text):
        self.anim.stop()
        self.value = 0.0
        self.label.setText(text)


def animate_value(widget, value, duration=None, prop=b'value'):
    """Animation cho thuộc tính số nguyên (vd. QProgressBar.value)."""
    anim = getattr(widget, '_value_anim', None)
    if anim is None:
        anim = QPropertyAnimation(widget, prop, widget)
        anim.setEasingCurve(EASE_OUT)
        widget._value_anim = anim
    anim.stop()
    anim.setDuration(duration or MOTION['count'])
    anim.setStartValue(widget.value())
    anim.setEndValue(int(value))
    anim.start()


# -------------------------------------------------------------- buttons ---
class HoverGlow(QObject):
    """Nút 'nổi' lên khi rê chuột: bóng đổ màu lan rộng dần."""

    def __init__(self, button: QPushButton, color: str, rest_alpha=50, hover_alpha=120):
        super().__init__(button)
        self.button = button
        self.color = QColor(color)
        self.rest, self.hover = rest_alpha, hover_alpha
        self.effect = QGraphicsDropShadowEffect(button)
        self.effect.setOffset(0, 2)
        button.setGraphicsEffect(self.effect)
        self.t = 0.0
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(MOTION['base'])
        self.anim.setEasingCurve(EASE_OUT)
        self.anim.valueChanged.connect(self._apply)
        self._apply(0.0)
        button.installEventFilter(self)

    def _apply(self, t):
        self.t = float(t)
        c = QColor(self.color)
        c.setAlpha(round(self.rest + (self.hover - self.rest) * self.t))
        self.effect.setColor(c)
        self.effect.setBlurRadius(8 + 16 * self.t)
        self.effect.setOffset(0, 2 + 3 * self.t)

    def _go(self, target):
        self.anim.stop()
        self.anim.setStartValue(self.t)
        self.anim.setEndValue(float(target))
        self.anim.start()

    def eventFilter(self, obj, ev):
        typ = ev.type()
        if typ == QEvent.Type.Enter and obj.isEnabled():
            self._go(1.0)
        elif typ == QEvent.Type.Leave:
            self._go(0.0)
        elif typ == QEvent.Type.MouseButtonPress:
            self._go(0.35)
        elif typ == QEvent.Type.MouseButtonRelease:
            self._go(1.0 if obj.underMouse() else 0.0)
        return False


class NavButton(QPushButton):
    """Nút điều hướng có nền hover hiện dần (thay cho đổi màu tức thì)."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._h = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(MOTION['base'])
        self._anim.setEasingCurve(EASE_OUT)
        self._anim.valueChanged.connect(self._set_h)

    def _set_h(self, v):
        self._h = float(v)
        self.update()

    def _go(self, target):
        self._anim.stop()
        self._anim.setStartValue(self._h)
        self._anim.setEndValue(float(target))
        self._anim.start()

    def enterEvent(self, e):
        self._go(1.0)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._go(0.0)
        super().leaveEvent(e)

    def paintEvent(self, e):
        if self._h > 0.01 and not self.isChecked():
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            c = QColor(255, 255, 255)
            c.setAlphaF(0.08 * self._h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(self.rect()).adjusted(0, 1, 0, -1), 12, 12)
            p.end()
        super().paintEvent(e)


class NavIndicator(QWidget):
    """Khối nền 'trượt' theo mục điều hướng đang chọn."""

    def __init__(self, buttons, parent):
        super().__init__(parent)
        self.buttons = list(buttons)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.anim = QPropertyAnimation(self, b'geometry', self)
        self.anim.setDuration(MOTION['slow'] - 80)
        self.anim.setEasingCurve(EASE_OUT)
        for b in self.buttons:
            b.toggled.connect(lambda on, b=b: on and self.move_to(b))
        parent.installEventFilter(self)
        self.lower()
        QTimer.singleShot(0, self.snap)

    def move_to(self, button, animate=True):
        target = button.geometry()
        if not animate or not self.isVisible() or self.geometry().height() < 10:
            self.anim.stop()
            self.setGeometry(target)
            self.show()
            self.lower()
            return
        self.anim.stop()
        self.anim.setStartValue(self.geometry())
        self.anim.setEndValue(target)
        self.anim.start()

    def snap(self):
        current = next((b for b in self.buttons if b.isChecked()), None)
        if current is None:
            self.hide()
            return
        if self.anim.state() != QAbstractAnimation.State.Running:
            self.setGeometry(current.geometry())
        self.show()
        self.lower()

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            QTimer.singleShot(0, self.snap)
        return False

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0, 1, 0, -1)
        g = QLinearGradient(r.topLeft(), r.topRight())
        a, b = QColor(255, 255, 255), QColor(255, 255, 255)
        a.setAlphaF(0.16)
        b.setAlphaF(0.07)
        g.setColorAt(0, a)
        g.setColorAt(1, b)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawRoundedRect(r, 12, 12)
        # vạch nhấn vàng bên trái
        p.setBrush(QColor(T['gold400']))
        h = min(22.0, r.height() - 16)
        p.drawRoundedRect(QRectF(r.left() + 5, r.center().y() - h / 2, 4, h), 2, 2)
        p.end()


# -------------------------------------------------------------- combobox ---
def _mix(c1, c2, t):
    a, b = QColor(c1), QColor(c2)
    return QColor(round(a.red() + (b.red() - a.red()) * t),
                  round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


class ComboItemDelegate(QStyledItemDelegate):
    """Vẽ từng mục trong danh sách thả xuống.

    - Giá trị đang chọn: KHÔNG tô nền, chỉ chữ đậm + dấu ✓ bên phải.
    - Mục đang rê chuột / chọn bằng phím mũi tên: "nổi lên" nhẹ (nền sáng,
      bóng đổ mềm, nhích lên 1px), hiện dần trong MOTION['fast'] ms.
    """
    ROW_H = 36

    def __init__(self, combo):
        super().__init__(combo)
        self.combo = combo
        self.active = -1
        self.levels = {}
        self.anims = {}
        self.check = icon('check', 'emerald')

    def reset(self):
        for anim in self.anims.values():
            anim.stop()
        self.levels.clear()
        self.active = -1

    def set_active(self, row):
        if row == self.active:
            return
        old, self.active = self.active, row
        if old >= 0:
            self._animate(old, 0.0)
        if row >= 0:
            self._animate(row, 1.0)

    def _animate(self, row, target):
        anim = self.anims.get(row)
        if anim is None:
            anim = QVariantAnimation(self)
            anim.setDuration(MOTION['fast'])
            anim.setEasingCurve(EASE_OUT)
            anim.valueChanged.connect(lambda v, r=row: self._set_level(r, float(v)))
            self.anims[row] = anim
        anim.stop()
        anim.setStartValue(self.levels.get(row, 0.0))
        anim.setEndValue(float(target))
        anim.start()

    def _set_level(self, row, v):
        self.levels[row] = v
        self.combo.view().viewport().update()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QSize(s.width() + 40, self.ROW_H)

    def paint(self, p, option, index):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(option.rect).adjusted(5, 2, -5, -2)
        t = self.levels.get(index.row(), 0.0)
        if t > 0.001:
            r = r.translated(0, -1.0 * t)
            p.setPen(Qt.PenStyle.NoPen)
            for i, a in ((3, 0.035), (2, 0.05), (1, 0.07)):
                c = QColor(6, 46, 30)
                c.setAlphaF(a * t)
                p.setBrush(c)
                p.drawRoundedRect(r.adjusted(-i + 1, -i + 2, i - 1, i + 1), 8 + i, 8 + i)
            p.setBrush(_mix('#FFFFFF', T['brand50'], t))
            border = QColor(T['brand200'])
            border.setAlphaF(t)
            p.setPen(border)
            p.drawRoundedRect(r, 8, 8)
        current = index.row() == self.combo.currentIndex()
        font = QFont(option.font)
        font.setBold(current)
        p.setFont(font)
        p.setPen(QColor(T['brand900'] if current else T['ink']))
        p.drawText(r.adjusted(12, 0, -34, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                   str(index.data() or ''))
        if current:
            self.check.paint(p, QRect(int(r.right()) - 26, int(r.center().y()) - 8, 16, 16))
        p.restore()


class PolishedComboBox(QComboBox):
    """Combobox với danh sách thả xuống:
    - luôn mở ngay dưới ô (không dịch theo giá trị đang chọn), tự lật lên khi sát mép màn hình;
    - một khung duy nhất bo góc + bóng đổ (cửa sổ ngoài trong suốt, không còn viền vuông);
    - hiện dần khi mở; mục được rê chuột nổi lên nhẹ (ComboItemDelegate).
    """
    SHADOW = 8
    GAP = 4
    RADIUS = 10
    PAD = 5          # khoảng đệm giữa thân khung và danh sách

    def __init__(self, parent=None):
        super().__init__(parent)
        self._delegate = ComboItemDelegate(self)
        self.setItemDelegate(self._delegate)
        view = self.view()
        view.setMouseTracking(True)
        view.entered.connect(lambda idx: self._delegate.set_active(idx.row()))
        view.viewport().installEventFilter(self)
        self._ready = False
        self._opening = False
        self._fade = None
        self._container = None

    def _prepare_popup(self):
        if self._ready:
            return
        container = self.view().window()
        container.setWindowFlags(container.windowFlags() | Qt.WindowType.FramelessWindowHint
                                 | Qt.WindowType.NoDropShadowWindowHint)
        container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if isinstance(container, QFrame):
            container.setFrameShape(QFrame.Shape.NoFrame)
        container.setStyleSheet('background: transparent; border: none;')
        s, pad = self.SHADOW, self.PAD
        if container.layout() is not None:
            container.layout().setContentsMargins(s, s - 2 + pad, s, s + 2 + pad)
        view = self.view()
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.viewport().setAutoFillBackground(False)
        view.selectionModel().currentChanged.connect(
            lambda cur, _prev: None if self._opening else self._delegate.set_active(cur.row()))
        container.installEventFilter(self)
        self._container = container
        self._ready = True

    def showPopup(self):
        self._prepare_popup()
        c = self._container
        c.setWindowOpacity(0.0)
        self._opening = True
        super().showPopup()
        self._opening = False
        self._delegate.reset()          # giá trị đang chọn không được tô khi vừa mở
        # Vị trí cố định: thân khung thẳng mép với ô, cách ô GAP px.
        s = self.SHADOW
        top_left = self.mapToGlobal(QPoint(0, 0))
        # Chiều cao tính lại theo số mục: Qt tính trước khi cộng phần đệm/bóng đổ của khung
        # nên mục cuối bị cắt mất.
        rows = min(self.count(), self.maxVisibleItems())
        row_h = self.view().sizeHintForRow(0) if self.count() else ComboItemDelegate.ROW_H
        m = c.layout().contentsMargins() if c.layout() is not None else None
        h = rows * row_h + ((m.top() + m.bottom()) if m else 2 * s)
        if self.count() > rows:
            self.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        below_y = top_left.y() + self.height() + self.GAP - (s - 2)
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is not None and below_y + h > screen.bottom():
            y = top_left.y() - self.GAP - h + (s + 2)     # lật lên trên ô
        else:
            y = below_y
        c.setGeometry(top_left.x() - s, y, self.width() + 2 * s, h)
        if self._fade is None:
            self._fade = QPropertyAnimation(c, b'windowOpacity', self)
            self._fade.setEasingCurve(EASE_OUT)
        self._fade.stop()
        self._fade.setDuration(MOTION['fast'])
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def hidePopup(self):
        super().hidePopup()
        self._delegate.reset()

    def popup_body_rect(self):
        """Hình chữ nhật thân khung (toạ độ trong cửa sổ popup)."""
        c = self._container
        s = self.SHADOW
        return QRectF(c.rect()).adjusted(s, s - 2, -s, -(s + 2))

    def eventFilter(self, obj, ev):
        if self._ready and obj is self._container and ev.type() == QEvent.Type.Paint:
            p = QPainter(obj)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            body = self.popup_body_rect()
            r = self.RADIUS
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(self.SHADOW, 0, -1):
                c = QColor(6, 46, 30)
                c.setAlphaF(0.016)
                p.setBrush(c)
                p.drawRoundedRect(body.translated(0, 2).adjusted(-i, -i + 1, i, i), r + i, r + i)
            p.setBrush(QColor(T['surface']))
            p.setPen(QColor(T['line2']))
            p.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), r, r)
            p.end()
            return False
        if obj is self.view().viewport() and ev.type() == QEvent.Type.Leave:
            self._delegate.set_active(-1)
        return super().eventFilter(obj, ev)
