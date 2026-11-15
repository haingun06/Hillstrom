import re
import sys
from pathlib import Path

from ..core.model_service import predict_customer as predict_engine, simulate as simulate_engine

from PyQt6.QtCore import Qt, QSize, QLocale, QTimer
from PyQt6.QtGui import QIcon, QValidator
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFrame, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QScrollArea, QAbstractSpinBox, QSpinBox,
    QDoubleSpinBox, QSlider, QProgressBar
)

from .theme import T, apply_light_theme, asset_url, qss
from .dialogs import show_error, show_info
from .theme import icon as theme_icon
from .motion import (
    CountUp, ElevatedFrame, HoverGlow, NavButton, NavIndicator, PolishedComboBox, SmoothScroller,
    animate_value, fade_in,
)



def icon(name, tone='emerald'):
    return theme_icon(name, tone)   # dùng chung, tự xử lý khi thiếu file


def label(text, cls='', size=None):
    item = QLabel(text)
    if cls:
        item.setProperty('class', cls)
    if size:
        font = item.font()
        font.setPointSize(size)
        item.setFont(font)
    return item


def button(text, image, style='primary', height=44, tone=None):
    b = NavButton(text) if style == 'nav' else QPushButton(text)
    b.setProperty('kind', style)
    b.setIcon(icon(image, tone or ('light' if style in ('primary', 'nav') else 'emerald')))
    b.setIconSize(QSize(19, 19))
    b.setMinimumHeight(height)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if style == 'primary':
        HoverGlow(b, T['brand600'], rest_alpha=45, hover_alpha=120)
    elif style == 'secondary':
        HoverGlow(b, T['gold500'], rest_alpha=0, hover_alpha=70)
    return b


def card(name):
    # Thẻ nổi có bóng đổ mềm (vẽ tay, không dùng QGraphicsEffect để cuộn mượt).
    w = ElevatedFrame(radius=18)
    w.setObjectName(name)
    return w


def field_title(text):
    t = label(text, 'fieldLabel')
    t.setWordWrap(True)
    return t


class _FriendlySpinMixin:
    """Hành vi nhập liệu thân thiện cho ô số.

    - Không tự bôi chọn giá trị: khi đang sửa chỉ hiện dấu nháy tại vị trí bấm
      (Qt mặc định bôi chọn khi Tab vào ô, khi nhấn Enter và khi tăng/giảm).
    - Gõ tự do trong lúc nhập (kể cả tạm vượt giới hạn, vd. 6 -> 68); khi xác nhận
      (Enter / Tab / bấm ra ngoài) giá trị được đưa về giới hạn gần nhất (68 -> 12)
      thay vì Qt âm thầm chặn phím như trước.
    - Chấp nhận cả dấu chấm và dấu phẩy làm dấu thập phân (locale vi_VN coi
      dấu chấm là phân cách hàng nghìn: gõ 350.5 từng bị hiểu thành 3505).
    - Cuộn chuột khi chưa chọn ô sẽ cuộn trang thay vì âm thầm đổi giá trị.
    """
    _NUMERIC = re.compile(r'^\s*\d*[.,]?\d*\s*$')

    def _normalise(self, text):
        if isinstance(self, QDoubleSpinBox) and self.decimals():
            return text.replace('.', ',')
        return text.replace('.', '').replace(',', '')

    def _core(self, text):
        """Phần số của chuỗi (bỏ tiền tố / hậu tố như ' %')."""
        suffix, prefix = self.suffix(), self.prefix()
        if suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
        elif suffix.strip() and text.rstrip().endswith(suffix.strip()):
            text = text.rstrip()[:-len(suffix.strip())]
        if prefix and text.startswith(prefix):
            text = text[len(prefix):]
        return text.strip()

    def _parse(self, text):
        core = self._core(self._normalise(text)).replace(',', '.')
        if not core or core == '.':
            return None
        try:
            return float(core)
        except ValueError:
            return None

    # ---- trạng thái trống ("--"): giá trị = real_min - 1, hiển thị bằng specialValueText
    def real_min(self):
        return getattr(self, '_real_min', self.minimum())

    def is_empty(self):
        return self.value() < self.real_min()

    def _is_empty_text(self, text):
        return bool(self.specialValueText()) and text.strip() == self.specialValueText()

    def validate(self, text, pos):
        if self._is_empty_text(text):
            # "--" là trạng thái trống; QDoubleSpinBox mặc định hiểu "-" là dấu âm và đổi thành 0.
            return QValidator.State.Acceptable, text, pos
        fixed = self._normalise(text)
        parsed = self._parse(fixed)
        if parsed is not None and parsed < self.real_min():
            return QValidator.State.Intermediate, fixed, pos   # fixup() đưa lên mức tối thiểu
        state, out, new_pos = super().validate(fixed, pos)
        if state == QValidator.State.Invalid and self._NUMERIC.match(self._core(fixed)) \
                and len(self._core(fixed)) <= 12:
            # Cho gõ tiếp; fixup() sẽ đưa về giới hạn khi xác nhận.
            return QValidator.State.Intermediate, fixed, pos
        return state, out, new_pos

    def fixup(self, text):
        v = self._parse(text)
        if v is None:
            if self.specialValueText():
                return self.specialValueText()    # xóa hết số -> ô trở về trống "--"
            return self.textFromValue(self.value()) + self.suffix()
        v = min(max(v, self.real_min()), self.maximum())
        if isinstance(self, QSpinBox):
            v = int(round(v))
        return self.prefix() + self.textFromValue(v) + self.suffix()

    def valueFromText(self, text):
        if self._is_empty_text(text):
            return self.minimum()
        return super().valueFromText(self._normalise(text))

    def _caret_only(self):
        le = self.lineEdit()
        if le.hasSelectedText():
            le.deselect()
            le.setCursorPosition(len(le.text()) - len(self.suffix()))

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.is_empty():
            QTimer.singleShot(0, self.lineEdit().clear)   # ô trống: gõ số mới, không dính "--"
        else:
            QTimer.singleShot(0, self._caret_only)        # Tab vào ô: chỉ hiện dấu nháy

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._caret_only()                      # Qt bôi chọn sau Enter -> bỏ

    def stepBy(self, steps):
        if self.is_empty():
            self.setValue(self.real_min())              # ô trống + / − -> bắt đầu từ mức tối thiểu
        else:
            target = self.value() + steps * self.singleStep()
            self.setValue(min(max(target, self.real_min()), self.maximum()))   # không lùi về "--"
        le = self.lineEdit()
        le.deselect()                               # Qt bôi chọn sau khi tăng/giảm -> bỏ
        if self.hasFocus():
            le.setCursorPosition(len(le.text()) - len(self.suffix()))

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class FriendlySpinBox(_FriendlySpinMixin, QSpinBox):
    pass


class FriendlyDoubleSpinBox(_FriendlySpinMixin, QDoubleSpinBox):
    pass


class NoWheelComboBox(PolishedComboBox):
    """Dropdown cố định vị trí, bo góc (PolishedComboBox) và không đổi lựa chọn
    khi người dùng chỉ đang cuộn trang qua ô."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus() and self.view().isVisible():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelSlider(QSlider):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NumberField(QWidget):
    def __init__(self, value=0, minimum=0, maximum=999999, decimals=0, step=1, suffix='', parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        if decimals:
            self.spin = FriendlyDoubleSpinBox()
            self.spin.setDecimals(decimals)
        else:
            self.spin = FriendlySpinBox()
        self.spin.setLocale(QLocale(QLocale.Language.Vietnamese, QLocale.Country.Vietnam))
        self.spin.setKeyboardTracking(False)
        self.spin.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        # Thêm 1 nấc dưới mức tối thiểu làm trạng thái "trống" (hiển thị "--" sau khi Làm mới).
        self.spin._real_min = minimum
        self.spin.setRange(minimum - 1, maximum)
        self.spin.setSpecialValueText(EMPTY_TEXT)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)
        self.spin.setSuffix(suffix)
        # KHÔNG dùng ButtonSymbols.NoButtons: với Qt 6.12, NoButtons + stylesheet (border/padding)
        # làm ô nhập bên trong co còn 1px -> không thấy chữ, không gõ được.
        # Nút mũi tên mặc định được ẩn bằng stylesheet (width: 0) trong apply_style.
        self.spin.lineEdit().setTextMargins(10, 0, 6, 0)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.spin.setMinimumHeight(42)
        self.spin.setMinimumWidth(80)
        self.spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.spin.setProperty('class', 'numberSpin')
        fmt = (lambda v: f'{v:,.{decimals}f}'.replace(',', '#').replace('.', ',').replace('#', '.'))
        self.spin.setToolTip(
            f'Nhập trực tiếp (dùng dấu phẩy hoặc dấu chấm cho số thập phân) rồi nhấn Enter, '
            f'hoặc dùng nút − / + (bước {fmt(step)}).\nGiới hạn: {fmt(minimum)} – {fmt(maximum)}{suffix}'
        )
        lay.addWidget(self.spin, 1)
        for sign, delta, tip in [('minus', -1, 'Giảm'), ('plus', 1, 'Tăng')]:
            b = QPushButton()
            b.setProperty('kind', 'step')
            b.setIcon(icon(sign))
            b.setIconSize(QSize(14, 14))
            b.setFixedSize(40, 42)
            b.setToolTip(f'{tip} {fmt(step)}{suffix}')
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(400)
            b.setAutoRepeatInterval(70)
            b.clicked.connect(lambda _, d=delta: self._step(d))
            lay.addWidget(b)

    def _step(self, delta):
        # Chốt giá trị đang gõ dở trước khi tăng/giảm.
        self.spin.interpretText()
        self.spin.stepBy(delta)
        # Chỉnh bằng nút − / + thì ô không ở trạng thái soạn thảo: không dấu nháy, không bôi chọn.
        if self.spin.hasFocus():
            self.spin.clearFocus()
        self.spin.lineEdit().deselect()

    def value(self):
        """Giá trị số, hoặc None nếu ô đang trống ("--")."""
        return None if self.spin.is_empty() else self.spin.value()

    def setValue(self, value):
        self.spin.setValue(self.spin.minimum() if value is None else value)

    def clear(self):
        self.setValue(None)


class Field(QWidget):
    def __init__(self, title, control):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)
        lay.addWidget(field_title(title))
        lay.addWidget(control)
        self.control = control
        # Ô thực sự nhận viền đỏ (ô số nằm trong NumberField).
        self.target = control.spin if hasattr(control, 'spin') else control
        # Dòng chữ báo lỗi nhập liệu ngay dưới ô (ẩn khi không có lỗi).
        self.error = label('', 'fieldError')
        self.error.setWordWrap(True)
        self.error.hide()
        lay.addWidget(self.error)
        lay.addStretch(1)   # luôn căn mép trên: ô cạnh ô có dòng báo lỗi không bị lệch xuống

    def set_error(self, text=None):
        self.error.setText(text or '')
        self.error.setVisible(bool(text))
        t = self.target
        if bool(t.property('missing')) != bool(text):
            t.setProperty('missing', bool(text))
            t.style().unpolish(t)
            t.style().polish(t)


EMPTY_TEXT = '--'
EMPTY_CHOICE = '-- Chưa chọn --'


def combo(items):
    w = NoWheelComboBox()
    w.addItems(items)
    w.setPlaceholderText(EMPTY_CHOICE)   # hiển thị khi chưa chọn (currentIndex = -1)
    w.setMinimumHeight(42)
    w.setCursor(Qt.CursorShape.PointingHandCursor)
    return w


def section_header(number, text, hint, image):
    head = QHBoxLayout()
    head.setSpacing(14)
    badge = QLabel()
    badge.setObjectName('sectionBadge')
    badge.setPixmap(icon(image).pixmap(22, 22))
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setFixedSize(44, 44)
    head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
    content = QVBoxLayout()
    content.setSpacing(2)
    title_row = QHBoxLayout()
    title_row.setSpacing(8)
    step = label(number, 'stepPill')
    step.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_row.addWidget(step)
    title_row.addWidget(label(text, 'sectionTitle'))
    title_row.addStretch(1)
    content.addLayout(title_row)
    content.addWidget(label(hint, 'caption'))
    head.addLayout(content, 1)
    return head


class PredictionCard(QFrame):
    def __init__(self, title, image, name, tone='emerald'):
        super().__init__()
        self.setObjectName(name)
        self.setProperty('winner', False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 18)
        lay.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(8)
        img = QLabel()
        img.setPixmap(icon(image, tone).pixmap(18, 18))
        top.addWidget(img)
        top.addWidget(label(title, 'smallHeading'), 1)
        self.badge = label('ĐỀ XUẤT', 'winnerBadge')
        self.badge.setVisible(False)
        top.addWidget(self.badge)
        lay.addLayout(top)
        self.value = label('— %', 'predictionValue')
        lay.addWidget(self.value)
        lay.addWidget(label('Xác suất mua hàng dự đoán', 'caption'))
        lay.addSpacing(2)
        self.progress = QProgressBar()
        self.progress.setRange(0, 10000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.setObjectName('probabilityProgress')
        lay.addWidget(self.progress)
        self.counter = CountUp(self.value, lambda v: f'{v:.2f}%'.replace('.', ','))

    def set_winner(self, on):
        if self.property('winner') == on:
            return
        self.setProperty('winner', on)
        self.badge.setVisible(on)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class CampaignPredictionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.build_ui()
        self.apply_style()
        self.connect_changes()

    def build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(245)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(18, 24, 18, 22)
        sb.setSpacing(8)
        brand = QHBoxLayout()
        brand.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(icon('brand', 'ink').pixmap(26, 26))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(42, 42)
        mark.setObjectName('brandMark')
        brand.addWidget(mark)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        texts.addWidget(label('HILLSTROM', 'brandTitle'))
        texts.addWidget(label('CAMPAIGN STUDIO', 'brandCaption'))
        brand.addLayout(texts)
        sb.addLayout(brand)
        sb.addSpacing(34)
        sb.addWidget(label('WORKSPACE', 'sidebarCaption'))
        sb.addSpacing(4)
        self.nav_buttons = []
        for title, img in [
            ('Dự đoán && mô phỏng', 'prediction'),
            ('Tổng quan chiến dịch', 'overview'),
            ('Khám phá dữ liệu', 'analytics'),
            ('Chất lượng dữ liệu', 'report')
        ]:
            nav = button(title, img, 'nav', 48, 'light')
            nav.setCheckable(True)
            sb.addWidget(nav)
            self.nav_buttons.append(nav)
        self.nav_buttons[0].setChecked(True)
        for b in self.nav_buttons[1:]:
            b.clicked.connect(lambda checked=False: show_info(self, 'Ứng dụng chung', 'Trang này sẽ do nhóm tích hợp trong Chương 6.'))
        # Khối nền trượt theo mục đang chọn (chuyển động thay cho đổi màu tức thì).
        self.nav_indicator = NavIndicator(self.nav_buttons, sidebar)
        sb.addStretch()
        sb.addWidget(label('HILLSTROM  ·  CAMPAIGN ANALYTICS', 'sidebarFooter'))
        # Không hiển thị thông tin kỹ thuật mô hình trên thanh điều hướng.
        root.addWidget(sidebar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName('scroll')
        body = QWidget()
        body.setObjectName('body')
        content = QVBoxLayout(body)
        # Thẻ có vùng bóng đổ 8px nên lề/khoảng cách được trừ tương ứng.
        content.setContentsMargins(20, 18, 20, 20)
        content.setSpacing(6)

        # Banner kiêm tiêu đề trang (trước đây tiêu đề và banner lặp ý nhau, tốn ~90px).
        hero = ElevatedFrame(radius=20, border=False, decor=True,
                             gradient=[(0, T['brand900']), (0.55, T['brand700']), (1, '#0F8F7A')])
        hero.setObjectName('hero')
        hl = QVBoxLayout(hero)
        hl.setContentsMargins(28, 20, 28, 22)
        hl.setSpacing(6)
        hl.addWidget(label('HILLSTROM CAMPAIGN INTELLIGENCE  ·  RA QUYẾT ĐỊNH TIẾP THỊ DỰA TRÊN DỮ LIỆU', 'heroEyebrow'))
        hl.addWidget(label('Phân tích chuyển đổi khách hàng', 'heroTitle'))
        hero_caption = label('Dự đoán khả năng mua hàng, so sánh phương án email và mô phỏng hiệu quả chiến dịch', 'heroCaption')
        hero_caption.setWordWrap(True)
        hl.addWidget(hero_caption)
        hl.addSpacing(6)
        steps = QHBoxLayout()
        steps.setSpacing(8)
        for n, txt in (('1', 'Nhập thông tin khách hàng'), ('2', 'Xem dự đoán & khuyến nghị'), ('3', 'Mô phỏng chiến dịch')):
            steps.addWidget(label(f'{n}   {txt}', 'heroStep'))
        steps.addStretch(1)
        hl.addLayout(steps)
        # Không hiển thị badge phiên bản huấn luyện trong giao diện người dùng.
        content.addWidget(hero)

        # Hai cột (form | kết quả) tự xếp chồng khi cửa sổ hẹp, xem _update_responsive_layout.
        self.columns = QHBoxLayout()
        self.columns.setSpacing(4)
        form_card = self.make_form()
        result_card = self.make_prediction()
        self.columns.addWidget(form_card, 1)
        self.columns.addWidget(result_card, 1)
        content.addLayout(self.columns)
        content.addWidget(self.make_simulation())
        content.addStretch(1)
        scroll.setWidget(body)
        SmoothScroller.register(scroll)
        root.addWidget(scroll, 1)

    def _field(self, key, title, control):
        f = Field(title, control)
        self.form_fields[key] = f
        return f

    def make_form(self):
        self.form_fields = {}
        box = card('customerCard')
        lay = QVBoxLayout(box)
        lay.setContentsMargins(22, 20, 22, 22)
        lay.setSpacing(18)
        lay.addLayout(section_header('01', 'Thông tin khách hàng', 'Đặc điểm và lịch sử mua sắm của khách hàng', 'people'))
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(16)
        self.recency = NumberField(6, 1, 12)
        self.history = NumberField(242.09, 0, 100000, 2, 10)
        self.channel = combo(['Multichannel', 'Web', 'Phone'])
        self.zip = combo(['Urban', 'Suburban', 'Rural'])
        self.mens = combo(['Có', 'Không'])
        self.womens = combo(['Không', 'Có'])
        self.newbie = combo(['Không', 'Có'])
        grid.addWidget(self._field('Recency', 'Recency · Lần mua gần nhất (1–12 tháng)', self.recency), 0, 0, 1, 3)
        grid.addWidget(self._field('History', 'History · Chi tiêu lịch sử (USD)', self.history), 0, 3, 1, 3)
        grid.addWidget(self._field('Channel', 'Channel · Kênh mua hàng', self.channel), 1, 0, 1, 3)
        grid.addWidget(self._field('Zip Code', 'Zip Code · Khu vực', self.zip), 1, 3, 1, 3)
        grid.addWidget(self._field('Mens', 'Mens · Mua hàng nam', self.mens), 2, 0, 1, 2)
        grid.addWidget(self._field('Womens', 'Womens · Mua hàng nữ', self.womens), 2, 2, 1, 2)
        grid.addWidget(self._field('Newbie', 'Newbie · KH mới', self.newbie), 2, 4, 1, 2)
        lay.addLayout(grid)
        lay.addStretch(1)
        actions = QHBoxLayout()
        actions.setSpacing(12)
        predict = button('XEM DỰ ĐOÁN', 'play', 'primary', 48)
        predict.clicked.connect(self.predict_customer)
        reset = button('Làm mới', 'refresh', 'secondary', 48)
        reset.clicked.connect(self.reset_form)
        actions.addWidget(predict, 3)
        actions.addWidget(reset, 2)
        lay.addLayout(actions)
        return box

    def make_prediction(self):
        box = card('predictionCard')
        lay = QVBoxLayout(box)
        lay.setContentsMargins(22, 20, 22, 22)
        lay.setSpacing(16)
        lay.addLayout(section_header('02', 'Dự đoán theo loại email', 'So sánh xác suất mua hàng giữa hai phương án tiếp thị', 'mail'))
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.mens_card = PredictionCard('MENS E-MAIL', 'mail', 'mensResult', 'emerald')
        self.womens_card = PredictionCard('WOMENS E-MAIL', 'mail', 'womensResult', 'indigo')
        cards.addWidget(self.mens_card)
        cards.addWidget(self.womens_card)
        lay.addLayout(cards)
        recomm = QFrame()
        recomm.setObjectName('recommendation')
        rl = QVBoxLayout(recomm)
        rl.setContentsMargins(18, 14, 18, 14)
        rl.setSpacing(6)
        self.recommendation_box = recomm
        rtitle = QHBoxLayout()
        rtitle.setSpacing(8)
        i = QLabel()
        i.setPixmap(icon('sparkle', 'gold').pixmap(18, 18))
        rtitle.addWidget(i)
        rtitle.addWidget(label('KHUYẾN NGHỊ TIẾP THỊ', 'recommendationTitle'), 1)
        rl.addLayout(rtitle)
        self.recommendation_message = label('Chưa có khuyến nghị', 'recommendationMessage')
        self.recommendation_detail = label('Nhấn XEM DỰ ĐOÁN để tính xác suất theo từng loại email.', 'caption')
        self.recommendation_detail.setWordWrap(True)
        self.recommendation_detail.setMinimumHeight(36)
        self.recommendation_detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        rl.addWidget(self.recommendation_message)
        rl.addWidget(self.recommendation_detail)
        lay.addWidget(recomm)
        foot = QHBoxLayout()
        info = QLabel()
        info.setPixmap(icon('info', 'sage').pixmap(16, 16))
        foot.addWidget(info)
        txt = label('Xác suất mua không đồng nghĩa với tác động gia tăng.', 'warning')
        txt.setWordWrap(True)
        foot.addWidget(txt, 1)
        lay.addLayout(foot)
        lay.addStretch(1)
        return box

    def make_simulation(self):
        box = card('simulationCard')
        lay = QVBoxLayout(box)
        lay.setContentsMargins(22, 20, 22, 22)
        lay.setSpacing(18)
        lay.addLayout(section_header('03', 'Mô phỏng chiến dịch', 'Ước tính quy mô tiếp cận, số đơn hàng và lợi nhuận theo kịch bản lựa chọn', 'simulation'))
        self.sim_grid = QGridLayout()
        self.sim_grid.setHorizontalSpacing(17)
        self.sim_grid.setVerticalSpacing(14)
        setup = QFrame()
        setup.setObjectName('scenario')
        sl = QVBoxLayout(setup)
        sl.setContentsMargins(0, 1, 0, 0)
        top = QHBoxLayout()
        top.addWidget(label('TỶ LỆ KHÁCH HÀNG (TOP-K)', 'fieldLabel'))
        self.top_value = label('10%', 'topValue')
        self.top_value.setFixedWidth(64)
        self.top_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.top_value)
        sl.addLayout(top)
        self.top_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.top_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.top_slider.setRange(1, 100)
        self.top_slider.setValue(10)
        self.top_slider.valueChanged.connect(lambda val: self.top_value.setText(f'{val}%'))
        sl.addWidget(self.top_slider)
        hints = QHBoxLayout()
        hints.addWidget(label('1%', 'caption'))
        hints.addStretch()
        hints.addWidget(label('100%', 'caption'))
        sl.addLayout(hints)
        self.cost = NumberField(.05, 0, 10000, 2, .01)
        self.aov = NumberField(116.36, 0, 100000, 2, 5)
        self.margin = NumberField(20, 0, 100, 0, 1, ' %')
        self._sim_setup = setup
        self._sim_fields = [Field('Chi phí/email (USD)', self.cost),
                            Field('Giá trị giao dịch (USD)', self.aov),
                            Field('Biên lợi nhuận (%)', self.margin)]
        self._narrow = None
        self._arrange_simulation(narrow=False)
        lay.addLayout(self.sim_grid)
        kpis = QHBoxLayout()
        kpis.setSpacing(12)
        self.simulation_kpis = {}
        self.simulation_counters = {}
        fmt = self._format_number
        formats = {
            'customersKpi': lambda v: fmt(v, 0),
            'ordersKpi': lambda v: fmt(v, 1),
            'costKpi': lambda v: f'{fmt(v)} USD',
            'profitKpi': lambda v: f'{fmt(v)} USD',
        }
        for title, placeholder, name, image in [
            ('Khách hàng được chọn', '—', 'customersKpi', 'people'),
            ('Đơn hàng kỳ vọng', '—', 'ordersKpi', 'check'),
            ('Chi phí email', '— USD', 'costKpi', 'mail'),
            ('Lợi nhuận kỳ vọng', '— USD', 'profitKpi', 'dollar')
        ]:
            hero_tile = name == 'profitKpi'
            if hero_tile:   # chỉ số chính: nền gradient nổi bật
                tile = ElevatedFrame(radius=14, border=False, shadow=False, decor=True,
                                     gradient=[(0, T['brand900']), (1, T['brand600'])])
            else:
                tile = QFrame()
            tile.setObjectName(name)
            tl = QVBoxLayout(tile)
            tl.setContentsMargins(16, 14, 16, 14)
            tl.setSpacing(6)
            ti = QHBoxLayout()
            ti.setSpacing(8)
            pic = QLabel()
            pic.setObjectName('kpiIconLight' if hero_tile else 'kpiIcon')
            pic.setPixmap(icon(image, 'ink' if hero_tile else 'emerald').pixmap(16, 16))
            pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pic.setFixedSize(28, 28)
            ti.addWidget(pic)
            ti.addWidget(label(title, 'kpiLabelLight' if hero_tile else 'kpiLabel'), 1)
            tl.addLayout(ti)
            kpi_value = label(placeholder, 'kpiValueLight' if hero_tile else 'kpiValue')
            tl.addWidget(kpi_value)
            self.simulation_kpis[name] = kpi_value
            self.simulation_counters[name] = CountUp(kpi_value, formats[name])
            kpis.addWidget(tile, 1)
        lay.addLayout(kpis)
        bottom = QHBoxLayout()
        self.simulation_note = label('Mô phỏng dựa trên dữ liệu lịch sử; không phải uplift hay lợi nhuận tăng thêm.', 'caption')
        self.simulation_note.setWordWrap(True)
        bottom.addWidget(self.simulation_note, 1)
        run = button('CHẠY MÔ PHỎNG', 'simulation', 'primary', 46)
        run.setMinimumWidth(190)
        run.clicked.connect(self.run_simulation)
        bottom.addWidget(run)
        lay.addLayout(bottom)
        return box

    def _arrange_simulation(self, narrow):
        grid = self.sim_grid
        for w in [self._sim_setup, *self._sim_fields]:
            grid.removeWidget(w)
        for c in range(6):
            grid.setColumnStretch(c, 0)
        if narrow:   # thanh trượt một hàng, ba ô thông số ở hàng dưới
            grid.addWidget(self._sim_setup, 0, 0, 1, 3)
            for i, f in enumerate(self._sim_fields):
                grid.addWidget(f, 1, i)
                grid.setColumnStretch(i, 1)
        else:
            grid.addWidget(self._sim_setup, 0, 0)
            grid.setColumnStretch(0, 3)
            for i, f in enumerate(self._sim_fields, 1):
                grid.addWidget(f, 0, i)
                grid.setColumnStretch(i, 2)

    def _update_responsive_layout(self):
        narrow = self.width() - 245 < 1060
        if narrow == self._narrow:
            return
        self._narrow = narrow
        self.columns.setDirection(QHBoxLayout.Direction.TopToBottom if narrow
                                  else QHBoxLayout.Direction.LeftToRight)
        self._arrange_simulation(narrow)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    MSG_EMPTY_NUMBER = 'Vui lòng nhập giá trị.'
    MSG_EMPTY_CHOICE = 'Vui lòng chọn một giá trị.'
    MSG_NO_GENDER = 'Cần chọn “Có” ở Mens hoặc Womens.'

    def _form_errors(self):
        """Kiểm tra form khách hàng; trả về {tên ô: thông báo lỗi}."""
        errors = {}
        for key, field in self.form_fields.items():
            if isinstance(field.control, NumberField):
                if field.control.value() is None:
                    errors[key] = self.MSG_EMPTY_NUMBER
            elif field.control.currentIndex() < 0:
                errors[key] = self.MSG_EMPTY_CHOICE
        if ('Mens' not in errors and 'Womens' not in errors
                and self.mens.currentText() == 'Không' and self.womens.currentText() == 'Không'):
            errors['Mens'] = errors['Womens'] = self.MSG_NO_GENDER
        return errors

    def _clear_field_error(self, key):
        self.form_fields[key].set_error(None)
        if key in ('Mens', 'Womens'):   # lỗi "chọn Có ở Mens hoặc Womens" thuộc về cả hai ô
            for k in ('Mens', 'Womens'):
                if self.form_fields[k].error.text() == self.MSG_NO_GENDER:
                    self.form_fields[k].set_error(None)

    def reset_form(self):
        # Làm mới: ô số hiển thị "--", dropdown hiển thị "-- Chưa chọn --".
        self.recency.clear()
        self.history.clear()
        for box in (self.channel, self.zip, self.mens, self.womens, self.newbie):
            box.setCurrentIndex(-1)
        for field in self.form_fields.values():
            field.set_error(None)
        self._clear_prediction()
        self.recommendation_message.setText('Chưa có khuyến nghị')
        self.recommendation_detail.setText('Nhập đầy đủ thông tin khách hàng rồi nhấn XEM DỰ ĐOÁN.')

    def connect_changes(self):
        """Không để kết quả cũ xuất hiện sau khi đầu vào thay đổi."""
        # Nhập / chọn lại thì bỏ viền đỏ và dòng báo lỗi của ô đó.
        for key, field in self.form_fields.items():
            sig = field.target.valueChanged if isinstance(field.control, NumberField) else field.control.currentIndexChanged
            sig.connect(lambda *_a, k=key: self._clear_field_error(k))
        for control in (self.recency, self.history):
            control.spin.valueChanged.connect(self.invalidate_prediction)
        for control in (self.channel, self.zip, self.mens, self.womens, self.newbie):
            control.currentIndexChanged.connect(self.invalidate_prediction)
        self.top_slider.valueChanged.connect(self.invalidate_simulation)
        for control in (self.cost, self.aov, self.margin):
            control.spin.valueChanged.connect(self.invalidate_simulation)
        for f in self._sim_fields:
            f.target.valueChanged.connect(lambda _v, f=f: f.set_error(None))

    def _clear_prediction(self):
        for c in (self.mens_card, self.womens_card):
            c.counter.reset('— %')
            animate_value(c.progress, 0, 260)
            c.set_winner(False)

    def invalidate_prediction(self, *_args):
        self._clear_prediction()
        self.recommendation_message.setText('Cần cập nhật dự đoán')
        self.recommendation_detail.setText('Thông tin đã thay đổi. Nhấn XEM DỰ ĐOÁN để tính lại kết quả.')

    def invalidate_simulation(self, *_args):
        for name, placeholder in (
            ('customersKpi', '—'), ('ordersKpi', '—'),
            ('costKpi', '— USD'), ('profitKpi', '— USD')
        ):
            self.simulation_counters[name].reset(placeholder)
        self.simulation_note.setText('Thông số đã thay đổi. Nhấn CHẠY MÔ PHỎNG để cập nhật kết quả.')

    def _customer_form(self):
        return {
            'recency': self.recency.value(),
            'history': self.history.value(),
            'channel': self.channel.currentText(),
            'zip_code': self.zip.currentText(),
            'mens': int(self.mens.currentText() == 'Có'),
            'womens': int(self.womens.currentText() == 'Có'),
            'newbie': int(self.newbie.currentText() == 'Có'),
        }

    @staticmethod
    def _format_number(value, digits=2):
        # vi-VN: dấu chấm phân cách hàng nghìn, dấu phẩy phân cách thập phân.
        return f'{value:,.{digits}f}'.replace(',', '#').replace('.', ',').replace('#', '.')

    def _show_system_error(self, title, message, exc):
        # Chỉ lỗi hệ thống (thiếu file mô hình / dữ liệu, lỗi thư viện…) mới hiện popup.
        show_error(self, title, message, f'{type(exc).__name__}: {exc}')

    def predict_customer(self):
        for field in (self.recency, self.history):
            field.spin.interpretText()
        errors = self._form_errors()
        for key, field in self.form_fields.items():
            field.set_error(errors.get(key))
        if errors:
            # Lỗi nhập liệu: chỉ báo tại ô (viền đỏ + dòng chữ), không popup.
            self._clear_prediction()
            self.recommendation_message.setText('Chưa thể dự đoán')
            self.recommendation_detail.setText('Kiểm tra các ô được đánh dấu đỏ ở phần Thông tin khách hàng.')
            return
        try:
            result = predict_engine(self._customer_form())
            pm, pw = float(result['mens']), float(result['womens'])
            # Xác suất thường chỉ ~1% nên thang 0–100% làm thanh gần như trống.
            # Thanh hiển thị so sánh tương đối: phương án cao hơn = đầy thanh.
            top = max(pm, pw)
            for card, p in ((self.mens_card, pm), (self.womens_card, pw)):
                card.counter.to(p * 100)
                animate_value(card.progress, round(p / top * 10000) if top > 0 else 0)
                card.progress.setToolTip('Độ dài thanh so sánh tương đối giữa hai phương án email.')
                card.set_winner(top > 0 and p == top and abs(pm - pw) > 1e-9)
            self.recommendation_message.setText(result['recommendation'])
            fade_in(self.recommendation_box)
            self.recommendation_detail.setText(
                f"Chênh lệch xác suất: {self._format_number(result['difference_pp'])} điểm phần trăm "
                '(xác suất dự đoán, không phải tác động gia tăng của email).'
            )
        except ValueError as exc:
            # Dữ liệu vào không hợp lệ theo mô hình -> báo ngay trong khung khuyến nghị.
            self._clear_prediction()
            self.recommendation_message.setText('Chưa thể dự đoán')
            self.recommendation_detail.setText(str(exc))
        except Exception as exc:
            self.invalidate_prediction()
            self._show_system_error('Không thể dự đoán',
                                    'Đã xảy ra lỗi hệ thống khi chạy mô hình dự đoán. Kiểm tra file '
                                    'models/final_model_FS2.joblib và phiên bản thư viện (chạy bằng run.bat).', exc)

    def run_simulation(self):
        for field in (self.cost, self.aov, self.margin):
            field.spin.interpretText()
        missing = False
        for f in self._sim_fields:
            empty = f.control.value() is None
            f.set_error(self.MSG_EMPTY_NUMBER if empty else None)
            missing = missing or empty
        if missing:
            self.invalidate_simulation()
            self.simulation_note.setText('Kiểm tra các ô được đánh dấu đỏ ở phần thông số mô phỏng.')
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            result = simulate_engine({
                'top_k': self.top_slider.value(),
                'cost': self.cost.value(),
                'aov': self.aov.value(),
                'margin': self.margin.value(),
            })
            for name, key in (('customersKpi', 'selected'), ('ordersKpi', 'expected_orders'),
                              ('costKpi', 'email_cost'), ('profitKpi', 'expected_profit')):
                self.simulation_counters[name].to(result[key])
            self.simulation_note.setText(
                f"Đã chọn {self._format_number(result['selected'], 0)}/{self._format_number(result['n'], 0)} "
                f"khách hàng (Top-{result['top_k']}%). Mỗi hồ sơ sử dụng phương án email có xác suất "
                'mua hàng cao hơn. Kết quả là ước tính theo mô hình và giả định chi phí, '
                'không phản ánh lợi nhuận gia tăng do email.'
            )
        except ValueError as exc:
            self.invalidate_simulation()
            self.simulation_note.setText(f'Chưa thể mô phỏng: {exc}')
        except Exception as exc:
            self.invalidate_simulation()
            QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
            self._show_system_error('Không thể mô phỏng',
                                    'Đã xảy ra lỗi hệ thống khi mô phỏng. Kiểm tra file data/hillstrom.csv, '
                                    'models/final_model_FS2.joblib và phiên bản thư viện (chạy bằng run.bat).', exc)
        finally:
            QApplication.restoreOverrideCursor()

    def apply_style(self):
        style = qss(r"""
        QWidget { font-family: "Segoe UI", Arial; color: @ink; font-size: 12px; }
        /* ---------- sidebar ---------- */
        QFrame#sidebar { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 @brand950, stop:1 @brand900); }
        QLabel#brandMark { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 @gold400, stop:1 @gold500); border-radius: 12px; }
        QLabel[class="brandTitle"] { color: white; font-size: 18px; font-weight: 800; letter-spacing: 1px; }
        QLabel[class="brandCaption"] { color: @gold200; font-size: 10px; letter-spacing: 2px; }
        QLabel[class="sidebarCaption"] { color: @brand300; font-weight: 800; font-size: 10px; letter-spacing: 2px; padding-left: 6px; }
        QLabel[class="sidebarFooter"] { color: rgba(209, 250, 229, 0.45); font-size: 9px; font-weight: 700; letter-spacing: 1px; padding-left: 6px; }
        QPushButton[kind="nav"] { border: 0; background: transparent; color: @brand100; border-radius: 12px; text-align: left; padding: 12px 12px 12px 20px; font-size: 13px; font-weight: 600; }
        QPushButton[kind="nav"]:checked { color: white; font-weight: 800; }
        /* ---------- page ---------- */
        QWidget#body, QScrollArea#scroll, QScrollArea#scroll > QWidget > QWidget { background: @bg; border: none; }
        QLabel[class="heroEyebrow"] { color: @gold400; font-size: 10px; font-weight: 800; letter-spacing: 2px; }
        QLabel[class="heroTitle"] { color: #FFFFFF; font-size: 26px; font-weight: 800; }
        QLabel[class="heroCaption"] { color: @brand100; font-size: 13px; }
        QLabel[class="heroStep"] { color: #FFFFFF; background: rgba(255, 255, 255, 0.13); border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 10px; padding: 3px 12px; font-size: 11px; font-weight: 600; }
        QLabel#sectionBadge { background: @brand50; border: 1px solid @brand100; border-radius: 22px; }
        QLabel[class="stepPill"] { background: @brand700; color: white; border-radius: 9px; padding: 1px 8px; font-size: 11px; font-weight: 800; }
        QLabel[class="sectionTitle"] { color: @brand900; font-weight: 800; font-size: 16px; }
        QLabel[class="caption"] { color: @muted; font-size: 12px; }
        QLabel[class="fieldLabel"] { color: @ink2; font-weight: 700; font-size: 12px; }
        QLabel[class="fieldError"] { color: @danger; font-size: 11px; font-weight: 600; padding-left: 2px; }
        /* ---------- inputs ---------- */
        QComboBox, QSpinBox[class="numberSpin"], QDoubleSpinBox[class="numberSpin"] {
            background: @surface; border: 1px solid @field; border-radius: 10px; padding: 0px 12px;
            min-height: 40px; color: @ink; font-size: 13px; combobox-popup: 0;
            selection-background-color: @brand100; selection-color: @ink; }
        QComboBox:hover, QSpinBox[class="numberSpin"]:hover, QDoubleSpinBox[class="numberSpin"]:hover { border: 1px solid @brand500; background: #FCFFFD; }
        QComboBox:focus, QSpinBox[class="numberSpin"]:focus, QDoubleSpinBox[class="numberSpin"]:focus { border: 2px solid @brand600; padding: 0px 11px; background: #FFFFFF; }
        QComboBox[missing="true"], QSpinBox[missing="true"], QDoubleSpinBox[missing="true"] { border: 2px solid @danger; padding: 0px 11px; background: #FFF7F7; }
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { width: 0px; border: 0; image: none; }
        QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow { image: none; width: 0px; height: 0px; }
        QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; border: none; width: 30px; }
        QComboBox::down-arrow { image: url(CHEVRON_URL); width: 12px; height: 12px; }
        /* Danh sách thả xuống: khung bo góc do PolishedComboBox vẽ, từng mục do ComboItemDelegate vẽ. */
        QComboBox QAbstractItemView { background: transparent; border: none; outline: 0; font-size: 13px; color: @ink; }
        QPushButton[kind="step"] { background: @surface2; border: 1px solid @field; border-radius: 10px; }
        QPushButton[kind="step"]:hover { background: @brand50; border-color: @brand500; }
        QPushButton[kind="step"]:pressed { background: @brand100; }
        /* ---------- buttons ---------- */
        QPushButton[kind="primary"] { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @brand700, stop:1 @brand600); color: white; border: none; border-radius: 12px; font-size: 13px; font-weight: 800; letter-spacing: 1px; padding: 0 20px; }
        QPushButton[kind="primary"]:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @brand800, stop:1 @brand700); }
        QPushButton[kind="primary"]:pressed { background: @brand900; }
        QPushButton[kind="secondary"] { background: @gold50; color: @gold800; border: 1px solid @gold200; border-radius: 12px; font-size: 13px; font-weight: 800; padding: 0 18px; }
        QPushButton[kind="secondary"]:hover { background: @gold100; border-color: @gold400; }
        QPushButton[kind="secondary"]:pressed { background: @gold200; }
        /* ---------- prediction ---------- */
        QFrame#mensResult { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 @brand50, stop:1 #F6FFFA); border: 2px solid @brand100; border-radius: 14px; }
        QFrame#womensResult { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 @indigo50, stop:1 #F8F9FF); border: 2px solid @indigo100; border-radius: 14px; }
        QFrame#mensResult[winner="true"] { border: 2px solid @brand500; }
        QFrame#womensResult[winner="true"] { border: 2px solid @indigo600; }
        QLabel[class="smallHeading"] { color: @brand800; font-size: 11px; font-weight: 800; letter-spacing: 1px; }
        QFrame#womensResult QLabel[class="smallHeading"] { color: @indigo800; }
        QLabel[class="predictionValue"] { color: @brand700; font-size: 32px; font-weight: 800; }
        QFrame#womensResult QLabel[class="predictionValue"] { color: @indigo700; }
        QLabel[class="winnerBadge"] { background: @brand600; color: white; border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 800; letter-spacing: 1px; }
        QFrame#womensResult QLabel[class="winnerBadge"] { background: @indigo600; }
        QProgressBar#probabilityProgress { border: none; background: rgba(4, 120, 87, 0.12); border-radius: 4px; }
        QProgressBar#probabilityProgress::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @brand700, stop:1 @brand500); border-radius: 4px; }
        QFrame#womensResult QProgressBar#probabilityProgress { background: rgba(91, 108, 240, 0.14); }
        QFrame#womensResult QProgressBar#probabilityProgress::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @indigo700, stop:1 @indigo600); }
        QFrame#recommendation { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 @gold50, stop:1 #FFFDF5); border: 1px solid @gold200; border-radius: 14px; }
        QLabel[class="recommendationTitle"] { color: @gold700; font-size: 11px; font-weight: 800; letter-spacing: 1px; }
        QLabel[class="recommendationMessage"] { color: @brand900; font-size: 18px; font-weight: 800; }
        QLabel[class="warning"] { color: @muted; font-size: 11px; }
        /* ---------- simulation ---------- */
        QLabel[class="topValue"] { background: @gold100; border: 1px solid @gold200; border-radius: 11px; color: @gold800; font-weight: 800; font-size: 16px; padding: 3px 0; }
        QSlider { min-height: 28px; }
        QSlider::groove:horizontal { height: 8px; background: @line; border-radius: 4px; }
        QSlider::sub-page:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @brand700, stop:1 @brand500); border-radius: 4px; }
        QSlider::handle:horizontal { background: #FFFFFF; border: 4px solid @brand600; width: 14px; height: 14px; margin: -7px 0; border-radius: 11px; }
        QSlider::handle:horizontal:hover { border-color: @brand700; background: @brand50; }
        QFrame#customersKpi, QFrame#costKpi, QFrame#ordersKpi { background: @surface2; border: 1px solid @line; border-radius: 14px; }
        QLabel#kpiIcon { background: @brand50; border: 1px solid @brand100; border-radius: 14px; }
        QLabel#kpiIconLight { background: @gold400; border-radius: 14px; }
        QLabel[class="kpiLabel"] { color: @muted; font-size: 12px; font-weight: 600; }
        QLabel[class="kpiValue"] { color: @brand900; font-size: 22px; font-weight: 800; }
        QLabel[class="kpiLabelLight"] { color: @brand100; font-size: 12px; font-weight: 600; }
        QLabel[class="kpiValueLight"] { color: white; font-size: 22px; font-weight: 800; }
        QScrollBar:vertical { background: transparent; width: 10px; margin: 4px 2px; border: 0; }
        QScrollBar::handle:vertical { background: rgba(6, 78, 59, 0.22); min-height: 36px; border-radius: 3px; }
        QScrollBar::handle:vertical:hover { background: rgba(6, 78, 59, 0.40); }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        self.setStyleSheet(style.replace('CHEVRON_URL', asset_url('chevron_emerald.png')))


def main():
    app = QApplication(sys.argv)
    apply_light_theme(app)
    window = QMainWindow()
    window.setWindowTitle('Hillstrom | Dự đoán mua hàng & Mô phỏng chiến dịch')
    window.setWindowIcon(icon('brand'))
    window.setCentralWidget(CampaignPredictionPage())
    window.resize(1470, 980)
    window.setMinimumSize(1060, 710)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
