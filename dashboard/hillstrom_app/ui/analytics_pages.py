
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QEvent, QRect, QSize, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QComboBox, QScrollArea, QStackedWidget,
    QGridLayout, QTableWidget, QTableWidgetItem, QSizePolicy, QHeaderView
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from ..core.data_logic import (
    LABELS, FILTERS, filtered, overview,
    group_stats, read_csv, conversion_by, data_quality
)
from ..paths import DATA_CSV, DATA_DIR
from .theme import MOTION, SERIES, T, apply_light_theme, asset_url, icon, qss
from .dialogs import show_error, show_success
from .motion import CountUp, ElevatedFrame, HoverGlow, PolishedComboBox, SmoothScroller, crossfade

rcParams['font.family'] = ['Segoe UI', 'DejaVu Sans', 'sans-serif']

# Bảng màu biểu đồ (đã kiểm tra phân biệt được với người mù màu, xem ui_theme.SERIES).
# Màu gắn với từng nhóm, không đổi khi lọc: Email nam = emerald, Email nữ = indigo, Đối chứng = cam.
CHART_COLORS = {'single':SERIES['Mens E-Mail'],'line':'#DCE5DA','grid':'#EDF2EC','muted':T['muted'],'ink':T['ink']}
SEGMENT_COLORS = dict(SERIES)
FUNNEL_COLORS = ['#0E9F6E', '#3FBF8F', '#8ED9BA']   # một sắc độ, đậm -> nhạt (không mượn màu nhóm)


PAGE_INFO = [
    ('Tổng quan chiến dịch', 'So sánh ba nhóm thử nghiệm: quy mô, truy cập website, chuyển đổi và chi tiêu'),
    ('Khám phá dữ liệu', 'Tỷ lệ chuyển đổi theo kênh, khu vực, recency và lịch sử chi tiêu của khách hàng'),
    ('Chất lượng dữ liệu', 'Kiểu dữ liệu, giá trị thiếu và số giá trị khác nhau của từng biến'),
]

APP_STYLE = qss('''
QMainWindow, QWidget#Root { background: @bg; }
QWidget#PageBody { background: @bg; }
QWidget#Sidebar { background: @brand900; }
QLabel#Brand { color: white; font-size: 19px; font-weight: 800; }
QLabel#SidebarTag { color: @gold200; font-size: 11px; }
QPushButton#SideButton { text-align: left; background: transparent; color: @brand100; border: 0px; border-radius: 11px; padding: 14px 15px; font-size: 13px; }
QPushButton#SideButton:checked { background: rgba(255,255,255,0.14); color: white; font-weight: bold; }
QPushButton#SideButton:hover { background: rgba(255,255,255,0.08); color: white; }
QLabel#PageTitle { color: @brand900; font-size: 26px; font-weight: 800; }
QLabel#Subtitle { color: @muted; font-size: 13px; }
QLabel#FilterLabel { color: @ink2; font-size: 12px; font-weight: 700; }
QLabel#SourceChip { color: @brand800; background: @brand50; border: 1px solid @brand100; border-radius: 14px; padding: 5px 14px; font-size: 12px; font-weight: 700; }
QLabel#SectionLabel { color: @brand800; font-size: 11px; font-weight: 800; letter-spacing: 2px; padding-left: 8px; }
QPushButton#Primary { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @brand700, stop:1 @brand600); color: white; border: 0px; padding: 0 20px; min-height: 40px; border-radius: 12px; font-size: 13px; font-weight: 800; }
QPushButton#Primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 @brand800, stop:1 @brand700); }
QPushButton#Primary:pressed { background: @brand900; }
QPushButton#Secondary { background: @surface; color: @brand800; border: 1px solid @field; padding: 0 18px; min-height: 38px; border-radius: 12px; font-size: 13px; font-weight: 800; }
QPushButton#Secondary:hover { background: @brand50; border-color: @brand500; }
QPushButton#Secondary:pressed { background: @brand100; }
QComboBox { combobox-popup: 0; background: @surface; color: @ink; padding: 0px 12px; border: 1px solid @field; border-radius: 10px; min-width: 190px; min-height: 38px; font-size: 13px; }
QComboBox:hover { border-color: @brand500; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; border: none; width: 30px; }
QComboBox::down-arrow { image: url(CHEVRON_URL); width: 12px; height: 12px; }
QComboBox QAbstractItemView { background: transparent; border: none; outline: 0; font-size: 13px; color: @ink; }
QComboBox:disabled { color: #A0A99C; background: @surface2; border-color: @line; }
QLabel#KpiCaption { color: @muted; font-size: 12px; font-weight: 700; }
QLabel#KpiValue { color: @brand900; font-size: 27px; font-weight: 800; }
QLabel#KpiCaptionLight { color: @brand100; font-size: 12px; font-weight: 700; }
QLabel#KpiValueLight { color: white; font-size: 27px; font-weight: 800; }
QLabel#NoteLight { color: @brand200; font-size: 12px; }
QLabel#KpiIcon { background: @brand50; border: 1px solid @brand100; border-radius: 16px; }
QLabel#KpiIconLight { background: @gold400; border-radius: 16px; }
QLabel#PlotTitle { color: @ink; font-size: 14px; font-weight: 800; }
QLabel#Note { color: @muted; font-size: 12px; }
QTableWidget { background: transparent; border: 0; gridline-color: @line; alternate-background-color: @surface2; selection-background-color: @brand50; selection-color: @brand900; font-size: 13px; }
QTableWidget::item { padding: 0 12px; border-bottom: 1px solid #F0F4EF; }
QHeaderView { background: transparent; }
QHeaderView::section { background: @brand50; padding: 11px 12px; border: 0px; border-bottom: 1px solid @brand100; color: @brand900; font-weight: 800; font-size: 12px; }
QHeaderView::section:first { border-top-left-radius: 12px; }
QHeaderView::section:last { border-top-right-radius: 12px; }
QTableCornerButton::section { background: @brand50; border: 0; }
QScrollArea { border: 0; background: @bg; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 4px 2px; border: 0; }
QScrollBar::handle:vertical { background: rgba(6, 78, 59, 0.22); min-height: 36px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: rgba(6, 78, 59, 0.40); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px 4px; border: 0; }
QScrollBar::handle:horizontal { background: rgba(6, 78, 59, 0.22); min-width: 36px; border-radius: 3px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
''').replace('CHEVRON_URL', asset_url('chevron_emerald.png'))


class DragScrollArea(QScrollArea):
    """Scroll by scrollbar, mouse wheel, or left-drag on non-interactive content."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._drag_last = None
        self._dragging = False
        self.viewport().installEventFilter(self)
        # Cuộn mượt (cài ở cấp ứng dụng nên vẫn hoạt động khi con trỏ nằm trên
        # biểu đồ matplotlib / bảng — vốn nuốt sự kiện cuộn).
        SmoothScroller.register(self)

    def setWidget(self, widget):
        super().setWidget(widget)
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            if not isinstance(child, (QPushButton, QComboBox, QTableWidget, FigureCanvas, QScrollArea)):
                child.installEventFilter(self)

    def eventFilter(self, obj, event):
        typ = event.type()
        if typ == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._drag_last = event.globalPosition().toPoint()
            self._dragging = True
            return True
        if typ == QEvent.Type.MouseMove and self._dragging and self._drag_last is not None:
            pos = event.globalPosition().toPoint()
            delta = pos - self._drag_last
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self._drag_last = pos
            return True
        if typ == QEvent.Type.MouseButtonRelease and self._dragging:
            self._dragging = False
            self._drag_last = None
            return True
        return super().eventFilter(obj, event)


def fmt_int(v: int) -> str:
    return f'{v:,}'.replace(',', '.')


def fmt_num(v: float, digits: int = 2) -> str:
    """Định dạng kiểu Việt Nam: 1.234,56"""
    return f'{v:,.{digits}f}'.replace(',', '#').replace('.', ',').replace('#', '.')


def fmt_pct(v: float, digits: int = 2) -> str:
    return fmt_num(v, digits) + '%'


def _axis_fmt(percent=False, money=False):
    def f(v, _pos):
        if percent:
            txt = fmt_num(v, 1)
            return (txt[:-2] if txt.endswith(',0') else txt) + '%'
        if money:
            return '$' + fmt_num(v, 2 if abs(v) < 10 else 0)
        return fmt_int(int(round(v)))
    return FuncFormatter(f)


def label(txt: str, name: str | None=None, wrap=False):
    w=QLabel(txt)
    if name: w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


class AutoHeightLabel(QLabel):
    """Nhãn xuống dòng trong vùng cuộn: chiều cao luôn vừa đúng số dòng chữ.

    QScrollArea không tự tính chiều cao theo bề rộng cho nhãn xuống dòng, nên trước đây phải
    giữ chỗ cố định 2 dòng (thừa khoảng trống khi chỉ có 1 dòng, cắt chữ khi có 3 dòng).
    """
    def _fit(self):
        # Tự đo bằng QFontMetrics: QLabel.heightForWidth() trả về ~2 dòng kể cả khi chữ chỉ có 1 dòng.
        m=self.contentsMargins()
        avail=max(1,self.width()-m.left()-m.right())
        text_h=self.fontMetrics().boundingRect(QRect(0,0,avail,100000),
                                               int(Qt.TextFlag.TextWordWrap),self.text()).height()
        h=text_h+m.top()+m.bottom()+2
        if h!=self.minimumHeight():self.setMinimumHeight(h);self.setMaximumHeight(h)
    def resizeEvent(self, e):
        super().resizeEvent(e);self._fit()
    def setText(self, text):
        super().setText(text);self._fit()


def wrap_label(txt: str, name: str):
    w=AutoHeightLabel(txt);w.setObjectName(name);w.setWordWrap(True)
    w.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    return w


class KpiCard(ElevatedFrame):
    """Thẻ chỉ số; `featured=True` là chỉ số chính (nền gradient). Số liệu chạy (count-up)."""
    def __init__(self, title, image='people', value='—', note='', featured=False):
        if featured:
            super().__init__(radius=16, border=False, decor=True, hover_lift=True,
                             gradient=[(0, T['brand900']), (1, T['brand600'])])
        else:
            super().__init__(radius=16, hover_lift=True)
        suffix='Light' if featured else ''
        box=QVBoxLayout(self);box.setContentsMargins(18,15,18,15);box.setSpacing(6)
        head=QHBoxLayout();head.setSpacing(10)
        badge=QLabel();badge.setObjectName('KpiIcon'+suffix);badge.setFixedSize(32,32)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setPixmap(icon(image,'ink' if featured else 'emerald').pixmap(17,17))
        self.heading=label(title,'KpiCaption'+suffix)
        head.addWidget(badge);head.addWidget(self.heading,1)
        self.number=label(value,'KpiValue'+suffix);self.note=label(note,'Note'+suffix)
        box.addLayout(head);box.addWidget(self.number);box.addWidget(self.note)
        self.setMinimumHeight(132)
        self.counter=CountUp(self.number, str)
    def update_data(self, number, note='', fmt=None):
        if fmt is None:
            self.counter.reset(str(number))
        else:
            self.counter.fmt=fmt
            self.counter.to(number)
        self.note.setText(note)


class ResponsiveRow(QWidget):
    """Một hàng thẻ; khi hẹp hơn `breakpoint` px tự chuyển thành lưới 2 cột."""
    def __init__(self, widgets, breakpoint=1000):
        super().__init__()
        self.widgets=list(widgets);self.breakpoint=breakpoint;self._cols=None
        self.grid=QGridLayout(self);self.grid.setContentsMargins(0,0,0,0);self.grid.setSpacing(0)
        self._arrange(len(self.widgets))
    def _arrange(self, cols):
        if cols==self._cols:return
        self._cols=cols
        for w in self.widgets:self.grid.removeWidget(w)
        for c in range(len(self.widgets)):self.grid.setColumnStretch(c,0)
        for i,w in enumerate(self.widgets):self.grid.addWidget(w,i//cols,i%cols)
        for c in range(cols):self.grid.setColumnStretch(c,1)
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._arrange(len(self.widgets) if self.width()>=self.breakpoint else 2)


class PlotCard(ElevatedFrame):
    def __init__(self, title, detail=''):
        super().__init__(radius=16, hover_lift=True)
        self.setObjectName('PlotCard')
        v=QVBoxLayout(self);v.setContentsMargins(18,16,18,10);v.setSpacing(3)
        v.addWidget(label(title,'PlotTitle'))
        v.addWidget(label(detail,'Note'))
        self.fig=Figure(figsize=(5,2.95), dpi=100, facecolor='white')
        # Tự căn lề lại mỗi lần vẽ (kể cả khi đổi kích thước cửa sổ) để nhãn trục không bị cắt.
        self.fig.set_layout_engine('tight', pad=1.45)
        self.canvas=FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(238)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        v.addWidget(self.canvas,stretch=1)
        self.ax=self.fig.add_subplot(111)
    def start(self):
        self.ax.clear(); self.ax.set_facecolor('white')
        self.ax.spines[['top','right','left']].set_visible(False)
        self.ax.spines['bottom'].set_color(CHART_COLORS['line'])
        self.ax.tick_params(axis='both',labelsize=9,length=0,pad=7,colors=CHART_COLORS['muted'])
        self.ax.tick_params(axis='x',labelcolor=CHART_COLORS['ink'])
        self.ax.grid(axis='y',color=CHART_COLORS['grid'],linewidth=.8,zorder=0)
        self.ax.set_axisbelow(True)
    def finish(self):
        self.canvas.draw_idle()
    def empty(self, msg='Mở CSV để xem biểu đồ chi tiết'):
        self.start();self.ax.text(.5,.52,msg,ha='center',va='center',transform=self.ax.transAxes,
                                color=CHART_COLORS['muted'],fontsize=10,wrap=True)
        self.ax.set_xticks([]);self.ax.set_yticks([]);self.finish()
    def bars(self, names, values, colors=None, percent=False, horizontal=False, money=False):
        self.start()
        if not names:
            self.empty('Không có dữ liệu phù hợp');return
        colors=colors or [CHART_COLORS['single']]*len(names)
        def text(v):
            if percent: return fmt_pct(v)
            if money: return '$' + fmt_num(v)
            return fmt_int(round(v))
        high=max(values or [0])
        if horizontal:
            coords=np.arange(len(names)); bars=self.ax.barh(coords,values,color=colors,height=.52,zorder=3)
            self.ax.set_yticks(coords,names);self.ax.invert_yaxis()
            self.ax.tick_params(axis='y',labelcolor=CHART_COLORS['ink'])
            self.ax.grid(axis='x',color=CHART_COLORS['grid'],linewidth=.8);self.ax.grid(axis='y',visible=False)
            self.ax.set_xlim(0,max(high*1.23,.5))
            self.ax.xaxis.set_major_formatter(_axis_fmt(percent, money))
            for b,v in zip(bars,values):
                self.ax.text(b.get_width()+max(high*.02,.02),b.get_y()+b.get_height()/2,text(v),
                             va='center',fontsize=9,color=CHART_COLORS['ink'],fontweight='semibold')
        else:
            bars=self.ax.bar(names,values,color=colors,width=.52,zorder=3)
            self.ax.set_ylim(0,max(high*1.19,.5))
            self.ax.yaxis.set_major_formatter(_axis_fmt(percent, money))
            for b,v in zip(bars,values):
                self.ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(high*.025,.02),text(v),
                             ha='center',fontsize=9,color=CHART_COLORS['ink'],fontweight='semibold')
        self.finish()
    def line(self, names, values):
        self.start()
        if not names: self.empty('Không có dữ liệu phù hợp');return
        x=np.arange(len(names));self.ax.plot(x,values,color=CHART_COLORS['single'],marker='o',lw=2,markersize=5,zorder=3)
        self.ax.fill_between(x,values,alpha=.12,color=CHART_COLORS['single'])
        self.ax.set_xticks(x,[str(y) for y in names],fontsize=9)
        self.ax.set_xlabel('Số tháng kể từ lần mua gần nhất',fontsize=9,color=CHART_COLORS['muted'])
        self.ax.yaxis.set_major_formatter(_axis_fmt(percent=True))
        self.ax.set_ylim(bottom=0,top=max(values)*1.3 if max(values)>0 else 1)
        self.finish()


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hillstrom | Campaign Studio')
        self.setMinimumSize(1080,740)
        self.resize(1380,900)
        self.df=None;self.csv_path=None
        # Vẽ biểu đồ "lười": trang đang xem vẽ ngay, trang ẩn vẽ dần lúc rảnh
        # (mỗi lần 1 biểu đồ) để đổi bộ lọc / chuyển trang không bị đứng hình.
        self._jobs={}
        self._quality_for=None
        self._history_bin=None;self._history_for=None
        self._idle_timer=QTimer(self);self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._idle_step)
        self.setStyleSheet(APP_STYLE)
        self.construct()
        default=DATA_CSV
        if default.is_file():
            self.load_path(default, silent=True)
        else:
            self.refresh()

    def construct(self):
        root=QWidget();root.setObjectName('Root');self.setCentralWidget(root)
        shell=QHBoxLayout(root);shell.setContentsMargins(0,0,0,0);shell.setSpacing(0)
        nav=QWidget();nav.setObjectName('Sidebar');nav.setFixedWidth(245)
        sv=QVBoxLayout(nav);sv.setContentsMargins(18,22,18,22);sv.setSpacing(12)
        sv.addWidget(label('HILLSTROM','Brand'))
        sv.addWidget(label('CAMPAIGN STUDIO','SidebarTag'))
        sv.addSpacing(33)
        sv.addWidget(label('WORKSPACE','SidebarTag'))
        self.nav_buttons=[]
        for idx, title in enumerate(['Tổng quan chiến dịch','Khám phá dữ liệu','Chất lượng dữ liệu']):
            b=QPushButton(title);b.setObjectName('SideButton');b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=idx:self.show_page(i))
            sv.addWidget(b);self.nav_buttons.append(b)
        sv.addStretch()
        footer=label('HILLSTROM  ·  CAMPAIGN ANALYTICS','SidebarTag')
        sv.addWidget(footer)
        shell.addWidget(nav)
        main=QWidget();ml=QVBoxLayout(main);ml.setContentsMargins(20,24,16,16);ml.setSpacing(14)
        top=QHBoxLayout();top.setContentsMargins(8,0,8,0)  # thẳng mép với thân thẻ (thẻ có 8px bóng đổ)
        self.title=label(PAGE_INFO[0][0],'PageTitle')
        self.subtitle=label(PAGE_INFO[0][1],'Subtitle',True)  # xuống dòng khi cửa sổ hẹp
        titles=QVBoxLayout();titles.setSpacing(4);titles.addWidget(self.title);titles.addWidget(self.subtitle)
        top.addLayout(titles,1)  # tiêu đề giãn hết chỗ trống, phụ đề chỉ xuống dòng khi thật sự hẹp
        self.export_btn=QPushButton('Xuất biểu đồ PNG');self.export_btn.setObjectName('Secondary')
        self.export_btn.setIcon(icon('report'))
        self.export_btn.setToolTip('Lưu tất cả biểu đồ của trang hiện tại thành file PNG')
        self.export_btn.clicked.connect(self.export_charts)
        self.open_btn=QPushButton('Mở dữ liệu CSV');self.open_btn.setObjectName('Primary');self.open_btn.clicked.connect(self.pick_csv)
        self.open_btn.setToolTip('Chọn file CSV Hillstrom khác để phân tích')
        self.open_btn.setIcon(icon('upload','light'))
        HoverGlow(self.open_btn, T['brand600'], rest_alpha=45, hover_alpha=120)
        HoverGlow(self.export_btn, T['brand500'], rest_alpha=0, hover_alpha=60)
        for b in (self.export_btn, self.open_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setIconSize(QSize(17,17))
            top.addWidget(b, alignment=Qt.AlignmentFlag.AlignTop)
        top.setSpacing(10)
        ml.addLayout(top)
        self.controls_row=QWidget()
        controls = QHBoxLayout(self.controls_row);controls.setSpacing(10);controls.setContentsMargins(8,0,8,0)
        self.filter_label = label('Phạm vi phân tích', 'FilterLabel')
        controls.addWidget(self.filter_label)
        self.filter=PolishedComboBox();self.filter.addItems(list(FILTERS))
        # Đổi bộ lọc: số liệu + biểu đồ chuyển cảnh mềm thay vì nhảy tức thì.
        self.filter.currentTextChanged.connect(lambda _value: crossfade(self.pages, self.refresh))
        self.filter.setCursor(Qt.CursorShape.PointingHandCursor)
        controls.addWidget(self.filter)
        controls.addStretch()
        self.source=label('','SourceChip');controls.addWidget(self.source)
        ml.addWidget(self.controls_row)
        ml.addSpacing(2)
        self.pages=QStackedWidget();ml.addWidget(self.pages,stretch=1)
        self.make_overview_page();self.make_eda_page();self.make_quality_page()
        shell.addWidget(main,stretch=1)
        self.show_page(0)

    def scroll_page(self):
        outer=DragScrollArea();outer.setWidgetResizable(True)
        # Thẻ có vùng bóng đổ 8px nên lề trang được trừ tương ứng để mép thẳng hàng với tiêu đề.
        inner=QWidget();inner.setObjectName('PageBody');v=QVBoxLayout(inner);v.setContentsMargins(0,0,6,8);v.setSpacing(8)
        outer.setWidget(inner)
        return outer,v

    def make_overview_page(self):
        scroll,v=self.scroll_page()
        self.kpis=[KpiCard('Tổng khách hàng','people'),KpiCard('Đã nhận email','mail'),
                   KpiCard('Mua hàng sau email','check'),KpiCard('Tỷ lệ chuyển đổi / email','analytics',featured=True)]
        v.addWidget(ResponsiveRow(self.kpis, breakpoint=1000))
        v.addSpacing(6)
        v.addWidget(label('SO SÁNH THEO NHÓM CHIẾN DỊCH','SectionLabel'))
        grid=QGridLayout();grid.setSpacing(4)
        self.overview_charts=[
            PlotCard('Quy mô ba nhóm', 'Số khách hàng theo nhóm ngẫu nhiên'),
            PlotCard('Tỷ lệ chuyển đổi', 'Tỷ lệ mua hàng trong 2 tuần sau chiến dịch'),
            PlotCard('Tỷ lệ truy cập website', 'Tỷ lệ visit theo nhóm'),
            PlotCard('Chi tiêu trung bình', 'USD/khách hàng trong 2 tuần sau chiến dịch')]
        for i,p in enumerate(self.overview_charts):grid.addWidget(p,i//2,i%2)
        v.addLayout(grid)
        self.home_note=wrap_label('','Note');self.home_note.setContentsMargins(8,0,8,0);v.addWidget(self.home_note)
        v.addStretch()
        self.pages.addWidget(scroll)

    def make_eda_page(self):
        scroll,v=self.scroll_page()
        self.eda_hint=wrap_label('','Subtitle');self.eda_hint.setContentsMargins(8,0,8,0);v.addWidget(self.eda_hint)
        grid=QGridLayout();grid.setSpacing(4)
        chartdefs=[
            ('Phễu hành vi khách hàng','Quy mô quan sát → truy cập → mua hàng'),
            ('Chuyển đổi theo kênh mua sắm','Channel: Phone / Web / Multichannel'),
            ('Chuyển đổi theo recency','Số tháng kể từ lần mua trước'),
            ('Chuyển đổi theo lịch sử chi tiêu','Các khoảng history_segment trong 12 tháng trước'),
            ('Chuyển đổi theo khu vực','Nhóm zip_code (chuẩn hóa Suburban)'),
            ('Khách hàng mới và cũ','So sánh tỷ lệ chuyển đổi theo newbie')]
        self.eda_charts=[PlotCard(t,n) for t,n in chartdefs]
        for i,p in enumerate(self.eda_charts):grid.addWidget(p,i//2,i%2)
        v.addLayout(grid);v.addStretch()
        self.pages.addWidget(scroll)

    def make_quality_page(self):
        scroll,v=self.scroll_page()
        self.quality_summary=wrap_label('','Subtitle');self.quality_summary.setContentsMargins(8,0,8,0)
        v.addWidget(self.quality_summary)
        self.quality_table=QTableWidget(0,6)
        self.quality_table.setHorizontalHeaderLabels(['Biến','Kiểu dữ liệu','Số dòng','Thiếu','Thiếu (%)','Giá trị khác nhau'])
        for col in range(6):
            # Tiêu đề căn cùng phía với dữ liệu: chữ căn trái, số căn phải.
            align=Qt.AlignmentFlag.AlignLeft if col<2 else Qt.AlignmentFlag.AlignRight
            self.quality_table.horizontalHeaderItem(col).setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        self.quality_table.setAlternatingRowColors(True)
        self.quality_table.verticalHeader().setVisible(False)
        self.quality_table.verticalHeader().setDefaultSectionSize(38)
        self.quality_table.setShowGrid(False)
        self.quality_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.quality_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.quality_table.setMinimumHeight(200)
        self.quality_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for col in range(1,6):self.quality_table.horizontalHeader().setSectionResizeMode(col,QHeaderView.ResizeMode.ResizeToContents)
        self.quality_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_card=ElevatedFrame(radius=14)
        tl=QVBoxLayout(table_card);tl.setContentsMargins(1,1,1,6);tl.addWidget(self.quality_table)
        v.addWidget(table_card)
        for txt in ('Lưu ý: dữ liệu khách hàng không có mã định danh duy nhất; số dòng giống nhau hoàn toàn không tự chứng minh đó là khách hàng trùng lặp.',
                    'visit và spend là kết quả sau chiến dịch, chỉ dùng cho EDA; không đưa vào X của mô hình dự đoán trước chiến dịch.'):
            note=wrap_label(txt,'Note');note.setContentsMargins(8,0,8,0);v.addWidget(note)
        v.addStretch()
        self.pages.addWidget(scroll)

    # ------------------------------------------------------- lazy rendering
    def _flush_page(self, page):
        for job in self._jobs.pop(page, []):
            job()

    def _idle_step(self):
        for page in list(self._jobs):
            jobs=self._jobs[page]
            if jobs:
                jobs.pop(0)()
                if not jobs: del self._jobs[page]
                self._idle_timer.start(30)
                return
            del self._jobs[page]

    def show_page(self, index):
        self._flush_page(index)
        self.pages.setCurrentIndex(index)
        for i, b in enumerate(self.nav_buttons): b.setChecked(i == index)
        self.title.setText(PAGE_INFO[index][0])
        self.subtitle.setText(PAGE_INFO[index][1])
        # Trang chất lượng luôn kiểm tra toàn bộ dữ liệu gốc nên bộ lọc bị khóa (giữ vị trí để bố cục không nhảy).
        self.filter.setEnabled(self.df is not None)
        # Trang Chất lượng dữ liệu luôn kiểm tra toàn bộ dữ liệu gốc: ẩn cả dòng phạm vi phân tích
        # (tên nguồn dữ liệu được đưa vào câu tóm tắt của trang).
        self.controls_row.setVisible(index != 2)
        self.export_btn.setVisible(index != 2)



    def pick_csv(self):
        filepath,_=QFileDialog.getOpenFileName(self,'Chọn Hillstrom CSV',str(DATA_DIR),'CSV (*.csv);;All files (*)')
        if filepath:self.load_path(filepath)

    def load_path(self,path,silent=False):
        try:
            df=read_csv(path)
            self.df=df;self.csv_path=Path(path)
            self.filter.setEnabled(True)
            self.refresh()
            if not silent:show_success(self,'Nạp dữ liệu thành công',f'Đã đọc {fmt_int(len(df))} dòng. Các trang phân tích đã được cập nhật.',str(path))
        except Exception as ex:
            self.df=None;self.csv_path=None
            self.refresh()
            if not silent:show_error(self,'Không thể đọc CSV','File được chọn không phải dữ liệu Hillstrom hợp lệ. Hãy chọn lại file hillstrom.csv.',str(ex))

    def refresh(self):
        if not hasattr(self,'kpis'):return
        has_raw=self.df is not None
        self.filter.setEnabled(has_raw)
        if not has_raw:
            self.filter.blockSignals(True);self.filter.setCurrentIndex(0);self.filter.blockSignals(False)
        group=self.filter.currentText()
        current=filtered(self.df,group) if has_raw else None
        all_overview=overview(self.df)
        active_stats=group_stats(current) if has_raw else group_stats(None)
        if has_raw and FILTERS.get(group) is not None:
            active_stats=active_stats[active_stats['segment'].isin(FILTERS[group])].copy()
        source_str=('Nguồn: '+self.csv_path.name+' · '+fmt_int(len(self.df))+' dòng') if has_raw else 'Chưa nạp CSV'
        self.source.setText(source_str)
        self.source.setToolTip(str(self.csv_path) if has_raw else 'Đang dùng số liệu tổng hợp từ tài liệu nhóm (Structure(1).pdf)')
        self.source.setMinimumWidth(self.source.sizeHint().width())  # không để nhãn nguồn bị cắt chữ
        if has_raw:
            n=len(current)
            target=int(current['segment'].isin(['Mens E-Mail','Womens E-Mail']).sum())
            buyers=int((current['conversion']==1).sum())
            rate=100*buyers/n if n else 0
            title='Số khách nhận email trong bộ lọc'
            self.kpis[2].heading.setText('Khách mua hàng (bộ lọc)')
            self.kpis[3].heading.setText('Tỷ lệ chuyển đổi (bộ lọc)')
            purchases_note='Số khách mua trong phạm vi lọc'
            rate_note='Mua / tổng khách trong phạm vi lọc'
        else:
            n=all_overview.total; target=all_overview.emailed;buyers=all_overview.email_buys or 0
            rate=all_overview.email_rate;title='Nhóm nhận email (2 nhóm)'
            self.kpis[2].heading.setText('Mua hàng sau khi nhận email')
            self.kpis[3].heading.setText('Tỷ lệ chuyển đổi / email')
            purchases_note='Chỉ hai nhóm nhận email'
            rate_note='456 / 42.694 khách nhận email'
        as_int=lambda v: fmt_int(int(round(v)))
        self.kpis[0].update_data(n, 'Quan sát đang được hiển thị', as_int)
        self.kpis[1].update_data(target, title, as_int)
        self.kpis[2].update_data(buyers, purchases_note, as_int)
        self.kpis[3].update_data(rate, rate_note, fmt_pct)
        names=[LABELS[x] for x in active_stats['segment']]
        cols=[SEGMENT_COLORS[x] for x in active_stats['segment']]
        metrics=[active_stats['n'].tolist(),active_stats['conversion_pct'].tolist(),active_stats['visit_pct'].tolist(),active_stats['avg_spend'].tolist()]
        overview_jobs=[(lambda plot=plot,vals=vals,i=i: plot.bars(names,vals,colors=cols,percent=(i in (1,2)),money=(i==3)))
                       for i,(plot,vals) in enumerate(zip(self.overview_charts,metrics))]
        self.home_note.setText(
            ('Số liệu nạp từ CSV và tính lại theo bộ lọc đang chọn. ' if has_raw else
             'Số liệu tổng hợp lấy từ tài liệu Structure(1).pdf của nhóm; tỷ lệ theo nhóm được làm tròn như tài liệu. ') +
            'Chi tiêu sau chiến dịch chỉ dùng để mô tả; không sử dụng làm đặc trưng dự đoán trước email.')
        self._jobs={0:overview_jobs, 1:self._eda_jobs(current)}
        if self._quality_for is not self.df or self.df is None:
            self._jobs[2]=[self.render_quality]
        self._flush_page(self.pages.currentIndex())
        self._idle_timer.start(MOTION['slow']+80)   # vẽ các trang ẩn sau khi hiệu ứng chuyển cảnh xong

    def _history_bins(self):
        """Nhóm history_segment (1..7) tính 1 lần cho mỗi bộ dữ liệu, dùng lại cho mọi bộ lọc."""
        if self._history_for is not self.df:
            # Prefixes preserve the documented order of seven pre-campaign spending bands.
            self._history_bin=self.df['history_segment'].astype(str).str.extract(r'^(\d+)')[0]
            self._history_for=self.df
        return self._history_bin

    def _eda_jobs(self, subset):
        """Trả về danh sách hàm vẽ (mỗi hàm 1 biểu đồ) cho trang Khám phá dữ liệu."""
        charts=self.eda_charts
        if subset is None:
            self.eda_hint.setText('Bấm “Mở dữ liệu CSV” và chọn hillstrom.csv để xem EDA theo từng đặc điểm khách hàng. Trang Tổng quan vẫn hiển thị số liệu tổng hợp từ tài liệu nhóm.')
            return [chart.empty for chart in charts]
        self.eda_hint.setText(f'Đang phân tích {fmt_int(len(subset))} quan sát. Các biểu đồ phản ánh dữ liệu được lọc, không tự động xóa dòng giống nhau.')
        if subset.empty:
            return [(lambda c=c: c.empty('Không có dữ liệu trong bộ lọc')) for c in charts]
        jobs=[]
        def funnel():
            n=len(subset);visits=int((subset['visit']==1).sum());buys=int((subset['conversion']==1).sum())
            charts[0].bars(['Khách hàng','Truy cập website','Mua hàng'],[n,visits,buys],colors=FUNNEL_COLORS)
        jobs.append(funnel)
        def by_column(pos,col):
            names,values,_=conversion_by(subset,col)
            if col=='newbie':names=['Khách cũ' if n in ('0','0.0') else 'Khách mới' if n in ('1','1.0') else n for n in names]
            charts[pos].bars(names,values,percent=True,horizontal=pos in (1,4)) if names else charts[pos].empty(f'CSV không có cột {col}')
        jobs.append(lambda: by_column(1,'channel'))
        def recency():
            if 'recency' in subset:
                r=subset.dropna(subset=['recency','conversion'])
                s=r.groupby('recency')['conversion'].mean().sort_index()*100
                charts[2].line([str(int(x)) if float(x).is_integer() else str(x) for x in s.index],s.tolist()) if len(s) else charts[2].empty()
            else:charts[2].empty('CSV không có cột recency')
        jobs.append(recency)
        def history():
            if 'history_segment' in subset:
                h=subset[['conversion']].assign(_history_bin=self._history_bins().loc[subset.index])
                names,values,_=conversion_by(h,'_history_bin',order=[str(i) for i in range(1,8)])
                charts[3].bars(['Nhóm '+x for x in names],values,percent=True) if names else charts[3].empty()
            elif 'history' in subset:
                h=subset.copy()
                try:
                    h['_history_bin']=pd.qcut(h['history'],q=4,duplicates='drop').astype('string')
                    names,values,_=conversion_by(h,'_history_bin')
                    charts[3].bars([f'Q{i+1}' for i in range(len(names))],values,percent=True) if names else charts[3].empty()
                except ValueError:charts[3].empty('Không đủ dữ liệu để chia nhóm lịch sử')
            else:charts[3].empty('CSV không có history_segment')
        jobs.append(history)
        jobs.append(lambda: by_column(4,'zip_code'))
        jobs.append(lambda: by_column(5,'newbie'))
        return jobs

    def render_quality(self):
        if self.df is None:
            self.quality_summary.setText('Chưa nạp CSV. Bấm “Mở dữ liệu CSV” để kiểm tra cấu trúc, giá trị thiếu và số dòng giống nhau.')
            self.quality_table.setRowCount(0)
            self.quality_table.setFixedHeight(200)
            return
        self._quality_for=self.df
        rows,duplicates=data_quality(self.df)
        source=f'Nguồn: {self.csv_path.name}. ' if self.csv_path is not None else ''
        self.quality_summary.setText(source+f'Kiểm tra toàn bộ dữ liệu gốc: {fmt_int(len(self.df))} dòng × {len(self.df.columns)} cột. ' +
            f'Số dòng có nội dung giống hệt dòng khác: {fmt_int(duplicates)} (chỉ cảnh báo; không tự xóa).')
        self.quality_table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            fmt=[values[0],values[1],fmt_int(values[2]),fmt_int(values[3]),fmt_pct(values[4]),fmt_int(values[5])]
            for c, s in enumerate(fmt):
                item=QTableWidgetItem(str(s));item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignLeft if c<2 else Qt.AlignmentFlag.AlignRight))
                self.quality_table.setItem(r,c,item)
        header_h=self.quality_table.horizontalHeader().sizeHint().height()
        self.quality_table.setFixedHeight(header_h+38*len(rows)+4)

    def export_charts(self):
        current=self.pages.currentIndex()
        self._flush_page(current)
        charts=self.overview_charts if current==0 else self.eda_charts if current==1 else []
        if not charts:return
        target=QFileDialog.getExistingDirectory(self,'Chọn thư mục xuất các biểu đồ',str(Path.home()))
        if not target:return
        from re import sub
        from unicodedata import normalize
        prefix='tong_quan' if current==0 else 'eda'
        try:
            for i,plot in enumerate(charts,1):
                title=plot.findChildren(QLabel)[0].text().replace('đ','d').replace('Đ','D')
                ascii_title=normalize('NFKD',title).encode('ascii','ignore').decode()
                name=sub(r'[^a-zA-Z0-9_-]+','_',ascii_title.lower()).strip('_') or f'chart{i}'
                plot.fig.savefig(str(Path(target)/f'{prefix}_{i:02d}_{name}.png'),dpi=170,bbox_inches='tight',facecolor='white')
            show_success(self,'Xuất biểu đồ thành công',f'Đã lưu {len(charts)} hình PNG.',target)
        except OSError as ex:show_error(self,'Không thể xuất biểu đồ','Không ghi được file vào thư mục đã chọn. Hãy chọn thư mục khác hoặc kiểm tra quyền ghi.',str(ex))


def main():
    app=QApplication(sys.argv)
    apply_light_theme(app)
    win=Dashboard();win.show()
    return app.exec()

if __name__=='__main__':
    sys.exit(main())
