"""Design tokens + thiết lập giao diện chung cho toàn bộ ứng dụng PyQt6.

Mọi màu sắc / bo góc / thời lượng chuyển động đều lấy từ đây. Stylesheet viết
với cú pháp `@token` (ví dụ `background: @brand700;`) rồi gọi `qss()` để thay
bằng giá trị thật — không hard-code mã màu rải rác trong các file giao diện.

Ghi chú kỹ thuật:
- Dùng style "Fusion": style "windows11" mặc định của Qt 6.7+ xung đột với
  stylesheet của QSpinBox.
- Luôn dùng bảng màu sáng: Windows để Dark mode sẽ tô nền đen cho mọi vùng
  không đặt màu riêng.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import QApplication

ASSETS = Path(__file__).resolve().parent / 'assets'

# ---------------------------------------------------------------- tokens ---
T = {
    # Nền & bề mặt
    'bg': '#F1F5F0',          # nền ứng dụng (trắng ánh xanh)
    'surface': '#FFFFFF',     # thẻ
    'surface2': '#F7FAF6',    # ô bên trong thẻ
    'line': '#E2E9E0',        # viền nhẹ
    'line2': '#CBD7C8',       # viền trang trí
    'field': '#879888',       # viền ô nhập / nút điều khiển (WCAG 1.4.11: 3,05:1 trên nền trắng)
    # Chữ (tương phản trên nền trắng: ink 15:1, ink2 10:1, muted 5,3:1)
    'ink': '#10261A',
    'ink2': '#2C4436',
    'muted': '#5A6B60',
    'subtle': '#8A988E',      # chỉ dùng cho chữ phụ không quan trọng
    # Thương hiệu: emerald
    'brand950': '#04281D',
    'brand900': '#064E3B',
    'brand800': '#065F46',
    'brand700': '#047857',    # nút chính (chữ trắng 5,5:1)
    'brand600': '#059669',
    'brand500': '#10B981',
    'brand300': '#6EE7B7',
    'brand200': '#A7F3D0',
    'brand100': '#D1FAE5',
    'brand50': '#ECFDF5',
    # Nhấn: vàng hổ phách
    'gold800': '#92400E',
    'gold700': '#B45309',
    'gold600': '#D97706',
    'gold500': '#F59E0B',
    'gold400': '#FBBF24',
    'gold200': '#FDE68A',
    'gold100': '#FEF3C7',
    'gold50': '#FFFBEB',
    # Phụ: indigo (phương án Womens E-Mail)
    'indigo800': '#3730A3',
    'indigo700': '#4338CA',
    'indigo600': '#5B6CF0',
    'indigo100': '#E0E7FF',
    'indigo50': '#EEF2FF',
    'danger': '#DC2626',
}

# Màu dữ liệu (đã kiểm tra phân biệt được với người mù màu, kèm nhãn số trực tiếp).
SERIES = {'Mens E-Mail': '#0E9F6E', 'Womens E-Mail': '#5B6CF0', 'No E-Mail': '#E8890C'}

RADIUS = {'sm': 8, 'md': 10, 'lg': 16, 'xl': 20}
MOTION = {'fast': 140, 'base': 220, 'slow': 420, 'count': 650}

_TOKEN = re.compile(r'@([a-z]+[0-9]*)')


def qss(template: str) -> str:
    """Thay `@token` trong stylesheet bằng giá trị trong T."""
    return _TOKEN.sub(lambda m: T[m.group(1)], template)


# Màu của từng tông icon (assets/<tên>_<tông>.png).
ICON_TONES = {
    'emerald': T['brand700'], 'gold': T['gold600'], 'indigo': '#4F5BD5',
    'ink': '#052E21', 'sage': '#7A8B74', 'light': '#FFFFFF',
}
_icon_cache: dict = {}
_warned: set = set()


def asset_url(filename: str) -> str:
    """Đường dẫn dùng trong stylesheet, có ngoặc kép để không vỡ khi thư mục có dấu cách / ngoặc."""
    return f'"{(ASSETS / filename).as_posix()}"'


def icon(name: str, tone: str = 'emerald') -> QIcon:
    """Nạp icon theo tông màu.

    Nếu thiếu đúng file `<name>_<tone>.png` (vd. bàn giao thiếu file), tự tô màu lại từ
    một tông khác của cùng icon và in cảnh báo, thay vì để icon trống không báo lỗi.
    """
    key = (name, tone)
    if key in _icon_cache:
        return _icon_cache[key]
    path = ASSETS / f'{name}_{tone}.png'
    if path.exists():
        result = QIcon(str(path))
    else:
        sources = sorted(ASSETS.glob(f'{name}_*.png'))
        result = QIcon()
        if sources and tone in ICON_TONES:
            img = QImage(str(sources[0])).convertToFormat(QImage.Format.Format_ARGB32)
            p = QPainter(img)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            p.fillRect(img.rect(), QColor(ICON_TONES[tone]))
            p.end()
            result = QIcon(QPixmap.fromImage(img))
        if key not in _warned:
            _warned.add(key)
            how = f'tô màu lại từ {sources[0].name}' if not result.isNull() else 'KHÔNG có file nguồn nào'
            print(f'[Hillstrom] Thiếu icon {path.name} trong {ASSETS} -> {how}', file=sys.stderr)
    _icon_cache[key] = result
    return result


def apply_light_theme(app: QApplication) -> None:
    app.setStyle('Fusion')
    hints = app.styleHints()
    if hasattr(hints, 'setColorScheme'):  # Qt >= 6.8
        hints.setColorScheme(Qt.ColorScheme.Light)
    font = QFont('Segoe UI', 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    # Tooltip hiện mượt. Dropdown tự làm hiệu ứng hiện dần (motion.PolishedComboBox),
    # nên tắt hiệu ứng trượt mặc định để không chồng 2 hiệu ứng.
    QApplication.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
    for effect in (Qt.UIEffect.UI_AnimateTooltip, Qt.UIEffect.UI_FadeTooltip):
        QApplication.setEffectEnabled(effect, True)

    pal = QPalette()
    role = QPalette.ColorRole
    for r, color in {
        role.Window: T['bg'], role.Base: T['surface'], role.AlternateBase: T['surface2'],
        role.WindowText: T['ink'], role.Text: T['ink'], role.ButtonText: T['ink'],
        role.Button: T['surface2'], role.Highlight: T['brand600'], role.HighlightedText: '#FFFFFF',
        role.ToolTipBase: T['surface'], role.ToolTipText: T['ink'],
        role.PlaceholderText: T['subtle'], role.Link: T['brand700'],
        role.Light: '#FFFFFF', role.Midlight: T['surface2'], role.Mid: T['line2'],
        role.Dark: '#A8B6A9', role.Shadow: '#7A8B74',
    }.items():
        pal.setColor(r, QColor(color))
    disabled = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(disabled, r, QColor('#A0A99C'))
    app.setPalette(pal)

    app.setStyleSheet(qss('''
    QToolTip { background: @surface; color: @ink; border: 1px solid @line2; padding: 7px 10px; border-radius: 8px; font-size: 12px; }
    QMessageBox { background: @surface; }
    QMessageBox QLabel { color: @ink; font-size: 13px; }
    QMessageBox QPushButton { background: @brand700; color: white; border: 0; border-radius: 9px; padding: 8px 20px; min-width: 72px; font-weight: 700; }
    QMessageBox QPushButton:hover { background: @brand800; }
    QMessageBox QPushButton:focus { border: 2px solid @gold400; }
    '''))
