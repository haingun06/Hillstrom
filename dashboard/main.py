
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from app import APP_STYLE, Dashboard
from tich_hop_FS2 import CampaignPredictionPage, icon
from motion import crossfade
from ui_theme import T, apply_light_theme


class HillstromIntegrated(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hillstrom | Campaign Studio')
        self.setWindowIcon(icon('brand'))
        self.resize(1470, 950)
        self.setMinimumSize(1080, 740)


        self.prediction = CampaignPredictionPage()
        self.setCentralWidget(self.prediction)


        shell_layout = self.prediction.layout()
        if shell_layout is None or shell_layout.count() != 2:
            raise RuntimeError('Bố cục trang dự đoán đã thay đổi; không thể tích hợp sidebar.')
        prediction_content = shell_layout.takeAt(1).widget()
        if prediction_content is None:
            raise RuntimeError('Không tìm thấy vùng nội dung trang dự đoán.')

        self.content_stack = QStackedWidget(self.prediction)
        self.content_stack.setObjectName('mainContentStack')
        shell_layout.addWidget(self.content_stack, 1)
        self.content_stack.addWidget(prediction_content)
        self.prediction_content = prediction_content
        self.dashboard = Dashboard()
        dashboard_root = self.dashboard.centralWidget()
        dashboard_layout = dashboard_root.layout()
        if dashboard_layout is None or dashboard_layout.count() != 2:
            raise RuntimeError('Bố cục trang phân tích đã thay đổi; không thể tích hợp.')

        old_sidebar = dashboard_layout.takeAt(0).widget()
        self.dashboard_content = dashboard_layout.takeAt(0).widget()
        if old_sidebar is None or self.dashboard_content is None:
            raise RuntimeError('Không thể tách vùng nội dung trang phân tích.')

        old_sidebar.hide()
        self.dashboard_content.setObjectName('dashboardContent')
        self.dashboard_content.setStyleSheet(
            APP_STYLE + f"\nQWidget#dashboardContent {{ background: {T['bg']}; }}"
        )
        self.content_stack.addWidget(self.dashboard_content)

        for index, nav in enumerate(self.prediction.nav_buttons[1:]):
            try:
                nav.clicked.disconnect()  # Gỡ popup placeholder cũ.
            except (TypeError, RuntimeError):
                pass
            nav.clicked.connect(lambda checked=False, i=index: self.show_dashboard(i))
        self.prediction.nav_buttons[0].clicked.connect(self.show_prediction)

        self.show_prediction()

    def _highlight_nav(self, active_index: int):
        for index, nav in enumerate(self.prediction.nav_buttons):
            nav.setChecked(index == active_index)

    def show_dashboard(self, index: int):
        if index not in (0, 1, 2):
            return
        self._highlight_nav(index + 1)
        if (self.content_stack.currentWidget() is self.dashboard_content
                and self.dashboard.pages.currentIndex() == index):
            return  # đang ở đúng trang, không chạy lại hiệu ứng

        def switch():
            self.dashboard.show_page(index)
            self.content_stack.setCurrentWidget(self.dashboard_content)

        # Trang cũ mờ dần, vùng nội dung trang mới trượt nhẹ lên.
        crossfade(self.content_stack, switch, slide_target=lambda: self.dashboard.pages.currentWidget())

    def show_prediction(self, checked=False):
        self._highlight_nav(0)
        if self.content_stack.currentWidget() is self.prediction_content:
            return
        crossfade(self.content_stack, lambda: self.content_stack.setCurrentWidget(self.prediction_content),
                  slide_target=lambda: self.prediction_content)


def main():
    app = QApplication(sys.argv)
    apply_light_theme(app)
    win = HillstromIntegrated()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
