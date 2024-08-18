from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect
from PySide6.QtCore import QPropertyAnimation, QEasingCurve
class AnimationHandler:


    def fadeInAnimation(self, qObject: QWidget, n=None):
        self.opacity_effect = QGraphicsOpacityEffect()
        qObject.setGraphicsEffect(self.opacity_effect)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)  # Duration in milliseconds
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

    def fadeOutAnimation(self, qObject: QWidget, n=None):
        self.opacity_effect = QGraphicsOpacityEffect()
        qObject.setGraphicsEffect(self.opacity_effect)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(200)  # Duration in milliseconds
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.start()

    def fade_to_color(self, color):
        self.animation = QPropertyAnimation(self, b"styleSheet")
        self.animation.setDuration(5000)  # Duration in milliseconds
        self.animation.setStartValue(self.styleSheet())
        self.animation.setEndValue(color)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()