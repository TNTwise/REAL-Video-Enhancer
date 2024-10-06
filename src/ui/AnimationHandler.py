from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect, QGraphicsItemAnimation
from PySide6.QtCore import (
    QPropertyAnimation,
    QEasingCurve,
    QRect,
    QPoint,
    QSize,
    QParallelAnimationGroup,
)


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

        self.animation.start()

    def dropDownAnimation(self, widget: QWidget, duration=200):
        return
        start_geometry = widget.geometry()
        end_geometry = QRect(
            start_geometry.x(),
            start_geometry.y(),
            start_geometry.width(),
            start_geometry.height(),
        )

        # Move the widget above its final position
        start_geometry.moveTop(start_geometry.y() - start_geometry.height())

        self.animation = QPropertyAnimation(widget, b"geometry")
        self.animation.setDuration(duration)
        self.animation.setStartValue(start_geometry)
        self.animation.setEndValue(end_geometry)
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.start()

    def moveUpAnimation(self, widget: QWidget, duration=200):
        return
        widget.setVisible(True)
        start_geometry = widget.geometry()
        end_geometry = QRect(
            start_geometry.x(),
            start_geometry.y() - start_geometry.height(),
            start_geometry.width(),
            start_geometry.height(),
        )

        start_geometry.moveTop(start_geometry.y() + start_geometry.height())

        self.animation = QPropertyAnimation(widget, b"geometry")
        self.animation.setDuration(duration)
        self.animation.setStartValue(start_geometry)
        self.animation.setEndValue(end_geometry)
        self.animation.setEasingCurve(QEasingCurve.Linear)
        self.animation.start()

    def dropDownFadeInAnimation(self, widget: QWidget, duration=500):
        return
        # Ensure the widget is initially hidden

        # Drop-down animation
        start_geometry = widget.geometry()
        end_geometry = QRect(
            start_geometry.x(),
            start_geometry.y(),
            start_geometry.width(),
            start_geometry.height(),
        )
        start_geometry.moveTop(start_geometry.y() - start_geometry.height())

        drop_down_animation = QPropertyAnimation(widget, b"geometry")
        drop_down_animation.setDuration(duration)
        drop_down_animation.setStartValue(start_geometry)
        drop_down_animation.setEndValue(end_geometry)
        drop_down_animation.setEasingCurve(QEasingCurve.OutBounce)

        # Fade-in animation
        opacity_effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(opacity_effect)
        fade_in_animation = QPropertyAnimation(opacity_effect, b"opacity")
        fade_in_animation.setDuration(duration)
        fade_in_animation.setStartValue(0)
        fade_in_animation.setEndValue(1)

        # Combine animations
        animation_group = QParallelAnimationGroup()
        animation_group.addAnimation(drop_down_animation)
        animation_group.addAnimation(fade_in_animation)

        # Show the widget and start the animation
        widget.setVisible(True)
        animation_group.start()
