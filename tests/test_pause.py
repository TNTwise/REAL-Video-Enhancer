from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
)

from backend.src import FFmpeg
from src.ui.ProcessTab import ProcessTab


@pytest.fixture
def pause_memory():
    memory = shared_memory.SharedMemory(create=True, size=1)
    memory.buf[0] = 0
    yield memory
    memory.close()
    memory.unlink()


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def process_tab(application, pause_memory):
    tab = ProcessTab.__new__(ProcessTab)
    tab.parent = SimpleNamespace(
        onRenderButtonsContiainer=QHBoxLayout(),
        startRenderButton=QPushButton(),
        progressBar=QProgressBar(),
        renderOutput=QPlainTextEdit(),
        FPS=QLabel(),
        ETA=QLabel(),
        STATUS=QLabel(),
    )
    tab.pausedSharedMemory = pause_memory
    tab.isPaused = False
    tab.status = "Rendering"
    tab.renderTextOutputList = ["FPS: 2 Current Frame: 10 ETA: 0:00:45"]
    tab.max_value = 100
    tab.currentFrame = 10
    tab.fps = "2"
    tab.eta = "0:00:45"
    return tab


def test_pauseRender_keeps_paused_status_during_progress_updates(process_tab):
    process_tab.pauseRender()
    process_tab.status = "Rendering"
    process_tab.updateProcessTab(QImage())

    assert process_tab.pausedSharedMemory.buf[0] == 1
    assert process_tab.parent.STATUS.text() == "Status: Paused"
    assert process_tab.parent.ETA.text() == "ETA: 0:00:45"

    process_tab.resumeRender()
    process_tab.updateProcessTab(QImage())

    assert process_tab.pausedSharedMemory.buf[0] == 0
    assert process_tab.parent.STATUS.text() == "Status: Rendering"


def test_writeOutInformation_excludes_repeated_pauses_from_fps_and_eta(
    monkeypatch, pause_memory
):
    samples = iter(
        [
            (5, 0, 10),
            (10, 1, 10),
            (40, 1, 10),
            (50, 0, 10),
            (60, 0, 20),
            (70, 1, 20),
            (90, 1, 20),
            (100, 0, 20),
        ]
    )
    now = 0
    monkeypatch.setattr(
        FFmpeg, "time", SimpleNamespace(time=lambda: now, monotonic=lambda: now)
    )
    information = FFmpeg.InformationWriteOut(
        None, 3, pause_memory.name, 1, 1, 1, 1, 100
    )
    information.setPreviewFrame(b"\0\0\0")
    messages = []
    monkeypatch.setattr(information, "realTimePrint", messages.append)

    def advance(_seconds: float):
        nonlocal now
        try:
            now, pause_memory.buf[0], frames = next(samples)
        except StopIteration:
            information.stopWriting()
        else:
            information.setFramesRendered(frames)

    monkeypatch.setattr(FFmpeg.time, "sleep", advance, raising=False)
    advance(0)
    try:
        information.writeOutInformation()
    finally:
        information.pausedManager.pausedSharedMemory.close()

    assert messages == [
        "FPS: 2 Current Frame: 10 ETA: 0:00:45",
        "FPS: 1 Current Frame: 10 ETA: 0:01:30",
        "FPS: 1 Current Frame: 20 ETA: 0:01:20",
        "FPS: 1 Current Frame: 20 ETA: 0:02:00",
    ]


def test_writeOutInformation_reads_pause_before_first_preview(monkeypatch, pause_memory):
    pause_memory.buf[0] = 1
    information = FFmpeg.InformationWriteOut(
        None, 3, pause_memory.name, 1, 1, 1, 1, 100
    )
    monkeypatch.setattr(FFmpeg.time, "sleep", lambda _seconds: information.stopWriting())
    try:
        information.writeOutInformation()
        assert information.get_is_paused() is True
    finally:
        information.pausedManager.pausedSharedMemory.close()
