from abc import ABC, abstractmethod


class WriteBuffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass

    @abstractmethod
    def put_frame_in_write_queue(self, frame):
        pass

    @abstractmethod
    def write_out_frames(self):
        pass

    @abstractmethod
    def get_num_frames_rendered(self) -> int:
        pass

    @abstractmethod
    def onErroredExit(self):
        pass
