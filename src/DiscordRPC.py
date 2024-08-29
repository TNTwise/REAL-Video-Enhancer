from pypresence import Presence, exceptions
from time import sleep
import signal
from contextlib import contextmanager
import os
from .Util import log


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def start_discordRPC(mode:str, videoName:str, backend:str):
    """
    Attempts to connect to discord for RPC suppor
    Args:
        mode (str): The mode of the video (Interpolating, Upscaling)
        videoName (str): The name of the video
    """
    try:
        with time_limit(2):
            os.system(
                "ln -sf {app/com.discordapp.Discord,$XDG_RUNTIME_DIR}/discord-ipc-0"
            )  # Enables discord RPC on flatpak
            client_id = "1278176380043132961"  # ID for rpc
            try:
                for i in range(10):
                    ipc_path = f"{os.getenv('XDG_RUNTIME_DIR')}/discord-ipc-{i}"
                    if not os.path.exists(ipc_path) or not os.path.isfile(ipc_path):
                        os.symlink(
                            f"{os.getenv('HOME')}/.config/discord/{client_id}", ipc_path
                        )
            except:
                log("Not flatpak")
            try:
                RPC = Presence(client_id)  # Initialize the client class
                RPC.connect()  # Start the handshake loop

                RPC.update(
                    state=f"{mode} Video", details=f"Backend: {backend}", large_image="logo-v2"
                )
            except exceptions.DiscordNotFound:
                pass

    # The presence will stay on as long as the program is running
    # Can only update rich presence every 15 seconds
    except TimeoutException as e:
        log("Timed out!")