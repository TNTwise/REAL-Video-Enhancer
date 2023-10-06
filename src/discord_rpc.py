from pypresence import Presence
from time import sleep
import signal
from contextlib import contextmanager
class TimeoutException(Exception): pass
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




def start_discordRPC(self, mode='Interpolating'):
        try:
            
            with time_limit(2):


              client_id = '1120814311246405743'  # ID for rpc
              self.RPC = Presence(client_id)  # Initialize the client class
              self.RPC.connect() # Start the handshake loop
              self.RPC.update(state=f"{self.videoName}", details=f"{mode} Video",large_image='logov1') 
            
            
          # The presence will stay on as long as the program is running
         # Can only update rich presence every 15 seconds
        except TimeoutException as e:
              print("Timed out!")
