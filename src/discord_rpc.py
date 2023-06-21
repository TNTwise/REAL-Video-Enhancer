from pypresence import Presence
from time import sleep
import asyncio
import os
def start_discordRPC(self):
        try:
            

            client_id = '1120814311246405743'  # ID for rpc
            self.RPC = Presence(client_id)  # Initialize the client class
            self.RPC.connect() # Start the handshake loop
            
            self.RPC.update(state=f"{self.videoName}", details=f"Interpolating Video",large_image='logov1') 
                
            
          # The presence will stay on as long as the program is running
         # Can only update rich presence every 15 seconds
        except Exception as e:
             print(e)
