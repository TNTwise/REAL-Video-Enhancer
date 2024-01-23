import time
from time import sleep
from src.settings import Settings
def convertTime(remaining_time):
    hours = remaining_time // 3600
    remaining_time-= 3600*hours
    minutes = remaining_time // 60
    remaining_time -= minutes * 60
    seconds = remaining_time
    if minutes < 10:
        minutes = str(f'0{minutes}')
    if seconds < 10:
        seconds = str(f'0{seconds}')
    return hours,minutes,seconds
def calculateETA(self):
        completed_iterations = int(self.files_processed)
                    
                    
                    
        # Estimate the remaining time
        elapsed_time = time.time() - self.start_time
        time_per_iteration = elapsed_time / completed_iterations
        remaining_iterations = self.filecount - completed_iterations
        remaining_time = remaining_iterations * time_per_iteration
        remaining_time = int(remaining_time) 
        # Print the estimated time remaining
        #convert to hours, minutes, and seconds
        hours,minutes,seconds=convertTime(remaining_time)
        return f'ETA: {hours}:{minutes}:{seconds}'