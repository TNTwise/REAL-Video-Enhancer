import src.thisdir
import datetime
import os
current_time = datetime.datetime.now().time()

thisdir = src.thisdir.thisdir()
try:
    os.mkdir(f'{thisdir}/logs/')
    
except:
        if len(os.listdir(f'{thisdir}/logs/')) > 4:
            oldest_file = min(os.listdir(f'{thisdir}/logs/'), key=lambda x: os.path.getctime(os.path.join(f'{thisdir}/logs/', x)))
            os.remove(f'{thisdir}/logs/{oldest_file}')
            
def log(log):
    
    with open(f'{thisdir}/logs/log_{current_time}.txt', 'a') as f:
        f.write(log + '\n')

    