import os
import requests
# Print iterations progress
thisdir = os.getcwd()
def start():
    try:
        os.mkdir(f'{thisdir}/files/')
    except:
        os.chdir(f"{thisdir}/files/")
    file=f"rife-ncnn-vulkan-20221029-ubuntu.zip"        
    
        
    
    
    response = requests.get(f"https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip", stream=True)

    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    
    
    
    total_block_size = 0
    with open(f'{thisdir}/files/{file}', 'wb') as f:
        for data in response.iter_content(block_size):
            total_block_size += block_size
            
            printProgressBar(total_block_size,total_size_in_bytes)
            f.write(data)
        os.chdir(f'{thisdir}')
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration >= total: 
        print('\nDownloaded!')
start()
