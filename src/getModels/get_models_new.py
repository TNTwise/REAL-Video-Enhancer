import os
thisdir=os.getcwd()
import requests
    
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    i=0
    if iteration >= total and i == 0: 
        print('\nDownloaded!')
        i=1
def install_modules(links_and_names_dict):
    total_size_in_bytes=0
    data_downloaded=0
    for link,name in links_and_names_dict.items():
        response = requests.get(link, stream=True)
        total_size_in_bytes+= int(response.headers.get('content-length', 0))
    for link,name in links_and_names_dict.items():
        response = requests.get(link, stream=True)
        with open(f'{thisdir}/files/{name}', 'wb') as f:
                for data in response.iter_content(1024):
                    f.write(data)
                    data_downloaded+=1024
                    printProgressBar(data_downloaded/total_size_in_bytes*100,100)
            
install_modules({'https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip':'rife-ncnn-vulkan-20221029-ubuntu.zip',
        'https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip':'realesrgan-ncnn-vulkan-20220424-ubuntu.zip',
        'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
        'https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip':'waifu2x-ncnn-vulkan-20220728-ubuntu.zip'})