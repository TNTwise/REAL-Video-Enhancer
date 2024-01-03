import os
import src.thisdir
thisdir = src.thisdir.thisdir()
def returnModelList(self,settings): # make sure names match up on both selectAI.ui and main.ui
                rife_install_list=[] 
                if os.path.exists(f'{settings.ModelDir}/') == False:
                       os.mkdir(f'{settings.ModelDir}/')
                try:
                    if self.ui.RifeCheckBox.isChecked():
                        with open(f'{thisdir}/models.txt', 'r') as f:
                            for i in f.readlines():
                                
                                i=i.replace('\n','')
                                rife_install_list.append(i)
                                print(f'{i}-ensemble')
                                if 'v4' in i:
                                        rife_install_list.append(f'{i}-ensemble')
                     
                except Exception as e:
                     if self.ui.RifeCheckBox.isChecked():
                        rife_install_list.append('rife-v4.6')
                
                '''https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
                'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
                '''
                
                
                install_modules_dict = {}
                
                if self.ui.RealSRCheckBox.isChecked() and os.path.exists(f'{settings.ModelDir}/realsr/') == False:
                       install_modules_dict['https://github.com/nihui/realsr-ncnn-vulkan/releases/download/20220728/realsr-ncnn-vulkan-20220728-ubuntu.zip'] = 'realsr-ncnn-vulkan-20220728-ubuntu.zip'
                
                if self.ui.RifeCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/rife/') == False:
                        install_modules_dict['https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                
                if self.ui.RealESRGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realesrgan') == False:
                        install_modules_dict['https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/realesrgan-ncnn-vulkan-20220424-ubuntu.zip'] = 'realesrgan-ncnn-vulkan-20220424-ubuntu.zip'
                
                if self.ui.Waifu2xCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/waifu2x') == False:
                        install_modules_dict['https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip'] = 'waifu2x-ncnn-vulkan-20220728-ubuntu.zip'
                
                if self.ui.CainCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/ifrnet') == False:
                          install_modules_dict['https://github.com/nihui/ifrnet-ncnn-vulkan/releases/download/20220720/ifrnet-ncnn-vulkan-20220720-ubuntu.zip'] = 'ifrnet-ncnn-vulkan-20220720-ubuntu.zip'
                
                
                if self.ui.RealCUGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realcugan') == False:
                        install_modules_dict['https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip'] = 'realcugan-ncnn-vulkan-20220728-ubuntu.zip'
                
                for i in rife_install_list:
                        if os.path.exists(f'{settings.ModelDir}/rife/rife-ncnn-vulkan') == False:
                                install_modules_dict['https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        if os.path.exists(f'{settings.ModelDir}/rife/{i}') == False:
                                install_modules_dict[f'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/{i}.tar.gz'] = f'{i}.tar.gz'
                if rife_install_list == [] and self.ui.RifeCheckBox.isChecked() and os.path.exists(f'{settings.ModelDir}/rife') == False:
                        install_modules_dict['https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        install_modules_dict[f'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6.tar.gz'] = f'rife-v4.6.tar.gz'
                        install_modules_dict[f'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6-ensemble.tar.gz'] = f'rife-v4.6-ensemble.tar.gz'
                if len(install_modules_dict) == 0 and len(os.listdir(f'{settings.ModelDir}/')) == 0:
                        install_modules_dict['https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        install_modules_dict[f'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6.tar.gz'] = f'rife-v4.6.tar.gz'
                        install_modules_dict[f'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/rife-v4.6-ensemble.tar.gz'] = f'rife-v4.6-ensemble.tar.gz'
                if os.path.isfile(f'{thisdir}/bin/ffmpeg') and os.path.isfile(f'{thisdir}/bin/glxinfo') and os.path.isfile(f'{thisdir}/bin/yt-dlp_linux'):
                        pass
                else:
                        install_modules_dict.update({
                               'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/glxinfo':'glxinfo',
                               'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/ffmpeg':'ffmpeg',
                                'https://github.com/TNTwise/Rife-Vulkan-Models/releases/download/models/yt-dlp_linux':'yt-dlp_linux'})
                        
                
                return install_modules_dict