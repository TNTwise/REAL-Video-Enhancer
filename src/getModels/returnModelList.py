import os
import src.thisdir
thisdir = src.thisdir.thisdir()
def returnModelList(self,settings):
                rife_install_list=[]
                try:
                    if self.ui.RifeCheckBox.isChecked():
                        with open(f'{thisdir}/models.txt', 'r') as f:
                            for i in f.readlines():
                                print(i)
                                i=i.replace('\n','')
                                rife_install_list.append(i)
                except Exception as e:
                     if self.ui.RifeCheckBox.isChecked():
                        rife_install_list.append('rife-v4.6')
                
                '''https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip':'realcugan-ncnn-vulkan-20220728-ubuntu.zip',
                'https://github.com/nihui/cain-ncnn-vulkan/releases/download/20220728/cain-ncnn-vulkan-20220728-ubuntu.zip':'cain-ncnn-vulkan-20220728-ubuntu.zip',
                '''
                if os.path.isfile(f'{thisdir}/bin/ffmpeg') and os.path.isfile(f'{thisdir}/bin/glxinfo') and os.path.isfile(f'{thisdir}/bin/yt-dlp_linux'):
                        install_modules_dict = {}
                else:
                        install_modules_dict = {'https://raw.githubusercontent.com/TNTwise/REAL-Video-Enhancer/main/bin/ffmpeg':'ffmpeg',
'https://raw.githubusercontent.com/TNTwise/REAL-Video-Enhancer/main/bin/yt-dlp_linux':'yt-dlp_linux',
'https://raw.githubusercontent.com/TNTwise/REAL-Video-Enhancer/main/bin/glxinfo':'glxinfo',}
                
                
                if self.ui.RifeCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/rife/') == False:
                        install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                if self.ui.RifeCheckBox.isChecked() == False:
                        
                        os.system(f'rm -rf "{settings.ModelDir}/rife/"')
                if self.ui.RealESRGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realesrgan') == False:
                        install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/realesrgan-ncnn-vulkan-20220424-ubuntu.zip'] = 'realesrgan-ncnn-vulkan-20220424-ubuntu.zip'
                if self.ui.RealESRGANCheckBox.isChecked() == False:
                        os.system(f'rm -rf "{settings.ModelDir}/realesrgan/"')
                if self.ui.Waifu2xCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/waifu2x') == False:
                        install_modules_dict['https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20220728/waifu2x-ncnn-vulkan-20220728-ubuntu.zip'] = 'waifu2x-ncnn-vulkan-20220728-ubuntu.zip'
                if self.ui.Waifu2xCheckBox.isChecked() == False:
                        os.system(f'rm -rf "{settings.ModelDir}/waifu2x/"')
                if self.ui.CainCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/ifrnet') == False:
                          install_modules_dict['https://github.com/nihui/ifrnet-ncnn-vulkan/releases/download/20220720/ifrnet-ncnn-vulkan-20220720-ubuntu.zip'] = 'ifrnet-ncnn-vulkan-20220720-ubuntu.zip'
                if self.ui.CainCheckBox.isChecked() == False:
                         os.system(f'rm -rf "{settings.ModelDir}/ifrnet/"')
                
                if self.ui.RealCUGANCheckBox.isChecked() == True and os.path.exists(f'{settings.ModelDir}/realcugan') == False:
                        install_modules_dict['https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-ubuntu.zip'] = 'realcugan-ncnn-vulkan-20220728-ubuntu.zip'
                if self.ui.RealCUGANCheckBox.isChecked() == False:
                        os.system(f'rm -rf "{settings.ModelDir}/realcugan/"')
                for i in rife_install_list:
                        if os.path.exists(f'{settings.ModelDir}/rife/rife-ncnn-vulkan') == False:
                                install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        if os.path.exists(f'{settings.ModelDir}/rife/{i}') == False:
                                install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/{i}.tar.gz'] = f'{i}.tar.gz'
                if rife_install_list == [] and self.ui.RifeCheckBox.isChecked() and os.path.exists(f'{settings.ModelDir}/rife') == False:
                        install_modules_dict['https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-ncnn-vulkan'] = 'rife-ncnn-vulkan'
                        install_modules_dict[f'https://raw.githubusercontent.com/TNTwise/Rife-Vulkan-Models/main/rife-v4.6.tar.gz'] = f'rife-v4.6.tar.gz'
                return install_modules_dict