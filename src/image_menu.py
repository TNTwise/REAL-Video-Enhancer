from src.settings import *
import os
def image_menu_on_start(self):
    settings = Settings()
    image_models=['Waifu2x','RealESRGAN']
    for i in os.listdir(settings.ModelDir):
        for model in image_models:
            if model.lower() == i.lower():
                self.ui.AICombo_Image.addItem(model)

