import json
import os
from typing import Dict, List, Any
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from .constants import PRESETS_PATH

class PresetManager:
    def __init__(self, ui_parent=None):
        self.ui_parent = ui_parent
        if not os.path.exists(PRESETS_PATH):
            os.makedirs(PRESETS_PATH)
        
        if self.ui_parent:
            self.setup_ui()

    def setup_ui(self):
        """
        Sets up the UI connections and populates the combobox.
        """
        if hasattr(self.ui_parent, 'selectAvailablePreset'):
            self.populate_presets_combobox()
            self.ui_parent.selectAvailablePreset.currentIndexChanged.connect(self.on_preset_selected)
            
        if hasattr(self.ui_parent, 'addPresetBtn'):
            self.ui_parent.addPresetBtn.clicked.connect(self.on_add_preset_clicked)
            
        if hasattr(self.ui_parent, 'savePresetBtn'):
            self.ui_parent.savePresetBtn.clicked.connect(self.on_save_preset_clicked)

    def populate_presets_combobox(self):
        """
        Populates the selectAvailablePreset combobox with available presets.
        """
        if not hasattr(self.ui_parent, 'selectAvailablePreset'):
            return
            
        self.ui_parent.selectAvailablePreset.blockSignals(True)
        self.ui_parent.selectAvailablePreset.clear()
        self.ui_parent.selectAvailablePreset.addItem("Select a preset...")
        
        presets = self.get_presets()
        for preset in presets:
            self.ui_parent.selectAvailablePreset.addItem(preset)
            
        self.ui_parent.selectAvailablePreset.blockSignals(False)

    def on_preset_selected(self, index):
        """
        Handler for when a preset is selected from the combobox.
        """
        if index <= 0: # "Select a preset..." or empty
            return
            
        preset_name = self.ui_parent.selectAvailablePreset.currentText()
        if preset_name:
            self.apply_preset_to_ui(preset_name, self.ui_parent)

    def on_add_preset_clicked(self):
        """
        Handler for the add preset button. Opens a file dialog to import a preset.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui_parent,
            "Import Preset",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    preset_data = json.load(f)
                
                # Get filename without extension for the preset name
                preset_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Save it to our presets folder
                if self.save_preset(preset_name, preset_data):
                    self.populate_presets_combobox()
                    
                    # Select the newly imported preset
                    index = self.ui_parent.selectAvailablePreset.findText(preset_name)
                    if index >= 0:
                        self.ui_parent.selectAvailablePreset.setCurrentIndex(index)
                        
                    QMessageBox.information(self.ui_parent, "Success", f"Preset '{preset_name}' imported successfully.")
            except Exception as e:
                QMessageBox.critical(self.ui_parent, "Error", f"Failed to import preset: {str(e)}")

    def on_save_preset_clicked(self):
        """
        Handler for the save preset button. Prompts for a name and saves current UI state.
        """
        preset_name, ok = QInputDialog.getText(
            self.ui_parent,
            "Save Preset",
            "Enter a name for the new preset:"
        )
        
        if ok and preset_name:
            # Check if it already exists
            if preset_name in self.get_presets():
                reply = QMessageBox.question(
                    self.ui_parent,
                    "Overwrite Preset",
                    f"A preset named '{preset_name}' already exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                    
            preset_data = self.get_preset_from_ui(self.ui_parent)
            if self.save_preset(preset_name, preset_data):
                self.populate_presets_combobox()
                
                # Select the newly saved preset
                index = self.ui_parent.selectAvailablePreset.findText(preset_name)
                if index >= 0:
                    self.ui_parent.selectAvailablePreset.setCurrentIndex(index)
                    
                QMessageBox.information(self.ui_parent, "Success", f"Preset '{preset_name}' saved successfully.")
            else:
                QMessageBox.critical(self.ui_parent, "Error", "Failed to save preset.")

    def save_preset(self, name: str, preset_data: Dict[str, Any]) -> bool:
        """
        Saves a preset to a JSON file.
        preset_data should contain model selections and their enabled states.
        Returns True if successful, False otherwise.
        """
        try:
            file_path = os.path.join(PRESETS_PATH, f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(preset_data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving preset {name}: {e}")
            return False

    def load_preset(self, name: str) -> Dict[str, Any]:
        """
        Loads a preset from a JSON file.
        Returns the preset data as a dictionary, or an empty dictionary if not found.
        """
        try:
            file_path = os.path.join(PRESETS_PATH, f"{name}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading preset {name}: {e}")
        return {}

    def get_presets(self) -> List[str]:
        """
        Returns a list of available preset names.
        """
        presets = []
        if os.path.exists(PRESETS_PATH):
            for file in os.listdir(PRESETS_PATH):
                if file.endswith(".json"):
                    presets.append(file[:-5])
        return presets

    def delete_preset(self, name: str) -> bool:
        """
        Deletes a preset file.
        Returns True if successful, False otherwise.
        """
        try:
            file_path = os.path.join(PRESETS_PATH, f"{name}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            print(f"Error deleting preset {name}: {e}")
        return False

    def get_preset_from_ui(self, ui_parent) -> Dict[str, Any]:
        """
        Extracts preset data from the UI.
        """
        return {
            "backend": ui_parent.backendComboBox.currentText(),
            "interpolate": ui_parent.interpolateCheckBox.isChecked(),
            "interpolate_model": ui_parent.interpolateModelComboBox.currentText(),
            "upscale": ui_parent.upscaleCheckBox.isChecked(),
            "upscale_model": ui_parent.upscaleModelComboBox.currentText(),
            "deblur": ui_parent.deblurCheckBox.isChecked(),
            "deblur_model": ui_parent.deblurModelComboBox.currentText(),
            "denoise": ui_parent.denoiseCheckBox.isChecked(),
            "denoise_model": ui_parent.denoiseModelComboBox.currentText(),
            "decompress": ui_parent.decompressCheckBox.isChecked(),
            "decompress_model": ui_parent.decompressModelComboBox.currentText(),
            "scene_change_detection_enabled": ui_parent.scene_change_detection_enabled.isChecked(),
            "scene_change_detection_method": ui_parent.scene_change_detection_method.currentText(),
        }

    def apply_preset_to_ui(self, name: str, ui_parent) -> bool:
        """
        Applies a preset to the UI.
        """
        preset_data = self.load_preset(name)
        if not preset_data:
            return False

        if "backend" in preset_data:
            index = ui_parent.backendComboBox.findText(preset_data["backend"])
            if index >= 0:
                ui_parent.backendComboBox.setCurrentIndex(index)

        if "interpolate" in preset_data:
            ui_parent.interpolateCheckBox.setChecked(preset_data["interpolate"])
        if "interpolate_model" in preset_data:
            index = ui_parent.interpolateModelComboBox.findText(preset_data["interpolate_model"])
            if index >= 0:
                ui_parent.interpolateModelComboBox.setCurrentIndex(index)

        if "upscale" in preset_data:
            ui_parent.upscaleCheckBox.setChecked(preset_data["upscale"])
        if "upscale_model" in preset_data:
            index = ui_parent.upscaleModelComboBox.findText(preset_data["upscale_model"])
            if index >= 0:
                ui_parent.upscaleModelComboBox.setCurrentIndex(index)

        if "deblur" in preset_data:
            ui_parent.deblurCheckBox.setChecked(preset_data["deblur"])
        if "deblur_model" in preset_data:
            index = ui_parent.deblurModelComboBox.findText(preset_data["deblur_model"])
            if index >= 0:
                ui_parent.deblurModelComboBox.setCurrentIndex(index)

        if "denoise" in preset_data:
            ui_parent.denoiseCheckBox.setChecked(preset_data["denoise"])
        if "denoise_model" in preset_data:
            index = ui_parent.denoiseModelComboBox.findText(preset_data["denoise_model"])
            if index >= 0:
                ui_parent.denoiseModelComboBox.setCurrentIndex(index)

        if "decompress" in preset_data:
            ui_parent.decompressCheckBox.setChecked(preset_data["decompress"])
        if "decompress_model" in preset_data:
            index = ui_parent.decompressModelComboBox.findText(preset_data["decompress_model"])
            if index >= 0:
                ui_parent.decompressModelComboBox.setCurrentIndex(index)

        if "scene_change_detection_enabled" in preset_data:
            ui_parent.scene_change_detection_enabled.setChecked(preset_data["scene_change_detection_enabled"])
        if "scene_change_detection_method" in preset_data:
            index = ui_parent.scene_change_detection_method.findText(preset_data["scene_change_detection_method"])
            if index >= 0:
                ui_parent.scene_change_detection_method.setCurrentIndex(index)

        # Update the UI to reflect the new checkbox states
        if hasattr(ui_parent, 'updateVideoGUIDetails'):
            ui_parent.updateVideoGUIDetails()

        return True
