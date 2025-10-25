# RVE 2.3.8 pre-release
### Added
 - Open output folder button
 - Cap output scale based on model scale
### Changed 
 - UI Tweaks
### Fixed
 - ROCm showing cuda on front end.
 - GMFSS not working
### Removed
 - GIMM.
 - locking the app to 1 instance.
# RVE 2.3.7
### Added:
 - PyTorch 2.9
 - Video Encoder Speed setting.
### Changed:
 - Force MacOS MPS to use PyTorch 2.9, as it should allow for uint16 support.
 - Better 10bit video support.
### Fixed:
 - Color shifting introduced in 2.3.6.
 - DRUNet error at some resolutions.
 - Adding to queue issues on MacOS.
 - Multiple subtitle videos causing issue with playback.
 - Settings not sticking after restart.
 - ROCm half precision not being detected.
# RVE 2.3.6
### Fixed:
 - Color shifting in well... everything.
 - MacOS MPS not working.
# RVE 2.3.5
- NOTE: Pre-releases are unstable, please use the stable build if you experience issues. 
        New features will be added to this release over time, the current changelog is not final.
### Added
 - Auto HDR mode.
 - Full HDR support with proper color encoding.
 - Full pytorch xpu Support.
 - Low storage warning when installing backends.
 - Thanks @adriabama06 for adding model name to video output, last input folder setting and fixing slo-mo mode data. 
 - IFRNet Pytorch
 - Denoise
### Changed
 - Added torch 2.8 as default, removed torch 2.7.
 - Bumped TensorRT to 10.12.
 - Limit release notes to 5 releases.
### Fixed
 - MacOS Arm now uses ARM version of FFMpeg
 - NCNN causing Win10 dump files to be created.
 - Tiling not working on restoration models.
 - Windows setup not installing for all users.
 - Some settings not saving after restart.
 - Some input files not working due to naming.
 - Insanely small amounts of color shifting when processing with rife pytorch/trt.
### Removed
 - ROCm override hack. (Manually set env vars)
 - Interpolation UHD Mode for RIFE.
# RVE 2.3.4
### Added
 - Decompress
 - Linux ARM support
### Fixed
 - Border Crop
 - Fixed Upscaling TRT again on windows.
# RVE 2.3.3
### Added
 - Batch input support.
 - Dynamic TensorRT engines. (Enabled by default, if the engine has issues please go to Settings -> Render Settings -> Uncheck TensorRT Dynamic Engine)
 - More RIFE Models.
 ### Fixed
 - Incorrect colors with some videos.
 - Incorrect RIFE 4K output
 - Upscaling TensorRT on Windows.
 - Upscaling Preview.
# RVE 2.3.2
### Added
 - Fallback to float32 if float16 does not work with custom arch.
### Fixed
 - TensorRT upscaling on windows due to PyTorch version. Please completely uninstall and reinstall if this issue impacts you.

# RVE 2.3.1
### Changed
 - Bump torch to 2.7.1
 - Patches for xpu (might work now)
### Removed
 - Fractional Interpolation.
### Fixed
 - Upscaling high ram usage.
 - Incorrect upscale override.
 - Slow upscale.
# RVE 2.3.0
### Added
 - PyTorch MPS Support for MacOS (Special thanks to Gold John King on RVE Discord for testing the code)
 - AniSD model suite.
### Changed
 - GUI Modifications.
 - Make torch 2.7 default, 2.6 now uses cuda 11.8
 - Improve Startup Time.
 - Update TensorRT to 10.10

### Fixed
 - Fixed ROCm installation not launching on linux.
 - Fix completely black output using specific resolutions with tensorrt. 
 - Fix auto cropping.

# RVE 2.2.5
 - Updates backend and python version.
 - Lock to make sure the app does not have duplicates open.
 - Switchable pytorch backend and version.
 - Blackwell GPU support. 
 - Selectable upscale scale independent of model scale.
 - Custom ffmpeg args in front end.
 - Windows installer.
### Changed
 - Installes to a preset directory, and updates python and backend dynamically.
 - Improved RIFE TensorRT speeds
 - Bump torch to 2.7
 - Bump ROCm to 6.3
 - Bump CUDA to 12.8
 - Enabled tensorrt and torch in flatpak.
 - No more terminal running when using RVE on windows.
### Fixed
 - Fixed HDR Mode
 
# RVE 2.2.0 
### Added
 - Hardware Encoding
 - Auto border cropping
 - GPU ID selection
 - Default video container setting
 - Batch input support
 - Full fallback on TRT upscaling, so the engine has the greatest chance of building correctly.
 - Additional Codec Support (Video, Audio, Subtitle)
 - Additional Animation and Realistic Models
 - Option to reencode audio, in case it has issues with output.
 - HDR Mode toggle (experimental)
### Changed
 - Adjusted dynamic scale.
 - Moved pausing to shared memory.
 - Changed checks on default output directory.
 - Optimized engine building for interpolate and upscale at the same time, should now use less VRAM.
 - Improved tooltips.
 - Adjusted default scene detect sensitivity to 3.5.
 - Bump torch to 2.6.0
 - Bump numpy to 2.2.2
 - Bump tensorrt to 10.7
### Fixed
 - Win10 tooltips and pop ups.
 - Issue when DAR is not equal to SAR (Thanks Yasand123).
   
# RVE 2.1.5
### Added
 - Stopping render, instead of having to kill the entire app.
 - More tiling options.
 - Better checks on imported models.
 - Subtitle passthrough.
 - Changelog view in home menu.
 - Upscale and Interpolate at the same time.
 - Sudo shuffle span for pytorch backend
 - [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI)
 - GMFSS Pro, which helps fix text warping.
 - SloMo mode
 - Ensemble for Pytorch/TensorRT interpolation.
 - Dynamic Optical Flow for Pytorch interpolation.
### Changed
 - Make RVE smaller by switching to pyside6-essentials. (thanks zeptofine!) 
 - Make GUI more compact.
 - Bump torch to 2.6.0-dev20241214.
 - Bump CUDA to 12.6
 - Remove CUDA install requirement for GMFSS
### Fixed
 - ROCm upscaling not working. 
# RVE 2.1.0
### Added
 - Custom Upscale Model Support (TensorRT/Pytorch/NCNN)
 - GMFSS
 - RIFE 4.25 (TensorRT/Pytorch/NCNN)
 - PySceneDetect for better scene change detections
 - More NCNN models
### Removed
 - MacOS Support fully, not coming back due to changes made by Apple.
### Changed
 - Simplified TensorRT Engine Building
 - Increased RIFE TensorRT speed
 - Better preview that pads instead of stretches
 - Updated PyTorch to 2.6.0.dev20241023
 - Updated TensorRT to 10.6
 - Naming scheme of upscaling models, should be easier to understand


