# RVE 2.2.5 pre-release
 - NOTE: Pre-releases are unstable, please use the stable build if you experience issues. 
        New features will be added to this release over time, the current changelog is not final. 

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


