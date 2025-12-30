import numpy as np
import os
import cv2
class InferenceSceneChangeDetectEfficientNetNCNN:
    """
    NCNN-based scene change detector using EfficientNet.
    Takes numpy arrays as inputs instead of torch tensors.
    """

    def __init__(self, threshold=0.3, model_path="", model_dtype="float32", model_device="cpu", model_gpu_id=0):
        """
        Initialize the NCNN scene change detector.

        Args:
            threshold: Scene change detection threshold (0-1 range, multiplied by 0.1 internally)
            model_path: Path to the NCNN model files (without .param/.bin extension)
            model_dtype: Data type for inference (unused for NCNN, kept for API compatibility)
            model_device: Device for inference (unused for NCNN, kept for API compatibility)
        """
        import ncnn

        self.threshold = threshold * 0.1
        self.ncnn = ncnn

        self.debug = False

        # Load NCNN model
        self.net = ncnn.Net()
        # Enable Vulkan if available for GPU acceleration
        self.net.opt.use_vulkan_compute = True

        self.net.set_vulkan_device(model_gpu_id)

        # Load param and bin files
        # Expecting model_path to be the base path (e.g., "model" for "model.param" and "model.bin")
        param_path = os.path.join(model_path, os.path.basename(model_path) + ".param") 
        bin_path = os.path.join(model_path, os.path.basename(model_path) + ".bin")
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # Store input/output layer names (adjust based on actual model)
        self.input_name = "in0"
        self.output_name = "out0"

    def _preprocess(self, frame: np.ndarray) -> "ncnn.Mat":
        """
        Preprocess a numpy array frame for NCNN inference.

        Args:
            frame: Input frame as numpy array, expected shape (H, W, C) or (C, H, W)
                   Values should be in range [0, 1] or [0, 255]

        Returns:
            ncnn.Mat ready for inference
        """
        # Ensure frame is in (H, W, C) format
        if frame.ndim == 3 and frame.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            frame = np.transpose(frame, (1, 2, 0))
        elif frame.ndim == 4:
            # (N, C, H, W) -> (H, W, C), take first batch
            frame = np.transpose(frame[0], (1, 2, 0))

        # Normalize to [0, 1] if values are in [0, 255]
        if frame.max() > 1.0:
            frame = frame / 255.0

        # Resize to 256x256 if needed
        if frame.shape[0] != 256 or frame.shape[1] != 256:
            
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        # Convert to NCNN Mat (expects H, W, C format with contiguous memory)
        frame = np.ascontiguousarray(frame)

        # Create ncnn Mat from numpy array
        # ncnn.Mat expects (H, W, C) layout for from_pixels
        mat = self.ncnn.Mat.from_pixels(
            (frame * 255).astype(np.uint8),
            self.ncnn.Mat.PixelType.PIXEL_RGB,
            frame.shape[1],  # width
            frame.shape[0],  # height
        )
        mean_vals = []
        norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat.substract_mean_normalize(mean_vals, norm_vals)
        
        return mat

    def __call__(self, frame_0: np.ndarray, frame_1: np.ndarray) -> bool:
        """
        Detect scene change between two frames.

        Args:
            frame_0: First frame as numpy array
            frame_1: Second frame as numpy array

        Returns:
            True if scene change detected, False otherwise
        """
        # Preprocess both frames
        mat_0 = self._preprocess(frame_0)
        mat_1 = self._preprocess(frame_1)

        # Create extractor for inference
        ex = self.net.create_extractor()

        # For models that take concatenated input (2 frames stacked)
        # We need to concatenate along channel dimension
        # This creates a 6-channel input from two 3-channel frames
        h, w = mat_0.h, mat_0.w
        c = mat_0.c

        # Extract numpy arrays from mats
        arr_0 = np.array(mat_0).reshape(c, h, w)
        arr_1 = np.array(mat_1).reshape(c, h, w)

        # Concatenate along batch dimension (for models expecting stacked frames)
        # Shape becomes (2, C, H, W) which is flattened for ncnn
        combined = np.concatenate([arr_0[np.newaxis, ...], arr_1[np.newaxis, ...]], axis=0)

        # Create combined mat - ncnn expects (C, H, W) so we reshape accordingly
        # For a model expecting batch of 2: we treat it as 6 channels
        combined_flat = combined.reshape(-1, h, w).astype(np.float32)
        
        if self.debug:
            np.save("debug_combined_input.npy", combined_flat)
            # visualize channels 0-2 (first frame) and 3-5 (second frame)
            vis0 = np.transpose(combined_flat[:3], (1, 2, 0))
            vis1 = np.transpose(combined_flat[3:6], (1, 2, 0))
            cv2.imwrite("debug_input_frame0.png", np.clip(vis0 * 255, 0, 255).astype(np.uint8))
            cv2.imwrite("debug_input_frame1.png", np.clip(vis1 * 255, 0, 255).astype(np.uint8))
            print("input stats:", combined_flat.min(), combined_flat.max(), combined_flat.mean())

        combined_mat = self.ncnn.Mat(combined_flat) 

        ex.input(self.input_name, combined_mat)

        # Extract output
        ret, output_mat = ex.extract(self.output_name)

        if ret != 0:
            # Extraction failed
            return False

        # Get output value
        
        output = np.array(output_mat)
        return output[0] > self.threshold