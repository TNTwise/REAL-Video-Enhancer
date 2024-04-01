import onnxruntime

# Load the ONNX model
model_path = "path/to/your/model.onnx"
session = onnxruntime.InferenceSession(model_path)
from PIL import Image
import numpy as np
import io


def process_bytes_input(image_bytes):
    # Decode bytes into an image
    image = Image.open(io.BytesIO(image_bytes))
    # Preprocess the image (e.g., resize, normalize)
    # Example preprocessing steps:
    resized_image = image.resize((224, 224))
    normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values
    # Convert image to the format expected by the model (e.g., NCHW format)
    input_data = np.transpose(
        normalized_image, (2, 0, 1)
    )  # Assuming NHWC to NCHW format conversion
    return input_data


def predict(image_bytes):
    input_data = process_bytes_input(image_bytes)
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    return result


if __name__ == "__main__":
    with open("path/to/your/image.png", "rb") as f:
        image_bytes = f.read()
        predictions = predict(image_bytes)
        print(predictions)
