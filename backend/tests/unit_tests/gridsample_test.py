import torch
import torch.nn.functional as F
import importlib
from pathlib import Path
from PIL import Image

torch_tensorrt = importlib.import_module('torch_tensorrt') if importlib.util.find_spec('torch_tensorrt') else None

class TorchModel(torch.nn.Module):
    def forward(self, tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
        dtype = tenInput.dtype
        tenInput = tenInput.to(torch.float)
        tenFlow = tenFlow.to(torch.float)

        tenFlow = torch.cat(
            [tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1
        )
        g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
        pd = 'border'
        if tenInput.device.type == "mps":
            pd = 'zeros'
            g = g.clamp(-1, 1)
        return F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode=pd, align_corners=True).to(dtype)


def _build_inputs(device, dtype=torch.float16, height=64, width=64, seed=1234):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tenInput = torch.rand(1, 3, height, width, device=device, dtype=dtype, generator=generator)
    tenFlow = torch.rand(1, 2, height, width, device=device, dtype=dtype, generator=generator)
    tenFlow_div = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    backwarp_tenGrid = torch.rand(1, 2, height, width, device=device, dtype=dtype, generator=generator)
    return tenInput, tenFlow, tenFlow_div, backwarp_tenGrid


def save_output_image(output: torch.Tensor, path: str):
    image_tensor = output.detach().to(torch.float32).cpu()[0]
    image_tensor = image_tensor.clamp(0.0, 1.0)

    if image_tensor.shape[0] == 1:
        image_array = (image_tensor[0] * 255.0).to(torch.uint8).numpy()
        image = Image.fromarray(image_array, mode='L')
    else:
        image_array = (
            image_tensor[:3].permute(1, 2, 0) * 255.0
        ).to(torch.uint8).numpy()
        image = Image.fromarray(image_array, mode='RGB')

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def test_warp():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TorchModel().to(device).eval()
    tenInput, tenFlow, tenFlow_div, backwarp_tenGrid = _build_inputs(
        device,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        seed=1234,
    )
    output = model(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)
    assert output.shape == tenInput.shape
    return output


def test_warp_trt():
    if torch_tensorrt is None or not torch.cuda.is_available():
        return None

    device = torch.device('cuda:0')
    model = TorchModel().to(device).half().eval()
    tenInput, tenFlow, tenFlow_div, backwarp_tenGrid = _build_inputs(
        device, dtype=torch.float16, seed=1234
    )
    example_inputs = (tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)

    exported = torch.export.export(model, example_inputs)
    model_trt = torch_tensorrt.dynamo.compile(
        exported,
        tuple(example_inputs),
        device=torch.device('cuda:0'),
        use_explicit_typing=True,
        num_avg_timing_iters=4,
        min_block_size=1,
    )
    output = model_trt(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)
    assert output.shape == tenInput.shape
    return output


if __name__ == "__main__":
    output = test_warp()
    print(output)
    
    save_output_image(output, 'backend/tests/unit_tests/output_pytorch.png')
    print('Saved backend/tests/unit_tests/output_pytorch.png')

    output_trt = test_warp_trt()
    print(output_trt)
    if output_trt is not None:
        save_output_image(output_trt, 'backend/tests/unit_tests/output_trt.png')
        print('Saved backend/tests/unit_tests/output_trt.png')
    else:
        print('Skipped TRT output image (TensorRT/CUDA unavailable).')
