import torch
import torch.nn.functional as F
import importlib

torch_tensorrt = importlib.import_module('torch_tensorrt') if importlib.util.find_spec('torch_tensorrt') else None

class TorchModel(torch.nn.Module):
    def forward(self, tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
        dtype = tenInput.dtype
        tenInput = tenInput.to(torch.float)
        tenFlow = tenFlow.to(torch.float)
        tenFlow_div = tenFlow_div.to(torch.float)

        tenFlow = torch.cat(
            [tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1
        )
        g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
        pd = 'border'
        pd = 'zeros'
        g = g.clamp(-1, 1)
        return F.grid_sample(
            input=tenInput,
            grid=g,
            mode='bilinear',
            padding_mode=pd,
            align_corners=True,
        ).to(dtype)


def _build_inputs(device, dtype=torch.float16, height=64, width=64):
    tenInput = torch.rand(1, 3, height, width, device=device, dtype=dtype)
    tenFlow = torch.rand(1, 2, height, width, device=device, dtype=dtype)
    tenFlow_div = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    backwarp_tenGrid = torch.rand(1, 2, height, width, device=device, dtype=dtype)
    return tenInput, tenFlow, tenFlow_div, backwarp_tenGrid


def test_warp():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TorchModel().to(device).eval()
    tenInput, tenFlow, tenFlow_div, backwarp_tenGrid = _build_inputs(
        device, dtype=torch.float16 if device.type == 'cuda' else torch.float32
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
        device, dtype=torch.float16
    )
    example_inputs = (tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)

    exported = torch.export.export(model, example_inputs)
    model_trt = torch_tensorrt.dynamo.compile(
        exported,
        tuple(example_inputs),
        device=torch.device('cuda:0'),
        enabled_precisions={torch.half},
        use_explicit_typing=False,
        num_avg_timing_iters=4,
        min_block_size=1,
    )
    output = model_trt(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)
    assert output.shape == tenInput.shape
    return output


if __name__ == "__main__":
    print(test_warp())
    print(test_warp_trt())
