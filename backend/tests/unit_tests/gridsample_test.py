import torch
import torch.nn.functional as F
import importlib
from pathlib import Path
from PIL import Image

torch_tensorrt = importlib.import_module('torch_tensorrt') if importlib.util.find_spec('torch_tensorrt') else None

def warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid, return_grid=False):
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)
    tenFlow_div = tenFlow_div.to(torch.float)
    backwarp_tenGrid = backwarp_tenGrid.to(torch.float)

    tenFlow = torch.cat(
        [tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1
    )
    g_raw = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    pd = 'zeros'
    g = g_raw.clamp(-1, 1)
    sampled = F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode=pd, align_corners=True)
    if return_grid:
        return sampled, g_raw
    return sampled

"""
MIT License

Copyright (c) 2024 Hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.nn.functional import interpolate


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class MyPixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        b, c, hh, hw = input.size()
        out_channel = c // (self.upscale_factor**2)
        h = hh * self.upscale_factor
        w = hw * self.upscale_factor
        x_view = input.view(
            b, out_channel, self.upscale_factor, self.upscale_factor, hh, hw
        )
        return x_view.permute(0, 1, 4, 2, 5, 3).reshape(b, out_channel, h, w)


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), MyPixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = interpolate(x, scale_factor=1.0 / scale, mode='bilinear')
        if flow is not None:
            flow = (
                interpolate(flow, scale_factor=1.0 / scale, mode='bilinear')
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = interpolate(tmp, scale_factor=scale, mode='bilinear')
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device: torch.device = torch.device('cuda'),
        width=1920,
        height=1080,
    ):
        super().__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        scale = scale
        self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dtype = dtype
        self.device = device
        self.width = width
        self.height = height
        self.block = [self.block0, self.block1, self.block2, self.block3]
        self.warp = warp

    def forward(
        self,
        img0,
        img1,
        timestep,
        tenFlow_div,
        backwarp_tenGrid,
        scale=None,
        return_intermediates=False,
    ):
        model_dtype = img0.dtype
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        img0_f32 = img0.to(torch.float32)
        img1_f32 = img1.to(torch.float32)
        timestep_f32 = timestep.to(torch.float32)
        tenFlow_div_f32 = tenFlow_div.to(torch.float32)
        backwarp_tenGrid_f32 = backwarp_tenGrid.to(torch.float32)
        if scale is not None:
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        warped_img0_f32 = img0_f32
        warped_img1_f32 = img1_f32
        flow_f32 = None
        mask_f32 = None
        debug_tensors = []

        for i in range(4):
            if flow_f32 is None:
                flow, mask = self.block[i](
                    torch.cat((img0, img1, timestep), 1),
                    None,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = self.block[i](
                        torch.cat((img1, img0, 1 - timestep), 1),
                        None,
                        scale=self.scale_list[i],
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
                flow_f32 = flow.to(torch.float32)
                mask_f32 = mask.to(torch.float32)
            else:
                warped_img0 = warped_img0_f32.to(model_dtype)
                warped_img1 = warped_img1_f32.to(model_dtype)
                flow = flow_f32.to(model_dtype)
                mask = mask_f32.to(model_dtype)
                f0, m0 = self.block[i](
                    torch.cat((warped_img0, warped_img1, timestep, mask), 1),
                    flow,
                    scale=self.scale_list[i],
                )
                if self.ensemble:
                    f1, m1 = self.block[i](
                        torch.cat(
                            (
                                warped_img1,
                                warped_img0,
                                1 - timestep,
                                -mask,
                            ),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=self.scale_list[i],
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2
                flow_f32 = flow_f32 + f0.to(torch.float32)
                mask_f32 = mask_f32 + m0.to(torch.float32)
            latest_mask_f32 = mask_f32
            if return_intermediates:
                warped_img0_f32, grid0_raw = self.warp(
                    img0_f32,
                    flow_f32[:, :2],
                    tenFlow_div_f32,
                    backwarp_tenGrid_f32,
                    return_grid=True,
                )
                warped_img1_f32, grid1_raw = self.warp(
                    img1_f32,
                    flow_f32[:, 2:4],
                    tenFlow_div_f32,
                    backwarp_tenGrid_f32,
                    return_grid=True,
                )
            else:
                warped_img0_f32 = self.warp(
                    img0_f32, flow_f32[:, :2], tenFlow_div_f32, backwarp_tenGrid_f32
                )
                warped_img1_f32 = self.warp(
                    img1_f32, flow_f32[:, 2:4], tenFlow_div_f32, backwarp_tenGrid_f32
                )
            if return_intermediates:
                debug_tensors.extend(
                    [
                        flow_f32,
                        mask_f32,
                        warped_img0_f32,
                        warped_img1_f32,
                        grid0_raw,
                        grid1_raw,
                    ]
                )

        temp = torch.sigmoid(latest_mask_f32)
        frame = warped_img0_f32 * temp + warped_img1_f32 * (1 - temp)
        frame = frame.to(model_dtype)
        if return_intermediates:
            return (frame, *debug_tensors)
        return frame


class IFNetDebugWrapper(nn.Module):
    def __init__(self, model: IFNet):
        super().__init__()
        self.model = model

    def forward(self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid):
        return self.model(
            img0,
            img1,
            timestep,
            tenFlow_div,
            backwarp_tenGrid,
            return_intermediates=True,
        )

def _build_inputs(device, dtype=torch.float16, height=256, width=256, seed=1234):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    img0 = torch.rand(1, 3, height, width, device=device, dtype=dtype, generator=generator)
    img1 = torch.rand(1, 3, height, width, device=device, dtype=dtype, generator=generator)
    timestep = torch.full((1, 1, height, width), 0.5, device=device, dtype=dtype)
    tenFlow_div = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    backwarp_tenGrid = torch.rand(1, 2, height, width, device=device, dtype=dtype, generator=generator)
    return img0, img1, timestep, tenFlow_div, backwarp_tenGrid


def _build_model(device, dtype=torch.float16, seed=1234):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = IFNet(device=device).to(device).eval()
    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()
    return model


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
    run_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = _build_model(device, dtype=run_dtype, seed=1234)
    img0, img1, timestep, tenFlow_div, backwarp_tenGrid = _build_inputs(
        device,
        dtype=run_dtype,
        seed=1234,
    )
    output = model(img0, img1, timestep, tenFlow_div, backwarp_tenGrid)
    assert output.shape == img0.shape
    return output


def test_warp_trt():
    if torch_tensorrt is None or not torch.cuda.is_available():
        return None

    device = torch.device('cuda:0')
    model = _build_model(device, dtype=torch.float16, seed=1234)
    img0, img1, timestep, tenFlow_div, backwarp_tenGrid = _build_inputs(
        device, dtype=torch.float16, seed=1234
    )
    example_inputs = (img0, img1, timestep, tenFlow_div, backwarp_tenGrid)

    exported = torch.export.export(model, example_inputs)
    model_trt = torch_tensorrt.dynamo.compile(
        exported,
        tuple(example_inputs),
        device=torch.device('cuda:0'),
        enabled_precisions=(torch.float,),
        use_explicit_typing=True,
        torch_executed_ops={
            torch.ops.aten.grid_sampler.default,
            torch.ops.aten.upsample_bilinear2d.vec,
        },
        num_avg_timing_iters=4,
        workspace_size=0,
        min_block_size=1,
    )
    output = model_trt(img0, img1, timestep, tenFlow_div, backwarp_tenGrid)
    assert output.shape == img0.shape
    return output


def _flatten_to_tuple(outputs):
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _stage_name(idx):
    names = ['flow', 'mask', 'warp_img0', 'warp_img1', 'grid_raw0', 'grid_raw1']
    return f'stage{idx // len(names)}_{names[idx % len(names)]}'


def _grid_stats(grid_tensor: torch.Tensor):
    grid = grid_tensor.detach().float()
    out_of_range = ((grid < -1.0) | (grid > 1.0)).float()
    out_of_range_ratio = float(out_of_range.mean().item())
    return {
        'min': float(grid.min().item()),
        'max': float(grid.max().item()),
        'out_of_range_ratio': out_of_range_ratio,
        'clamped_ratio': out_of_range_ratio,
    }


def debug_compare_stages_trt(seed=1234, dtype=torch.float16):
    if torch_tensorrt is None or not torch.cuda.is_available():
        print('TRT/CUDA unavailable; skipping stage debug')
        return None

    device = torch.device('cuda:0')
    model = _build_model(device, dtype=dtype, seed=seed)
    debug_model = IFNetDebugWrapper(model).eval()

    img0, img1, timestep, tenFlow_div, backwarp_tenGrid = _build_inputs(
        device, dtype=dtype, seed=seed
    )
    example_inputs = (img0, img1, timestep, tenFlow_div, backwarp_tenGrid)

    with torch.no_grad():
        pt_outputs = _flatten_to_tuple(debug_model(*example_inputs))

    exported = torch.export.export(debug_model, example_inputs)
    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        tuple(example_inputs),
        device=device,
        enabled_precisions=(torch.float,),
        use_explicit_typing=True,
        torch_executed_ops={
            torch.ops.aten.grid_sampler.default,
            torch.ops.aten.upsample_bilinear2d.vec,
        },
        num_avg_timing_iters=4,
        workspace_size=0,
        min_block_size=1,
    )

    with torch.no_grad():
        trt_outputs = _flatten_to_tuple(trt_model(*example_inputs))

    if len(pt_outputs) != len(trt_outputs):
        raise RuntimeError(
            f'Output tuple length mismatch: pt={len(pt_outputs)}, trt={len(trt_outputs)}'
        )

    metrics = []
    for idx, (pt_t, trt_t) in enumerate(zip(pt_outputs, trt_outputs)):
        d = (pt_t.detach().float() - trt_t.detach().float()).abs()
        if idx == 0:
            name = 'final_frame'
        else:
            name = _stage_name(idx - 1)
        metrics.append(
            {
                'name': name,
                'max_abs': float(d.max().item()),
                'mean_abs': float(d.mean().item()),
            }
        )

    for item in metrics:
        print(
            f"{item['name']}: max_abs={item['max_abs']:.6f}, mean_abs={item['mean_abs']:.6f}"
        )

    print('\nGrid stats (raw, pre-clamp):')
    for idx, (pt_t, trt_t) in enumerate(zip(pt_outputs, trt_outputs)):
        name = 'final_frame' if idx == 0 else _stage_name(idx - 1)
        if 'grid_raw' not in name:
            continue
        pt_stats = _grid_stats(pt_t)
        trt_stats = _grid_stats(trt_t)
        print(
            f"{name} PT: min={pt_stats['min']:.6f}, max={pt_stats['max']:.6f}, "
            f"out_of_range_ratio={pt_stats['out_of_range_ratio']:.6f}, "
            f"clamped_ratio={pt_stats['clamped_ratio']:.6f}"
        )
        print(
            f"{name} TRT: min={trt_stats['min']:.6f}, max={trt_stats['max']:.6f}, "
            f"out_of_range_ratio={trt_stats['out_of_range_ratio']:.6f}, "
            f"clamped_ratio={trt_stats['clamped_ratio']:.6f}"
        )

    return metrics


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

    print('\nStage-by-stage TRT debug:')
    debug_compare_stages_trt(seed=1234, dtype=torch.float16)
