import torch
import torch.nn.functional as F
import importlib
from pathlib import Path
from PIL import Image
import torch.nn as nn
from torch.nn.functional import interpolate

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch_tensorrt = importlib.import_module('torch_tensorrt') if importlib.util.find_spec('torch_tensorrt') else None

def warp(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid):
    dtype = tenInput.dtype
    tenInput = tenInput.to(torch.float)
    tenFlow = tenFlow.to(torch.float)

    tenFlow = torch.cat([tenFlow[:, 0:1] / tenFlow_div[0], tenFlow[:, 1:2] / tenFlow_div[1]], 1)
    g = (backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode="bilinear", padding_mode="border", align_corners=True).to(dtype)
    
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
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
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
        self, img0, img1, timestep, tenFlow_div, backwarp_tenGrid, scale=None
    ):
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        if scale is not None:
            self.scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        for i in range(4):
            if flow is None:
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
            else:
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
                flow = flow + f0
                mask = mask + m0
            latest_mask = mask
            warped_img0 = self.warp(
                img0, flow[:, :2], tenFlow_div, backwarp_tenGrid
            )
            warped_img1 = self.warp(
                img1, flow[:, 2:4], tenFlow_div, backwarp_tenGrid
            )

        temp = torch.sigmoid(latest_mask)
        frame = warped_img0 * temp + warped_img1 * (1 - temp)
        return frame


def _build_inputs(device, dtype=torch.float16, height=1088, width=1920, seed=1234):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    img0 = torch.rand(1, 3, height, width, device=device, dtype=dtype, generator=generator)
    img1 = torch.rand(1, 3, height, width, device=device, dtype=dtype, generator=generator)
    tenFlow = torch.rand(1, 2, height, width, device=device, dtype=dtype, generator=generator)
    tenFlow_div = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
    backwarp_tenGrid = torch.rand(1, 2, height, width, device=device, dtype=dtype, generator=generator)
    timestep = timestep_tens = torch.full(
                (1, 1, height, width),
                0.5,
            ).to(device).to(dtype)
    return img0, img1, tenFlow_div, backwarp_tenGrid, timestep


def save_output_image(output: torch.Tensor, path: str):
    image_tensor = output.detach().to(torch.float32).cpu()[0]
    image_tensor = image_tensor.clamp(0.0, 1.0)

    image_array = (
        image_tensor[:3].permute(1, 2, 0) * 255.0
    ).to(torch.uint8).numpy()
    image = Image.fromarray(image_array, mode='RGB')

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def test_warp(img0, img1, timestep, tenFlow_div, backwarp_tenGrid, dtype=torch.float32):
    img0 = img0.to(device, dtype)
    img1 = img1.to(device, dtype)
    timestep = timestep.to(device, dtype)
    tenFlow_div = tenFlow_div.to(device, dtype)
    backwarp_tenGrid =backwarp_tenGrid.to(device, dtype)
    model = IFNet().eval().to(device, dtype)
    state_dict = torch.load('rife4.6.pkl', map_location=device)
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if "module." in k
    }
    model.load_state_dict(state_dict)
    
    output = model(img0, img1, timestep, tenFlow_div, backwarp_tenGrid)
    assert output.shape == img0.shape
    return output


def test_warp_trt(img0, img1, timestep, tenFlow_div, backwarp_tenGrid, use_explicit_typing=True, dtype=torch.float32):
    img0 = img0.to(device, dtype)
    img1 = img1.to(device, dtype)
    timestep = timestep.to(device, dtype)
    tenFlow_div = tenFlow_div.to(device, dtype)
    backwarp_tenGrid =backwarp_tenGrid.to(device, dtype)
    model = IFNet().eval().to(device, dtype)
    
    state_dict = torch.load('rife4.6.pkl', map_location=device)
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if "module." in k
    }
    
    example_inputs = (img0, img1, timestep, tenFlow_div, backwarp_tenGrid)

    exported = torch.export.export(model, example_inputs)
    model_trt = torch_tensorrt.dynamo.compile(
        exported,
        tuple(example_inputs),
        device=torch.device('cuda:0'),
        use_explicit_typing=use_explicit_typing,
        num_avg_timing_iters=4,
        workspace_size=0,
        min_block_size=1,
    )
    output = model_trt(img0, img1, timestep, tenFlow_div, backwarp_tenGrid)
    assert output.shape == img0.shape
    return output


if __name__ == "__main__":
    img0, img1, tenFlow_div, backwarp_tenGrid, timestep = _build_inputs(
        device,
        dtype=torch.float32,
        seed=1234,
    )
    output = test_warp(img0, img1, timestep, tenFlow_div, backwarp_tenGrid)
    
    save_output_image(output, 'backend/tests/unit_tests/output_pytorch.png')

    output_trt = test_warp_trt(img0, img1, timestep, tenFlow_div, backwarp_tenGrid, use_explicit_typing=True, dtype=torch.float16)
    save_output_image(output_trt, 'backend/tests/unit_tests/output_trt_broken.png')

    output_trt = test_warp_trt(img0, img1, timestep, tenFlow_div, backwarp_tenGrid, use_explicit_typing=False, dtype=torch.float32)
    save_output_image(output_trt, 'backend/tests/unit_tests/output_trt_correct.png')