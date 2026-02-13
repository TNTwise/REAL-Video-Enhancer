import torch
import torch.nn.functional as F
import torch_tensorrt

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

def test_warp():
    tenInput = torch.rand(1, 3, 4, 4).cuda()
    tenFlow = torch.rand(1, 2, 4, 4).cuda()
    tenFlow_div = torch.tensor([1.0, 1.0]).cuda()
    backwarp_tenGrid = torch.rand(1, 2, 4, 4).cuda()

    model = TorchModel().cuda()
    output = model(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)

def test_warp_trt():
    tenInput = torch.rand(1, 3, 4, 4).cuda()
    tenFlow = torch.rand(1, 2, 4, 4).cuda()
    tenFlow_div = torch.tensor([1.0, 1.0]).cuda()
    backwarp_tenGrid = torch.rand(1, 2, 4, 4).cuda()

    model = TorchModel()
    trt_model = torch_tensorrt.compile(model, inputs=[tenInput, tenFlow, tenFlow_div, backwarp_tenGrid])
    output = trt_model(tenInput, tenFlow, tenFlow_div, backwarp_tenGrid)

if __name__ == "__main__":
    test_warp()
    test_warp_trt()
