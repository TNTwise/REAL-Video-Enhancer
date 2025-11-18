from gimmvfi_r import GIMMVFI_R

import torch
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np


class InputPadder:
    """Pads images such that dimensions are divisible by divisor"""

    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, *inputs):
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode="replicate")
        else:
            return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]

    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "xpu" if torch.xpu.is_available() else "cpu")
model = GIMMVFI_R("GIMMVFI_RAFT.pth").to(device)


def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module" in k}


ckpt = torch.load("gimmvfi_r_arb_lpips.pt", map_location="cpu")
raft = torch.load("raft-things.pth", map_location="cpu")
combined_state_dict = {"gimmvfi_r": ckpt["state_dict"], "raft": convert(raft)}
torch.save(combined_state_dict, "GIMMVFI_RAFT.pth")
model.load_state_dict(combined_state_dict["gimmvfi_r"])

images = []


def load_image(img_path):
    img = Image.open(img_path)
    raw_img = np.array(img.convert("RGB"))
    img = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
    return img.to(torch.float).unsqueeze(0)


img_path0 = "0001.png"
img_path2 = "0004.png"
# prepare data b,c,h,w
I0 = load_image(img_path0)
I2 = load_image(img_path2)
padder = InputPadder(I0.shape, 32)
I0, I2 = padder.pad(I0, I2)
xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(device, non_blocking=True)
print(I0.shape)
print(xs.shape)
model.eval()
batch_size = xs.shape[0]
s_shape = xs.shape[-2:]

model.zero_grad()
ds_factor = 0.5
interp_factor = 4
with torch.no_grad():
    coord_inputs = [
        (
            model.sample_coord_input(
                batch_size,
                s_shape,
                [1 / interp_factor * i],
                device=xs.device,
                upsample_ratio=ds_factor,
            ),
            None,
        )
        for i in range(1, interp_factor)
    ]
    timesteps = [
        i
        * 1
        / interp_factor
        * torch.ones(xs.shape[0]).to(xs.device).to(torch.float).reshape(-1, 1, 1, 1)
        for i in range(1, interp_factor)
    ]
    output = model(xs, coord_inputs[2], timestep=timesteps[2], ds_factor=ds_factor)
    # out_flowts = [padder.unpad(f) for f in all_outputs["flowt"]]

    images.append((output.detach().cpu().numpy()).astype(np.uint8))