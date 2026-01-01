import inspect
import torch
import torch.nn as nn
import timm

def change_first_layer(m):
    for name, child in m.named_children():
        if isinstance(child, nn.Conv2d):
            kwargs = {
                'out_channels': child.out_channels,
                'kernel_size': child.kernel_size,
                'stride': child.stride,
                'padding': child.padding,
                'bias': False if child.bias is None else True
            }
            new_conv = nn.Conv2d(2, **kwargs)
            setattr(m, name, new_conv)
            return True
        else:
            if change_first_layer(child):
                return True
    return False

netD = timm.create_model("maxxvitv2_nano_rw_256.sw_in1k", num_classes=2, pretrained=True, in_chans=6)
#change_first_layer(netD)

# Check the modified first layer

def custom_forward(self, x):
    x = x.unsqueeze(0)
    x = torch.clamp(x, 0, 1)
    x = self.forward_features(x)
    x = self.forward_head(x)
    x = torch.softmax(x, dim=1)
    return x

source_code = inspect.getsource(netD.forward)
print(source_code)

# Replace the forward method with custom_forward
import types
funcType = types.MethodType
netD.forward = funcType(custom_forward, netD)
#netD.forward = custom_forward

source_code = inspect.getsource(netD.forward)
print(source_code)

netD.load_state_dict(torch.load("sc_maxxvitv2_nano_rw_256.sw_in1k_256px_b100_30k_coloraug0.4.pth"))

dummy_input = torch.rand(6, 256, 256).cuda()
netD.cuda()
res = netD(dummy_input)  # Call forward method with dummy input
print(res)
# Export the model to ONNX format
with torch.no_grad():
    exported = torch.export.export(netD, (dummy_input,))
    torch.export.save(exported, "sudo_efficientnet_scenedetect.pt2")
    dynamic_axes = {
        "input": {0: "batch_size", 2: "width", 3: "height"},
        "output": {0: "batch_size"}
    }
    torch.onnx.export(
        netD,
        dummy_input,
        "sudo_efficientnet_scenedetect.onnx",
        verbose=False,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        #dynamic_axes=dynamic_axes
    )