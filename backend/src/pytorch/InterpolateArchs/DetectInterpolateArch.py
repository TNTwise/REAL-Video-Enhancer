from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class Arch(ABC):
    base_arch: str
    unique_shapes: dict
    excluded_keys: list

    @staticmethod
    @abstractmethod
    def module() -> torch.nn.Module:
        """The actual module"""


class RIFE46(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {}
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'module.encode.cnn0.bias',
        'module.encode.cnn1.weight',
        'module.encode.cnn1.bias',
        'module.encode.cnn2.weight',
        'module.encode.cnn2.bias',
        'module.encode.cnn3.weight',
        'module.encode.cnn3.bias',
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'module.caltime.0.weight',
        'module.caltime.0.bias',
        'module.caltime.2.weight',
        'module.caltime.2.bias',
        'module.caltime.4.weight',
        'module.caltime.4.bias',
        'module.caltime.6.weight',
        'module.caltime.6.bias',
        'module.caltime.8.weight',
        'module.caltime.8.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'caltime.8.bias',  # thos is because for some reason, this arch is detected when loading gmfss pro's rife component.
        'encoder.pyramid1.0.0.weight',  # ifrnet
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife46IFNET import IFNet

        return IFNet()


class RIFE47(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {}
    excluded_keys: list = [
        'module.encode.cnn0.bias',
        'module.encode.cnn1.weight',
        'module.encode.cnn1.bias',
        'module.encode.cnn2.weight',
        'module.encode.cnn2.bias',
        'module.encode.cnn3.weight',
        'module.encode.cnn3.bias',
        'module.caltime.0.weight',
        'module.caltime.0.bias',
        'module.caltime.2.weight',
        'module.caltime.2.bias',
        'module.caltime.4.weight',
        'module.caltime.4.bias',
        'module.caltime.6.weight',
        'module.caltime.6.bias',
        'module.caltime.8.weight',
        'module.caltime.8.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife47IFNET import IFNet

        return IFNet


class RIFE413(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {}

    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'module.caltime.0.weight',
        'module.caltime.0.bias',
        'module.caltime.2.weight',
        'module.caltime.2.bias',
        'module.caltime.4.weight',
        'module.caltime.4.bias',
        'module.caltime.6.weight',
        'module.caltime.6.bias',
        'module.caltime.8.weight',
        'module.caltime.8.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife413IFNET import IFNet

        return IFNet


class RIFE420(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {'module.block0.conv0.1.0.bias': 'torch.Size([384])'}
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife420IFNET import IFNet

        return IFNet


class RIFE421(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {'module.block0.conv0.1.0.bias': 'torch.Size([256])'}
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    def module():
        from .RIFE.rife421IFNET import IFNet

        return IFNet


class RIFE422lite(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {'module.block0.conv0.1.0.bias': 'torch.Size([192])'}
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife422_liteIFNET import IFNet

        return IFNet


class RIFE425(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {
        'module.block4.lastconv.0.bias': 'torch.Size([52])',
        'module.block0.conv0.0.0.weight': 'torch.Size([96, 15, 3, 3])',
    }
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife425IFNET import IFNet

        return IFNet


class RIFE425_heavy(Arch):
    base_arch: str = 'rife'
    unique_shapes: dict = {
        'module.block0.lastconv.0.bias': 'torch.Size([52])',
        'module.block0.conv0.0.0.weight': 'torch.Size([96, 39, 3, 3])',
    }
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .RIFE.rife425_heavyIFNET import IFNet

        return IFNet


class GMFSS(Arch):
    base_arch: str = 'gmfss'
    unique_shapes: dict = {
        'transformer.layers.4.self_attn.merge.weight': 'torch.Size([128, 128])'
    }
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'fnet.layer1.0.conv1.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    @staticmethod
    def module() -> torch.nn.Module:
        from .GMFSS.GMFSS import GMFSS

        return GMFSS


class GIMM(Arch):
    base_arch: str = 'gimm'
    unique_shapes: dict = {
        'fnet.layer1.0.conv1.weight': 'torch.Size([64, 64, 3, 3])'
    }
    excluded_keys: list = [
        'module.encode.0.weight',
        'module.encode.0.bias',
        'module.encode.1.weight',
        'module.encode.1.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'encoder.pyramid1.0.0.weight',
    ]

    def module() -> torch.nn.Module:
        from .GIMM.GIMM import GIMMVFI_R

        return GIMMVFI_R


class IFRNET(Arch):
    base_arch: str = 'ifrnet'
    unique_shapes: dict = {
        'encoder.pyramid1.0.0.weight': 'torch.Size([32, 3, 3, 3])',
    }
    excluded_keys: list = [
        'module.caltime.8.bias',
        'module.block4.lastconv.0.bias',
        'transformer.layers.4.self_attn.merge.weight',
        'fnet.layer1.0.conv1.weight',
        'caltime.8.bias',
    ]

    def module() -> torch.nn.Module:
        from .IFRNET.IFRNet import IFRNet

        return IFRNet


archs = [
    RIFE46,
    RIFE47,
    RIFE413,
    RIFE420,
    RIFE421,
    RIFE422lite,
    RIFE425,
    GMFSS,
    GIMM,
    IFRNET,
]


class ArchDetect:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        if type(pkl_path) is OrderedDict:
            self.state_dict = pkl_path
        else:
            self.state_dict = torch.load(
                pkl_path, weights_only=True, map_location=torch.device('cpu')
            )
        # this is specific to loading gmfss, as its loaded in as one big pkl
        if 'flownet' in self.state_dict:
            self.state_dict = self.state_dict[
                'flownet'
            ]  # load in GMFSS FLOWNET
        if 'raft' in self.state_dict:  # load in GIMM RAFT
            self.state_dict = self.state_dict['raft']
        self.keys = self.state_dict.keys()
        self.key_shape_pair = self.detect_weights()
        self.detected_arch = self.compare_arch()
        del self.state_dict

    def detect_weights(self) -> dict:
        key_shape_pair = {}
        for key in self.keys:
            key_shape_pair[key] = str(self.state_dict[key].shape)
        return key_shape_pair

    def compare_arch(self) -> Arch:
        arch_dict = {}
        for arch in Arch.__subclasses__():
            arch: Arch
            arch_dict[arch] = True
            # see if there are any excluded keys in the state_dict
            for key, shape in self.key_shape_pair.items():
                if key in arch.excluded_keys:
                    arch_dict[arch] = False
                    continue
            # unique shapes will return tuple if there is no unique shape, dict if there is
            # parse the unique shape and compare with the state_dict shape
            if type(arch.unique_shapes) is dict:
                for key1, uniqueshape1 in arch.unique_shapes.items():
                    try:  # the key might not be in the state_dict
                        if not str(self.state_dict[key1].shape) == str(
                            uniqueshape1
                        ):
                            arch_dict[arch] = False
                    except KeyError:
                        arch_dict[arch] = False

        for key, value in arch_dict.items():
            if value:
                return key

    """def getArchName(self):
        return "ifrnet"
        return self.detected_arch.__name__

    def getArchBase(self):
        return "ifrnet"
        return self.detected_arch.base_arch

    def getArchModule(self):
        return IFRNET
        return self.detected_arch.module()"""

    def getArchName(self):
        return self.detected_arch.__name__

    def getArchBase(self):
        return self.detected_arch.base_arch

    def getArchModule(self):
        return self.detected_arch.module()


if __name__ == '__main__':
    import os

    for file in os.listdir('.'):
        if '.pkl' in file:
            ra = ArchDetect(file)
            print(ra.getArchName())
