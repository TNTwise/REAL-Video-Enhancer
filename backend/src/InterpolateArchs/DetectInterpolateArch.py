import torch


class RIFE46:
    def __init__():
        pass

    def __name__():
        return "rife46"

    def unique_shapes() -> tuple:
        return ()

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "module.encode.cnn0.bias",
            "module.encode.cnn1.weight",
            "module.encode.cnn1.bias",
            "module.encode.cnn2.weight",
            "module.encode.cnn2.bias",
            "module.encode.cnn3.weight",
            "module.encode.cnn3.bias",
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "module.caltime.0.weight",
            "module.caltime.0.bias",
            "module.caltime.2.weight",
            "module.caltime.2.bias",
            "module.caltime.4.weight",
            "module.caltime.4.bias",
            "module.caltime.6.weight",
            "module.caltime.6.bias",
            "module.caltime.8.weight",
            "module.caltime.8.bias",
            "module.block4.lastconv.0.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class RIFE47:
    def __init__():
        pass

    def __name__():
        return "rife47"

    def unique_shapes() -> tuple:
        return ()

    def excluded_keys() -> tuple:
        return [
            "module.encode.cnn0.bias",
            "module.encode.cnn1.weight",
            "module.encode.cnn1.bias",
            "module.encode.cnn2.weight",
            "module.encode.cnn2.bias",
            "module.encode.cnn3.weight",
            "module.encode.cnn3.bias",
            "module.caltime.0.weight",
            "module.caltime.0.bias",
            "module.caltime.2.weight",
            "module.caltime.2.bias",
            "module.caltime.4.weight",
            "module.caltime.4.bias",
            "module.caltime.6.weight",
            "module.caltime.6.bias",
            "module.caltime.8.weight",
            "module.caltime.8.bias",
            "module.block4.lastconv.0.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class RIFE413:
    def __init__():
        pass

    def __name__():
        return "rife413"

    def unique_shapes() -> tuple:
        return ()

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "module.caltime.0.weight",
            "module.caltime.0.bias",
            "module.caltime.2.weight",
            "module.caltime.2.bias",
            "module.caltime.4.weight",
            "module.caltime.4.bias",
            "module.caltime.6.weight",
            "module.caltime.6.bias",
            "module.caltime.8.weight",
            "module.caltime.8.bias",
            "module.block4.lastconv.0.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class RIFE420:
    def __init__():
        pass

    def __name__():
        return "rife413"

    def unique_shapes() -> dict:
        return {"module.block0.conv0.1.0.bias": "torch.Size([384])"}

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "module.block4.lastconv.0.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class RIFE421:
    def __init__():
        pass

    def __name__():
        return "rife413"

    def unique_shapes() -> dict:
        return {"module.block0.conv0.1.0.bias": "torch.Size([256])"}

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "module.block4.lastconv.0.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class RIFE422lite:
    def __init__():
        pass

    def __name__():
        return "rife413"

    def unique_shapes() -> dict:
        return {"module.block0.conv0.1.0.bias": "torch.Size([192])"}

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "module.block4.lastconv.0.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class RIFE425:
    def __init__():
        pass

    def __name__():
        return "rife413"

    def unique_shapes() -> dict:
        return {"module.block4.lastconv.0.bias": "torch.Size([52])"}

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
            "transformer.layers.4.self_attn.merge.weight",
        ]


class GMFSS:
    def __init__():
        pass

    def __name__():
        return "rife413"

    def unique_shapes() -> dict:
        return {"transformer.layers.4.self_attn.merge.weight": "torch.Size([128, 128])"}

    def excluded_keys() -> tuple:
        return [
            "module.encode.0.weight",
            "module.encode.0.bias",
            "module.encode.1.weight",
            "module.encode.1.bias",
        ]


archs = [RIFE46, RIFE47, RIFE413, RIFE420, RIFE421, RIFE422lite, RIFE425, GMFSS]


class ArchDetect:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        self.state_dict = torch.load(pkl_path, weights_only=True)
        self.keys = self.state_dict.keys()
        self.key_shape_pair = self.detect_weights()
        self.detected_arch = self.compare_arch()

    def detect_weights(self) -> dict:
        key_shape_pair = {}
        for key in self.keys:
            key_shape_pair[key] = str(self.state_dict[key].shape)
        return key_shape_pair

    def compare_arch(self) -> tuple:
        arch_dict = {}
        for arch in archs:
            arch_dict[arch.__name__] = True
            # see if there are any excluded keys in the state_dict
            for key, shape in self.key_shape_pair.items():
                if key in arch.excluded_keys():
                    arch_dict[arch.__name__] = False
                    continue
            # unique shapes will return tuple if there is no unique shape, dict if there is
            # parse the unique shape and compare with the state_dict shape
            if type(arch.unique_shapes()) is dict:
                for key1, uniqueshape1 in arch.unique_shapes().items():
                    try:  # the key might not be in the state_dict
                        if not str(self.state_dict[key1].shape) == str(uniqueshape1):
                            arch_dict[arch.__name__] = False
                    except:
                        arch_dict[arch.__name__] = False

        for key, value in arch_dict.items():
            if value:
                return key

    def getArch(self):
        return self.detected_arch


if __name__ == "__main__":
    pkl_path = "rife4.15.pkl"
    ra = ArchDetect(pkl_path)
    print(ra.getArch())
