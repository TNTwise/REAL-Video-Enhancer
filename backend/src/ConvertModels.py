import os
import torch
import pnnx

from .Util import log, warnAndLog
from .UpscaleTorch import loadTorchModel

cwd = os.getcwd()


class ConvertModels:
    def __init__(
        self,
        modelName: str,
        pathToModel: str,
        inputFormat: str = "pytorch",
        outputFormat: str = "onnx",
        ncnnConversionMethod: str = "onnx",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        opset: int = 18,
        onnxDynamicAxess: dict = None,
    ):
        self.modelName = modelName
        self.pathToModel = pathToModel
        self.basepath = os.path.dirname(pathToModel)
        self.inputFormat = inputFormat
        self.outputFormat = outputFormat
        self.ncnnConversionMethod = ncnnConversionMethod
        self.device = device
        self.opset = opset
        self.dtype = dtype
        self.onnxDynamicAxes = onnxDynamicAxess

    def convertModel(self):
        self.input = torch.rand(1, 3, 256, 256).to(device=self.device, dtype=self.dtype)
        if self.outputFormat == "onnx":
            self.convertPyTorchToONNX()
        if self.outputFormat == "ncnn":
            self.convertPytorchToNCNN()

    @torch.inference_mode
    def convertPyTorchToONNX(self):
        # load model
        model = loadModel(self.pathToModel, self.dtype, self.device).model
        state_dict = model.state_dict()
        model.eval()
        model.load_state_dict(state_dict, strict=True)
        torch.onnx.export(
            model,
            self.input,
            f"{self.modelName}_op{self.opset}_{self.dtype}.onnx",
            opset_version=self.opset,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamic_axes=self.onnxDynamicAxes,
        )

    def fixNCNNParamInput(self, paramFile):
        """
        replaces in0 with data and out0 with output in a ncnn param file
        """
        with open(paramFile, "r") as f:
            lines = f.readlines()

        with open(paramFile, "w") as f:
            for line in lines:
                line = line.replace("in0", "data")
                line = line.replace("out0", "output")
                f.write(line)

    def convertPytorchToNCNN(self):
        """
        Takes in a pytorch model, and uses JIT tracing with PNNX to convert it to ncnn.
        This method removed unnecessary files, and fixes the param file to be compadible with most NCNN appliacitons.
        """
        model = loadTorchModel(self.pathToModel, torch.float32, self.device).model
        model.eval()
        input = torch.rand(1, 3, 256, 256)
        jitTracedModelLocation = self.pathToModel + ".pt"
        jitTracedModel = torch.jit.trace(model, input)
        jitTracedModel.save(jitTracedModelLocation)

        pnnxBinLocation = self.pathToModel + ".pnnx.bin"
        pnnxParamLocation = self.pathToModel + ".pnnx.param"
        pnnxPythonLocation = self.pathToModel + "_pnnx.py"
        pnnxOnnxLocation = self.pathToModel + ".pnnx.onnx"
        ncnnPythonLocation = self.pathToModel + "_ncnn.py"
        ncnnParamLocation = self.pathToModel + ".ncnn.param"

        # pnnx gives out a lot of weird errors, so i will be try/excepting this.
        # usually nothing goes wrong, but it cant take in the pnnxbin/pnnxparam location on windows.

        try:
            model = pnnx.convert(
                ptpath=jitTracedModelLocation,
                inputs=input,
                device=self.device,
                optlevel=2,
                fp16=self.dtype == torch.float16,
                pnnxbin=pnnxBinLocation,
                pnnxparam=pnnxParamLocation,
                pnnxpy=pnnxPythonLocation,
                pnnxonnx=pnnxOnnxLocation,
                ncnnpy=ncnnPythonLocation,
            )
        except Exception as e:
            warnAndLog(f"Something may have gone wrong with conversion! {e}")

        # remove stuff that we dont need
        try:
            os.remove(jitTracedModelLocation)
            os.remove(pnnxBinLocation)
            os.remove(pnnxParamLocation)
            os.remove(pnnxPythonLocation)
            os.remove(pnnxOnnxLocation)
            os.remove(ncnnPythonLocation)
        except Exception as e:
            warnAndLog(f"Could not remove unnecessary files. {e} ")
        try:
            os.remove(os.path.join(cwd, "debug.bin"))
            os.remove(os.path.join(cwd, "debug.param"))
            os.remove(os.path.join(cwd, "debug2.bin"))
            os.remove(os.path.join(cwd, "debug2.param"))
        except:
            warnAndLog(f"Failed to remove debug pnnx files. {e} ")
        self.fixNCNNParamInput(ncnnParamLocation)
