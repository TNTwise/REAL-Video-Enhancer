
import tensorrt as trt


def ONNX2TRT( args, calib=None):
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(G_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, trt.OnnxParser(network, G_LOGGER) as parser:
            builder.max_batch_size = args.batch_size

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30

            profile = builder.create_optimization_profile()
            profile.set_shape(
                "input", (1, 3, 8, 8), (1, 3, 1080, 1920), (1, 3, 1080, 1920)
            )
            config.add_optimization_profile(profile)
            # builder.max_workspace_size = 1 << 30
            if args.mode.lower() == "int8":
                assert builder.platform_has_fast_int8, "not support int8"
                assert calib is not None, "need calib!"
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib
            elif args.mode.lower() == "fp16":
                assert builder.platform_has_fast_fp16, "not support fp16"
                config.set_flag(trt.BuilderFlag.FP16)

            print("Loading ONNX file from path {}...".format(args.onnx_file_path))
            with open(args.onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    for e in range(parser.num_errors):
                        print(parser.get_error(e))
                    raise TypeError("Parser parse failed.")

            print("Parsing ONNX file complete!")

            print(
                "Building an engine from file {}; this may take a while...".format(
                    args.onnx_file_path
                )
            )
            engine = builder.build_engine(network, config)
            if engine is not None:
                print("Create engine success! ")
            else:
                print("ERROR: Create engine failed! ")
                return

            print("Saving TRT engine file to path {}...".format(args.engine_file_path))
            with open(args.engine_file_path, "wb") as f:
                f.write(engine.serialize())

            print("Engine file has already saved to {}!".format(args.engine_file_path))

            return engine

def loadEngine2TensorRT(self, filepath):
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    
args = type("", (), {})()
args.mode = "fp16"
args.onnx_file_path = "2x_ModernSpanimationV1_fp16_op17.onnx"
args.batch_size = 1
args.engine_file_path = r"engine.trt"

#engine = loadEngine2TensorRT(args.engine_file_path)
engine = ONNX2TRT(args)