# https://github.com/chaiNNer-org/chaiNNer/blob/2aa0b46233ba8cd90d4bb405e2bc6e16a3430546/backend/src/nodes/impl/ncnn/model.py


from __future__ import annotations

import os
from copy import deepcopy
from io import BufferedReader, StringIO
from json import loads as jload
from pathlib import Path

import numpy as np
from typing import TypeVar
import sys

from upscale_ncnn_py import UPSCALE

import cv2

T = TypeVar("T")


def checked_cast(t: type[T], value: object) -> T:
    assert isinstance(value, t), f"Value is {type(value)}, must be type {t}"
    return value


schemaf = """
{
  "AbsVal": {
    "weightOrder": {}
  },
  "ArgMax": {
    "0": {
      "paramId": "0",
      "paramPhase": "out_max_val",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "topk",
      "defaultValue": 1
    },
    "weightOrder": {}
  },
  "BatchNorm": {
    "0": {
      "paramId": "0",
      "paramPhase": "channels",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "eps",
      "defaultValue": 0.0
    },
    "weightOrder": {
      "slope": [1],
      "mean": [1],
      "variance": [1],
      "bias": [1]
    }
  },
  "Bias": {
    "0": {
      "paramId": "0",
      "paramPhase": "bias_data_size",
      "defaultValue": 0
    },
    "weightOrder": {
      "bias": [1]
    }
  },
  "BinaryOp": {
    "0": {
      "paramId": "0",
      "paramPhase": "op_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "with_scalar",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "b",
      "defaultValue": 0.0
    },
    "weightOrder": {}
  },
  "BNLL": {
    "weightOrder": {}
  },
  "Cast": {
    "0": {
      "paramId": "0",
      "paramPhase": "type_from",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "type_to",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Clip": {
    "0": {
      "paramId": "0",
      "paramPhase": "min",
      "defaultValue": "-FLT_MAX"
    },
    "1": {
      "paramId": "1",
      "paramPhase": "max",
      "defaultValue": "FLT_MAX"
    },
    "weightOrder": {}
  },
  "Concat": {
    "0": {
      "paramId": "0",
      "paramPhase": "axis",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Convolution": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "int8_scale_term",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "pad_value",
      "defaultValue": 0.0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "dynamic_weight",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Convolution1D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "pad_value",
      "defaultValue": 0.0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "dynamic_weight",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Convolution3D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "17": {
      "paramId": "17",
      "paramPhase": "pad_behind",
      "defaultValue": "pad_front"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "pad_value",
      "defaultValue": 0.0
    },
    "21": {
      "paramId": "21",
      "paramPhase": "kernel_d",
      "defaultValue": "kernel_w"
    },
    "22": {
      "paramId": "22",
      "paramPhase": "dilation_d",
      "defaultValue": "dilation_w"
    },
    "23": {
      "paramId": "23",
      "paramPhase": "stride_d",
      "defaultValue": "stride_w"
    },
    "24": {
      "paramId": "24",
      "paramPhase": "pad_front",
      "defaultValue": "pad_left"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "ConvolutionDepthWise": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "8": {
      "paramId": "8",
      "paramPhase": "int8_scale_term",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "pad_value",
      "defaultValue": 0.0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "dynamic_weight",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "ConvolutionDepthWise1D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "pad_value",
      "defaultValue": 0.0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "dynamic_weight",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "ConvolutionDepthWise3D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "17": {
      "paramId": "17",
      "paramPhase": "pad_behind",
      "defaultValue": "pad_front"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "pad_value",
      "defaultValue": 0.0
    },
    "21": {
      "paramId": "21",
      "paramPhase": "kernel_d",
      "defaultValue": "kernel_w"
    },
    "22": {
      "paramId": "22",
      "paramPhase": "dilation_d",
      "defaultValue": "dilation_w"
    },
    "23": {
      "paramId": "23",
      "paramPhase": "stride_d",
      "defaultValue": "stride_w"
    },
    "24": {
      "paramId": "24",
      "paramPhase": "pad_front",
      "defaultValue": "pad_left"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Crop": {
    "0": {
      "paramId": "0",
      "paramPhase": "woffset",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "hoffset",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "coffset",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "outw",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "outh",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "outc",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "woffset2",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "hoffset2",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "coffset2",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "starts",
      "defaultValue": []
    },
    "10": {
      "paramId": "10",
      "paramPhase": "ends",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "axes",
      "defaultValue": []
    },
    "weightOrder": {}
  },
  "Deconvolution": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "output_pad_right",
      "defaultValue": 0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "output_pad_bottom",
      "defaultValue": "output_pad_right"
    },
    "20": {
      "paramId": "20",
      "paramPhase": "output_w",
      "defaultValue": 0
    },
    "21": {
      "paramId": "21",
      "paramPhase": "output_h",
      "defaultValue": "output_w"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Deconvolution1D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "output_pad_right",
      "defaultValue": 0
    },
    "20": {
      "paramId": "20",
      "paramPhase": "output_w",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Deconvolution3D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "17": {
      "paramId": "17",
      "paramPhase": "pad_behind",
      "defaultValue": "pad_front"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "output_pad_right",
      "defaultValue": 0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "output_pad_bottom",
      "defaultValue": "output_pad_right"
    },
    "20": {
      "paramId": "20",
      "paramPhase": "output_pad_behind",
      "defaultValue": "output_pad_right"
    },
    "21": {
      "paramId": "21",
      "paramPhase": "kernel_d",
      "defaultValue": "kernel_w"
    },
    "22": {
      "paramId": "22",
      "paramPhase": "dilation_d",
      "defaultValue": "dilation_w"
    },
    "23": {
      "paramId": "23",
      "paramPhase": "stride_d",
      "defaultValue": "stride_w"
    },
    "24": {
      "paramId": "24",
      "paramPhase": "pad_front",
      "defaultValue": "pad_left"
    },
    "25": {
      "paramId": "25",
      "paramPhase": "output_w",
      "defaultValue": 0
    },
    "26": {
      "paramId": "26",
      "paramPhase": "output_h",
      "defaultValue": "output_w"
    },
    "27": {
      "paramId": "27",
      "paramPhase": "output_d",
      "defaultValue": "output_w"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "DeconvolutionDepthWise": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "output_pad_right",
      "defaultValue": 0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "output_pad_bottom",
      "defaultValue": "output_pad_right"
    },
    "20": {
      "paramId": "20",
      "paramPhase": "output_w",
      "defaultValue": 0
    },
    "21": {
      "paramId": "21",
      "paramPhase": "output_h",
      "defaultValue": "output_w"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "DeconvolutionDepthWise1D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "output_pad_right",
      "defaultValue": 0
    },
    "20": {
      "paramId": "20",
      "paramPhase": "output_w",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "DeconvolutionDepthWise3D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "17": {
      "paramId": "17",
      "paramPhase": "pad_behind",
      "defaultValue": "pad_front"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "output_pad_right",
      "defaultValue": 0
    },
    "19": {
      "paramId": "19",
      "paramPhase": "output_pad_bottom",
      "defaultValue": "output_pad_right"
    },
    "20": {
      "paramId": "20",
      "paramPhase": "output_pad_behind",
      "defaultValue": "output_pad_right"
    },
    "21": {
      "paramId": "21",
      "paramPhase": "kernel_d",
      "defaultValue": "kernel_w"
    },
    "22": {
      "paramId": "22",
      "paramPhase": "dilation_d",
      "defaultValue": "dilation_w"
    },
    "23": {
      "paramId": "23",
      "paramPhase": "stride_d",
      "defaultValue": "stride_w"
    },
    "24": {
      "paramId": "24",
      "paramPhase": "pad_front",
      "defaultValue": "pad_left"
    },
    "25": {
      "paramId": "25",
      "paramPhase": "output_w",
      "defaultValue": 0
    },
    "26": {
      "paramId": "26",
      "paramPhase": "output_h",
      "defaultValue": "output_w"
    },
    "27": {
      "paramId": "27",
      "paramPhase": "output_d",
      "defaultValue": "output_w"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "DeformableConv2D": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "dilation_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "4": {
      "paramId": "4",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "int8_scale_term",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "dilation_h",
      "defaultValue": "dilation_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Dequantize": {
    "0": {
      "paramId": "0",
      "paramPhase": "scale",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "bias_data_size",
      "defaultValue": 0
    },
    "weightOrder": {
      "scale": [1],
      "bias": [1]
    }
  },
  "DetectionOutput": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_class",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "nms_threshold",
      "defaultValue": 0.05
    },
    "2": {
      "paramId": "2",
      "paramPhase": "nms_top_k",
      "defaultValue": 300
    },
    "3": {
      "paramId": "3",
      "paramPhase": "keep_top_k",
      "defaultValue": 100
    },
    "4": {
      "paramId": "4",
      "paramPhase": "confidence_threshold",
      "defaultValue": 0.5
    },
    "5": {
      "paramId": "5",
      "paramPhase": "variances[0]",
      "defaultValue": 0.1
    },
    "6": {
      "paramId": "6",
      "paramPhase": "variances[1]",
      "defaultValue": 0.1
    },
    "7": {
      "paramId": "7",
      "paramPhase": "variances[2]",
      "defaultValue": 0.2
    },
    "8": {
      "paramId": "8",
      "paramPhase": "variances[3]",
      "defaultValue": 0.2
    },
    "weightOrder": {}
  },
  "Dropout": {
    "0": {
      "paramId": "0",
      "paramPhase": "scale",
      "defaultValue": 1.0
    },
    "weightOrder": {}
  },
  "Eltwise": {
    "0": {
      "paramId": "0",
      "paramPhase": "op_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "coeffs",
      "defaultValue": []
    },
    "weightOrder": {}
  },
  "ELU": {
    "0": {
      "paramId": "0",
      "paramPhase": "alpha",
      "defaultValue": 0.1
    },
    "weightOrder": {}
  },
  "Embed": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "input_dim",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight": [1, 3, 10],
      "bias": [1]
    }
  },
  "Exp": {
    "0": {
      "paramId": "0",
      "paramPhase": "base",
      "defaultValue": -1.0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "scale",
      "defaultValue": 1.0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "shift",
      "defaultValue": 0.0
    },
    "weightOrder": {}
  },
  "ExpandDims": {
    "0": {
      "paramId": "0",
      "paramPhase": "expand_w",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "expand_h",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "expand_c",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "axes",
      "defaultValue": []
    },
    "weightOrder": {}
  },
  "Flatten": {
    "weightOrder": {}
  },
  "GELU": {
    "0": {
      "paramId": "0",
      "paramPhase": "fast_gelu",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Gemm": {
    "0": {
      "paramId": "0",
      "paramPhase": "alpha",
      "defaultValue": 1.0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "beta",
      "defaultValue": 1.0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "transA",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "transB",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "GLU": {
    "0": {
      "paramId": "0",
      "paramPhase": "axis",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "GridSample": {
    "0": {
      "paramId": "0",
      "paramPhase": "sample_type",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "padding_mode",
      "defaultValue": 1
    },
    "2": {
      "paramId": "2",
      "paramPhase": "align_corner",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "GroupNorm": {
    "0": {
      "paramId": "0",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "channels",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "eps",
      "defaultValue": 0.001
    },
    "3": {
      "paramId": "3",
      "paramPhase": "affine",
      "defaultValue": 1
    },
    "weightOrder": {
      "gamma": [1],
      "beta": [1]
    }
  },
  "GRU": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "direction",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight_xc": [1, 3, 10],
      "bias": [1, 3, 10],
      "weight_hc": [1, 3, 10]
    }
  },
  "HardSigmoid": {
    "0": {
      "paramId": "0",
      "paramPhase": "alpha",
      "defaultValue": 0.2
    },
    "1": {
      "paramId": "1",
      "paramPhase": "beta",
      "defaultValue": 0.5
    },
    "weightOrder": {}
  },
  "HardSwish": {
    "0": {
      "paramId": "0",
      "paramPhase": "alpha",
      "defaultValue": 0.2
    },
    "1": {
      "paramId": "1",
      "paramPhase": "beta",
      "defaultValue": 0.5
    },
    "weightOrder": {}
  },
  "InnerProduct": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "int8_scale_term",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "weightOrder": {
      "weight": [1, 10],
      "bias": [1]
    }
  },
  "Input": {
    "0": {
      "paramId": "0",
      "paramPhase": "w",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "h",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "c",
      "defaultValue": 0
    },
    "11": {
      "paramId": "11",
      "paramPhase": "d",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "InstanceNorm": {
    "0": {
      "paramId": "0",
      "paramPhase": "channels",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "eps",
      "defaultValue": 0.001
    },
    "2": {
      "paramId": "2",
      "paramPhase": "affine",
      "defaultValue": 1
    },
    "weightOrder": {
      "gamma": [1],
      "bias": [1]
    }
  },
  "Interp": {
    "0": {
      "paramId": "0",
      "paramPhase": "resize_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "height_scale",
      "defaultValue": 1.0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "width_scale",
      "defaultValue": 1.0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "output_height",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "output_width",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "dynamic_target_size",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "align_corner",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "LayerNorm": {
    "0": {
      "paramId": "0",
      "paramPhase": "affine_size",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "eps",
      "defaultValue": 0.001
    },
    "2": {
      "paramId": "2",
      "paramPhase": "affine",
      "defaultValue": 1
    },
    "weightOrder": {
      "gamma": [1],
      "beta": [1]
    }
  },
  "Log": {
    "0": {
      "paramId": "0",
      "paramPhase": "base",
      "defaultValue": -1.0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "scale",
      "defaultValue": 1.0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "shift",
      "defaultValue": 0.0
    },
    "weightOrder": {}
  },
  "LRN": {
    "0": {
      "paramId": "0",
      "paramPhase": "region_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "local_size",
      "defaultValue": 5
    },
    "2": {
      "paramId": "2",
      "paramPhase": "alpha",
      "defaultValue": 1.0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "beta",
      "defaultValue": 0.75
    },
    "4": {
      "paramId": "4",
      "paramPhase": "bias",
      "defaultValue": 1.0
    },
    "weightOrder": {}
  },
  "LSTM": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "weight_data_size",
      "defaultValue": 1
    },
    "2": {
      "paramId": "2",
      "paramPhase": "direction",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight_xc": [1, 3, 10],
      "bias_c": [1, 3, 10],
      "weight_hc": [1, 3, 10],
      "weight_hr": [1, 3, 10]
    }
  },
  "MemoryData": {
    "0": {
      "paramId": "0",
      "paramPhase": "w",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "h",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "c",
      "defaultValue": 0
    },
    "weightOrder": {
      "data": [1]
    }
  },
  "Mish": {
    "weightOrder": {}
  },
  "MultiHeadAttention": {
    "0": {
      "paramId": "0",
      "paramPhase": "embed_dim",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "num_head",
      "defaultValue": 1
    },
    "2": {
      "paramId": "2",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "weightOrder": {
      "q_weight": [1, 3, 10],
      "q_bias": [1],
      "k_weight": [1, 3, 10],
      "k_bias": [1],
      "v_weight": [1, 3, 10],
      "v_bias": [1],
      "out_weight": [1, 3, 10],
      "out_bias": [1]
    }
  },
  "MVN": {
    "0": {
      "paramId": "0",
      "paramPhase": "normalize_variance",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "across_channels",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "eps",
      "defaultValue": 0.0001
    },
    "weightOrder": {}
  },
  "Noop": {
    "weightOrder": {}
  },
  "Normalize": {
    "0": {
      "paramId": "0",
      "paramPhase": "across_spatial",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "channel_shared",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "eps",
      "defaultValue": 0.0001
    },
    "3": {
      "paramId": "3",
      "paramPhase": "scale_data_size",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "across_channel",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "eps_mode",
      "defaultValue": 0
    },
    "weightOrder": {
      "scale": [1]
    }
  },
  "Packing": {
    "0": {
      "paramId": "0",
      "paramPhase": "out_packing",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "use_padding",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "cast_type_from",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "cast_type_to",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "storage_type_from",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "storage_type_to",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Padding": {
    "0": {
      "paramId": "0",
      "paramPhase": "top",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "bottom",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "left",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "right",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "type",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "value",
      "defaultValue": 0.0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "per_channel_pad_data_size",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "front",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "behind",
      "defaultValue": 0
    },
    "weightOrder": {
      "per_channel_pad_data": [1]
    }
  },
  "Permute": {
    "0": {
      "paramId": "0",
      "paramPhase": "order_type",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "PixelShuffle": {
    "0": {
      "paramId": "0",
      "paramPhase": "upscale_factor",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "mode",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Pooling": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooling_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "global_pooling",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "pad_mode",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "avgpool_count_include_pad",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "adaptive_pooling",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "out_w",
      "defaultValue": 0
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "out_h",
      "defaultValue": "out_w"
    },
    "weightOrder": {}
  },
  "Pooling1D": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooling_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "global_pooling",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "pad_mode",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "avgpool_count_include_pad",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "adaptive_pooling",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "out_w",
      "defaultValue": 0
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "weightOrder": {}
  },
  "Pooling3D": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooling_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "kernel_w",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "stride_w",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "pad_left",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "global_pooling",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "pad_mode",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "avgpool_count_include_pad",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "adaptive_pooling",
      "defaultValue": 0
    },
    "8": {
      "paramId": "8",
      "paramPhase": "out_w",
      "defaultValue": 0
    },
    "11": {
      "paramId": "11",
      "paramPhase": "kernel_h",
      "defaultValue": "kernel_w"
    },
    "12": {
      "paramId": "12",
      "paramPhase": "stride_h",
      "defaultValue": "stride_w"
    },
    "13": {
      "paramId": "13",
      "paramPhase": "pad_top",
      "defaultValue": "pad_left"
    },
    "14": {
      "paramId": "14",
      "paramPhase": "pad_right",
      "defaultValue": "pad_left"
    },
    "15": {
      "paramId": "15",
      "paramPhase": "pad_bottom",
      "defaultValue": "pad_top"
    },
    "16": {
      "paramId": "16",
      "paramPhase": "pad_behind",
      "defaultValue": "pad_front"
    },
    "18": {
      "paramId": "18",
      "paramPhase": "out_h",
      "defaultValue": "out_w"
    },
    "21": {
      "paramId": "21",
      "paramPhase": "kernel_d",
      "defaultValue": "kernel_w"
    },
    "22": {
      "paramId": "22",
      "paramPhase": "stride_d",
      "defaultValue": "stride_w"
    },
    "23": {
      "paramId": "23",
      "paramPhase": "pad_front",
      "defaultValue": "pad_left"
    },
    "28": {
      "paramId": "28",
      "paramPhase": "out_d",
      "defaultValue": "out_w"
    },
    "weightOrder": {}
  },
  "Power": {
    "0": {
      "paramId": "0",
      "paramPhase": "power",
      "defaultValue": 1.0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "scale",
      "defaultValue": 1.0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "shift",
      "defaultValue": 0.0
    },
    "weightOrder": {}
  },
  "PReLU": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_slope",
      "defaultValue": 0
    },
    "weightOrder": {
      "slope": [1]
    }
  },
  "PriorBox": {
    "0": {
      "paramId": "0",
      "paramPhase": "min_sizes",
      "defaultValue": []
    },
    "1": {
      "paramId": "1",
      "paramPhase": "max_sizes",
      "defaultValue": []
    },
    "2": {
      "paramId": "2",
      "paramPhase": "aspect_ratios",
      "defaultValue": []
    },
    "3": {
      "paramId": "3",
      "paramPhase": "variances[0]",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "variances[1]",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "variances[2]",
      "defaultValue": 0
    },
    "6": {
      "paramId": "6",
      "paramPhase": "variances[3]",
      "defaultValue": 0
    },
    "7": {
      "paramId": "7",
      "paramPhase": "flip",
      "defaultValue": 1
    },
    "8": {
      "paramId": "8",
      "paramPhase": "clip",
      "defaultValue": 0
    },
    "9": {
      "paramId": "9",
      "paramPhase": "image_width",
      "defaultValue": 0
    },
    "10": {
      "paramId": "10",
      "paramPhase": "image_height",
      "defaultValue": 0
    },
    "11": {
      "paramId": "11",
      "paramPhase": "step_width",
      "defaultValue": -233
    },
    "12": {
      "paramId": "12",
      "paramPhase": "step_height",
      "defaultValue": -233
    },
    "13": {
      "paramId": "13",
      "paramPhase": "offset",
      "defaultValue": 0
    },
    "14": {
      "paramId": "14",
      "paramPhase": "step_mmdetection",
      "defaultValue": 0
    },
    "15": {
      "paramId": "15",
      "paramPhase": "center_mmdetection",
      "defaultValue": 0
    },
    "weightOrder": {
      "slope": [1]
    }
  },
  "Proposal": {
    "0": {
      "paramId": "0",
      "paramPhase": "feat_stride",
      "defaultValue": 16
    },
    "1": {
      "paramId": "1",
      "paramPhase": "base_size",
      "defaultValue": 16
    },
    "2": {
      "paramId": "2",
      "paramPhase": "pre_nms_topN",
      "defaultValue": 6000
    },
    "3": {
      "paramId": "3",
      "paramPhase": "after_nms_topN",
      "defaultValue": 300
    },
    "4": {
      "paramId": "4",
      "paramPhase": "num_thresh",
      "defaultValue": 0.7
    },
    "5": {
      "paramId": "5",
      "paramPhase": "min_size",
      "defaultValue": 16
    },
    "weightOrder": {}
  },
  "PSROIPooling": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooled_width",
      "defaultValue": 7
    },
    "1": {
      "paramId": "1",
      "paramPhase": "pooled_height",
      "defaultValue": 7
    },
    "2": {
      "paramId": "2",
      "paramPhase": "spatial_scale",
      "defaultValue": 0.0625
    },
    "3": {
      "paramId": "3",
      "paramPhase": "output_dim",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Quantize": {
    "0": {
      "paramId": "0",
      "paramPhase": "scale",
      "defaultValue": 1
    },
    "weightOrder": {
      "scale": [1]
    }
  },
  "Reduction": {
    "0": {
      "paramId": "0",
      "paramPhase": "operation",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "reduce_all",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "coeff",
      "defaultValue": 1.0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "axes",
      "defaultValue": []
    },
    "4": {
      "paramId": "4",
      "paramPhase": "keepdims",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "UNKNOWN",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "ReLU": {
    "0": {
      "paramId": "0",
      "paramPhase": "slope",
      "defaultValue": 0.0
    },
    "weightOrder": {}
  },
  "Reorg": {
    "0": {
      "paramId": "0",
      "paramPhase": "stride",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Requantize": {
    "0": {
      "paramId": "0",
      "paramPhase": "scale_in_data_size",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "scale_out_data_size",
      "defaultValue": 1
    },
    "2": {
      "paramId": "2",
      "paramPhase": "bias_data_size",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "activation_type",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "activation_params",
      "defaultValue": []
    },
    "weightOrder": {
      "scale_in": [1],
      "scale_out": [1],
      "bias": [1]
    }
  },
  "Reshape": {
    "0": {
      "paramId": "0",
      "paramPhase": "w",
      "defaultValue": -233
    },
    "1": {
      "paramId": "1",
      "paramPhase": "h",
      "defaultValue": -233
    },
    "2": {
      "paramId": "2",
      "paramPhase": "c",
      "defaultValue": -233
    },
    "3": {
      "paramId": "3",
      "paramPhase": "permute",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "RNN": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_output",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "weight_data_size",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "direction",
      "defaultValue": 0
    },
    "weightOrder": {
      "weight_xc": [1, 3, 10],
      "bias_c": [1, 3, 10],
      "weight_hc": [1, 3, 10]
    }
  },
  "ROIAlign": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooled_width",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "pooled_height",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "spatial_scale",
      "defaultValue": 1
    },
    "3": {
      "paramId": "3",
      "paramPhase": "sampling_ratio",
      "defaultValue": 0
    },
    "4": {
      "paramId": "4",
      "paramPhase": "aligned",
      "defaultValue": 0
    },
    "5": {
      "paramId": "5",
      "paramPhase": "version",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "ROIPooling": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooled_width",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "pooled_height",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "spatial_scale",
      "defaultValue": 1
    },
    "weightOrder": {}
  },
  "Scale": {
    "0": {
      "paramId": "0",
      "paramPhase": "scale_data_size",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "bias_term",
      "defaultValue": 0
    },
    "weightOrder": {
      "scale": [1],
      "bias": [1]
    }
  },
  "SELU": {
    "0": {
      "paramId": "0",
      "paramPhase": "alpha",
      "defaultValue": 1.67326324
    },
    "1": {
      "paramId": "1",
      "paramPhase": "lambda",
      "defaultValue": 1.050700987
    },
    "weightOrder": {}
  },
  "ShuffleChannel": {
    "0": {
      "paramId": "0",
      "paramPhase": "group",
      "defaultValue": 1
    },
    "1": {
      "paramId": "1",
      "paramPhase": "reverse",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Sigmoid": {
    "weightOrder": {}
  },
  "Slice": {
    "0": {
      "paramId": "0",
      "paramPhase": "slices",
      "defaultValue": []
    },
    "1": {
      "paramId": "1",
      "paramPhase": "axis",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Softmax": {
    "0": {
      "paramId": "0",
      "paramPhase": "axis",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "fixbug0",
      "defaultValue": 1
    },
    "weightOrder": {}
  },
  "Softplus": {
    "weightOrder": {}
  },
  "Split": {
    "weightOrder": {}
  },
  "SPP": {
    "0": {
      "paramId": "0",
      "paramPhase": "pooling_type",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "pyramid_height",
      "defaultValue": 1
    },
    "weightOrder": {}
  },
  "Squeeze": {
    "0": {
      "paramId": "0",
      "paramPhase": "squeeze_w",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "squeeze_h",
      "defaultValue": 0
    },
    "2": {
      "paramId": "2",
      "paramPhase": "squeeze_c",
      "defaultValue": 0
    },
    "3": {
      "paramId": "3",
      "paramPhase": "axes",
      "defaultValue": []
    },
    "weightOrder": {}
  },
  "StatisticsPooling": {
    "0": {
      "paramId": "0",
      "paramPhase": "include_stddev",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "Swish": {
    "weightOrder": {}
  },
  "TanH": {
    "weightOrder": {}
  },
  "Threshold": {
    "0": {
      "paramId": "0",
      "paramPhase": "threshold",
      "defaultValue": 0.0
    },
    "weightOrder": {}
  },
  "Tile": {
    "0": {
      "paramId": "0",
      "paramPhase": "dim",
      "defaultValue": 0
    },
    "1": {
      "paramId": "1",
      "paramPhase": "tiles",
      "defaultValue": 1
    },
    "2": {
      "paramId": "2",
      "paramPhase": "repeats",
      "defaultValue": []
    },
    "weightOrder": {}
  },
  "UnaryOp": {
    "0": {
      "paramId": "0",
      "paramPhase": "op_type",
      "defaultValue": 0
    },
    "weightOrder": {}
  },
  "YoloDetectionOutput": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_class",
      "defaultValue": 20
    },
    "1": {
      "paramId": "1",
      "paramPhase": "num_box",
      "defaultValue": 5
    },
    "2": {
      "paramId": "2",
      "paramPhase": "confidence_threshold",
      "defaultValue": 0.01
    },
    "3": {
      "paramId": "3",
      "paramPhase": "num_threshold",
      "defaultValue": 0.45
    },
    "4": {
      "paramId": "4",
      "paramPhase": "biases",
      "defaultValue": []
    },
    "weightOrder": {}
  },
  "Yolov3DetectionOutput": {
    "0": {
      "paramId": "0",
      "paramPhase": "num_class",
      "defaultValue": 20
    },
    "1": {
      "paramId": "1",
      "paramPhase": "num_box",
      "defaultValue": 5
    },
    "2": {
      "paramId": "2",
      "paramPhase": "confidence_threshold",
      "defaultValue": 0.01
    },
    "3": {
      "paramId": "3",
      "paramPhase": "num_threshold",
      "defaultValue": 0.45
    },
    "4": {
      "paramId": "4",
      "paramPhase": "biases",
      "defaultValue": []
    },
    "5": {
      "paramId": "5",
      "paramPhase": "mask",
      "defaultValue": []
    },
    "6": {
      "paramId": "6",
      "paramPhase": "anchors_scale",
      "defaultValue": []
    },
    "weightOrder": {}
  }
}
"""

param_schema = jload(schemaf)

DTYPE_FP32 = b"\x00\x00\x00\x00"
DTYPE_FP16 = b"\x47\x6b\x30\x01"
DTYPE_DICT = {b"\x00\x00\x00\x00": np.float32, b"\x47\x6b\x30\x01": np.float16}


class UnaryOpTypes:
    ABS = 0
    NEG = 1
    FLOOR = 2
    CEIL = 3
    SQUARE = 4
    SQRT = 5
    RSQ = 6
    EXP = 7
    LOG = 8
    SIN = 9
    COS = 10
    TAN = 11
    ASIN = 12
    ACOS = 13
    ATAN = 14
    RECIPROCAL = 15
    TANH = 16


class BinaryOpTypes:
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MAX = 4
    MIN = 5
    POW = 6
    RSUB = 7
    RDIV = 8


class CastElementTypes:
    AUTO = 0
    FLOAT32 = 1
    FLOAT16 = 2
    INT8 = 3
    BFLOAT16 = 4


class EltwiseOpTypes:
    PROD = 0
    SUM = 1
    MAX = 2


class GruDirectionFlags:
    FORWARD = 0
    REVERSE = 1
    BIDIRECTIONAL = 2


class InterpResizeTypes:
    NEAREST = 1
    BILINEAR = 2
    BICUBIC = 3


class NormalizeEpsModes:
    CAFFE = 0
    PYTORCH = 1
    TENSORFLOW = 2


class PaddingTypes:
    CONSTANT = 0
    REPLICATE = 1
    REFLECT = 2


class PadModes:
    FULL = 0
    VALID = 1
    SAMEUPPER = 2
    SAMELOWER = 3


class PermuteOrderTypes:
    WH_WHC_WHDC = 0
    HW_HWC_HWDC = 1
    WCH_WDHC = 2
    CWH_DWHC = 3
    HCW_HDWC = 4
    CHW_DHWC = 5
    WHCD = 6
    HWCD = 7
    WCHD = 8
    CWHD = 9
    HCWD = 10
    CHWD = 11
    WDCH = 12
    DWCH = 13
    WCDH = 14
    CWDH = 15
    DCWH = 16
    CDWH = 17
    HDCW = 18
    DHCW = 19
    HCDW = 20
    CHDW = 21
    DCHW = 22
    CDHW = 23


class ReductionOpTypes:
    SUM = 0
    ASUM = 1
    SUMSQ = 2
    MEAN = 3
    MAX = 4
    MIN = 5
    PROD = 6
    L1 = 7
    L2 = 8
    LOGSUM = 9
    LOGSUMEXP = 10


class GridSampleSampleTypes:
    NEAREST = 1
    BILINEAR = 2
    BICUBIC = 3


class GridSamplePadModes:
    ZEROS = 1
    BORDER = 2
    REFLECTION = 3


class LrnRegionTypes:
    ACROSS_CHANNELS = 0
    WITH_CHANNEL = 1


class NcnnWeight:
    def __init__(self, weight: np.ndarray, quantize_tag: bytes = b""):
        self.quantize_tag = quantize_tag
        self.weight = weight

    @property
    def shape(self) -> tuple:
        return self.weight.shape


class NcnnParam:
    def __init__(
        self,
        pid: str,
        name: str,
        value: float | int | list[float | int],
        default: float | int,
    ) -> None:
        self.id: str = pid
        self.name: str = name
        self.value: float | int | list[float | int] = value
        self.default: float | int = default


class NcnnParamCollection:
    def __init__(
        self,
        op: str,
        param_dict: dict[int, NcnnParam] | None = None,
    ) -> None:
        self.op: str = op
        self.param_dict: dict[int, NcnnParam] = {} if param_dict is None else param_dict
        self.weight_order: dict[str, list[int]] = (
            param_schema[self.op]["weightOrder"] if self.op else {}
        )

    def __getitem__(self, pid: int) -> NcnnParam:
        try:
            return self.param_dict[pid]
        except KeyError as exc:
            idstr = str(pid)
            param_dict = param_schema[self.op]
            try:
                param = param_dict[idstr]
            except KeyError:
                raise

            default_value = param["defaultValue"]
            value = param["defaultValue"]
            if isinstance(value, str):
                for key, val in list(param_dict.items())[:-1]:
                    if value == val["paramPhase"]:
                        try:
                            value = self.param_dict[int(key)].value
                        except KeyError:
                            value = val["defaultValue"]
                        default_value = val["defaultValue"]

                        break
                else:
                    msg = f"Op {self.op} does not have param {value}, please report"
                    raise KeyError(msg) from exc

            return NcnnParam(idstr, param["paramPhase"], value, default_value)

    def __setitem__(self, pid: int, value: float | int | list[float | int]) -> None:
        idstr = str(pid)
        param_dict = param_schema[self.op]
        try:
            param = param_dict[idstr]
        except KeyError:
            raise
        name = param["paramPhase"]
        def_val = param["defaultValue"]

        self.param_dict[pid] = NcnnParam(idstr, name, value, def_val)

    def __delitem__(self, key: int) -> None:
        try:
            del self.param_dict[key]
        except KeyError:
            pass

    def __contains__(self, item: int) -> bool:
        if item in self.param_dict:
            return True
        return False

    def __str__(self) -> str:
        output = ""
        param_dict = param_schema[self.op]
        self.param_dict = dict(sorted(self.param_dict.items()))
        for v in self.param_dict.values():
            if v.value == v.default:
                continue
            if isinstance(v.default, str) and "FLT_MAX" not in v.default:
                pid = None
                for key, val in list(param_dict.items())[:-1]:
                    if v.default == val["paramPhase"]:
                        pid = int(key)
                        break
                else:
                    msg = f"Op {self.op} does not have param {v.default}, please report"
                    raise KeyError(msg)

                # If a param that defaults to the value of another param, if it's value
                # equals that of the second param or its default, skip writing it
                if v.value in (
                    self.param_dict[pid].value,
                    self.param_dict[pid].default,
                ):
                    continue

            if isinstance(v.value, list):
                output += " -233" + v.id.zfill(2) + "="
            else:
                output += " " + v.id + "="

            if isinstance(v.value, float):
                v_str = np.format_float_scientific(v.value, 6, False, exp_digits=2)
            elif isinstance(v.value, list):
                v_str = ",".join(
                    [
                        np.format_float_scientific(n, 6, False, exp_digits=2)
                        if isinstance(n, float)
                        else str(n)
                        for n in v.value
                    ]
                )
            else:
                v_str = str(v.value)

            output += v_str

        return output

    def set_op(self, op: str) -> None:
        self.op = op
        self.weight_order = param_schema[op]["weightOrder"]


class NcnnLayer:
    def __init__(
        self,
        op_type: str = "",
        name: str = "",
        num_inputs: int = 0,
        num_outputs: int = 0,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        params: NcnnParamCollection | None = None,
        weight_data: dict[str, NcnnWeight] | None = None,
    ):
        self.op_type: str = op_type
        self.name: str = name
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs
        self.inputs: list[str] = [] if inputs is None else inputs
        self.outputs: list[str] = [] if outputs is None else outputs
        self.params: NcnnParamCollection = (
            NcnnParamCollection(op_type) if params is None else params
        )
        self.weight_data: dict[str, NcnnWeight] = (
            {} if weight_data is None else weight_data
        )

    def add_param(self, pid: int, value: float | int | list[float | int]) -> None:
        self.params[pid] = value

    def add_weight(
        self,
        weight_name: str,
        data: float | int | np.ndarray,
        quantize_tag: bytes = b"",
    ) -> int:
        if isinstance(data, float):
            data_array = np.array(data, np.float32)
        elif isinstance(data, int):
            data_array = np.array(data, np.int32)
        else:
            data_array = data

        if quantize_tag == DTYPE_FP16:
            data_array = data_array.astype(np.float16)
        else:
            # Since int8 not supported, all data that is not fp16 is fp32.
            # This covers issues caused by converting fp16 ONNX models.
            data_array = data_array.astype(np.float32)
        self.weight_data[weight_name] = NcnnWeight(data_array, quantize_tag)

        return len(quantize_tag) + len(data_array.tobytes())


class NcnnModel:
    def __init__(
        self,
        node_count: int = 0,
        blob_count: int = 0,
    ) -> None:
        self.node_count: int = node_count
        self.blob_count: int = blob_count
        self.layers: list[NcnnLayer] = []
        self.bin_length = 0

    @property
    def magic(self):
        return "7767517"

    def load_from_file(param_path: str = "", bin_path: str = "") -> NcnnModel:
        if bin_path == "":
            bin_path = param_path.replace(".param", ".bin")
        elif param_path == "":
            param_path = bin_path.replace(".bin", ".param")

        model = NcnnModel()
        with open(param_path, encoding="utf-8") as paramf:
            with open(bin_path, "rb") as binf:
                paramf.readline()
                counts = paramf.readline().strip().split(" ")
                model.node_count = int(counts[0])
                model.blob_count = int(counts[1])

                for line in paramf:
                    op_type, layer = model.parse_param_layer(line)
                    model.add_layer(layer)

                binf.seek(0, os.SEEK_END)
                model.bin_length = binf.tell()

        return model

    def interp_layers(
        a: NcnnLayer, b: NcnnLayer, alpha_a: float
    ) -> tuple[NcnnLayer, bytes]:
        weights_a = a.weight_data
        weights_b = b.weight_data
        weights_interp: dict[str, NcnnWeight] = {}
        layer_bytes = b""

        if weights_a:
            assert len(weights_a) == len(
                weights_b
            ), "All corresponding nodes must have same number of weights"

            layer_bytes_list = []
            for weight_name, weight_a in weights_a.items():
                try:
                    weight_b = weights_b[weight_name]
                except KeyError:
                    raise

                assert (
                    weight_a.shape == weight_b.shape
                ), "Corresponding weights must have the same size and shape"

                assert len(weight_a.quantize_tag) == len(
                    weight_b.quantize_tag
                ), "Weights must either both have or both not have a quantize tag"

                if (
                    weight_a.quantize_tag == DTYPE_FP16
                    and weight_b.quantize_tag == DTYPE_FP32
                ):
                    weight_b.quantize_tag = DTYPE_FP16
                    weight_b.weight = weight_b.weight.astype(np.float16)
                elif (
                    weight_a.quantize_tag == DTYPE_FP32
                    and weight_b.quantize_tag == DTYPE_FP16
                ):
                    weight_a.quantize_tag = DTYPE_FP16
                    weight_a.weight = weight_a.weight.astype(np.float16)

                weight_c = NcnnWeight(
                    (weight_a.weight * alpha_a + weight_b.weight * (1 - alpha_a)),
                    weight_a.quantize_tag,
                )
                layer_bytes_list.append(
                    weight_c.quantize_tag + weight_c.weight.tobytes()
                )

                weights_interp[weight_name] = weight_c

            layer_bytes = b"".join(layer_bytes_list)

        return (
            NcnnLayer(
                a.op_type,
                a.name,
                a.num_inputs,
                a.num_outputs,
                a.inputs,
                a.outputs,
                a.params,
                weights_interp,
            ),
            layer_bytes,
        )

    def add_layer(self, layer: NcnnLayer) -> None:
        self.layers.append(layer)

    def parse_param_layer(self, layer_str: str) -> tuple[str, NcnnLayer]:
        param_list = layer_str.strip().split()
        op_type, name = param_list[:2]

        num_inputs = int(param_list[2])
        num_outputs = int(param_list[3])
        input_end = 4 + num_inputs
        output_end = input_end + num_outputs
        inputs = list(param_list[4:input_end])
        outputs = list(param_list[input_end:output_end])

        params = param_list[output_end:]
        param_dict = {}
        for param_str in params:
            ks, vs = param_str.split("=")
            k = int(ks)
            if k < 0:
                v = []
                for vi in vs.split(","):
                    vi = float(vi) if "." in vi or "e" in vi else int(vi)  # noqa: PLW2901
                    v.append(vi)
                k = abs(k + 23300)
                ks = str(k)
            elif "." in vs or "e" in vs:
                v = float(vs)
            else:
                v = int(vs)

            param = NcnnParam(
                ks,
                param_schema[op_type][ks]["paramPhase"],
                v,
                param_schema[op_type][ks]["defaultValue"],
            )
            param_dict[k] = param

        return op_type, NcnnLayer(
            op_type,
            name,
            num_inputs,
            num_outputs,
            inputs,
            outputs,
            NcnnParamCollection(op_type, param_dict),
        )

    def load_layer_weights(
        self, binf: BufferedReader, op_type: str, layer: NcnnLayer
    ) -> dict[str, NcnnWeight]:
        weight_dict = {}
        if op_type == "BatchNorm":
            channels_data = checked_cast(int, layer.params[0].value) * 4
            slope = np.frombuffer(binf.read(channels_data), np.float32)
            weight_dict["slope"] = NcnnWeight(slope)
            mean = np.frombuffer(binf.read(channels_data), np.float32)
            weight_dict["mean"] = NcnnWeight(mean)
            variance = np.frombuffer(binf.read(channels_data), np.float32)
            weight_dict["variance"] = NcnnWeight(variance)
            bias = np.frombuffer(binf.read(channels_data), np.float32)
            weight_dict["bias"] = NcnnWeight(bias)
        elif op_type in ("Convolution", "ConvolutionDepthWise"):
            quantize_tag = binf.read(4)
            dtype = DTYPE_DICT[quantize_tag]
            weight_data_length = checked_cast(int, layer.params[6].value)
            weight_data_size = (
                weight_data_length * 2
                if quantize_tag == DTYPE_FP16
                else weight_data_length * 4
            )

            has_bias = layer.params[5].value

            num_filters = checked_cast(int, layer.params[0].value)
            kernel_w = checked_cast(int, layer.params[1].value)
            kernel_h = checked_cast(int, layer.params[11].value)
            if op_type == "ConvolutionDepthWise":
                group = checked_cast(int, layer.params[7].value)
                num_input = (
                    weight_data_length // (num_filters // group) // kernel_w // kernel_h
                )
                shape = (
                    group,
                    num_filters // group,
                    num_input // group,
                    kernel_h,
                    kernel_w,
                )
            else:
                num_input = weight_data_length // num_filters // kernel_w // kernel_h
                shape = (num_filters, num_input, kernel_h, kernel_w)

            weight_data = np.frombuffer(binf.read(weight_data_size), dtype)
            weight_data = weight_data.reshape(shape)
            weight_dict["weight"] = NcnnWeight(weight_data, quantize_tag)

            if has_bias:
                bias_data_size = num_filters * 4
                bias_data = np.frombuffer(binf.read(bias_data_size), np.float32)
                weight_dict["bias"] = NcnnWeight(bias_data)
        elif op_type == "Deconvolution":
            quantize_tag = binf.read(4)
            dtype = DTYPE_DICT[quantize_tag]
            weight_data_length = checked_cast(int, layer.params[6].value)
            weight_data_size = (
                weight_data_length * 2
                if quantize_tag == DTYPE_FP16
                else weight_data_length * 4
            )

            has_bias = layer.params[5].value

            num_filters = checked_cast(int, layer.params[0].value)
            kernel_w = checked_cast(int, layer.params[1].value)
            kernel_h = checked_cast(int, layer.params[11].value)
            num_input = weight_data_length // num_filters // kernel_w // kernel_h
            shape = (num_filters, num_input, kernel_h, kernel_w)

            weight_data = np.frombuffer(binf.read(weight_data_size), dtype)
            weight_data = weight_data.reshape(shape)
            weight_dict["weight"] = NcnnWeight(weight_data, quantize_tag)

            if has_bias:
                bias_data_size = num_filters * 4
                bias_data = np.frombuffer(binf.read(bias_data_size), np.float32)
                weight_dict["bias"] = NcnnWeight(bias_data)
        elif op_type == "InnerProduct":
            quantize_tag = binf.read(4)
            dtype = DTYPE_DICT[quantize_tag]
            weight_data_length = layer.params[2].value
            assert isinstance(weight_data_length, int), "Weight data size must be int"
            weight_data_size = (
                weight_data_length * 2
                if quantize_tag == DTYPE_FP16
                else weight_data_length * 4
            )
            weight_data = np.frombuffer(binf.read(weight_data_size), dtype)
            num_output = layer.params[0].value
            assert isinstance(num_output, int), "Num output must be int"
            num_input = weight_data_length // num_output
            weight_data = weight_data.reshape((num_input, num_output))
            weight_dict["weight"] = NcnnWeight(weight_data, quantize_tag)

            has_bias = layer.params[1].value
            if has_bias == 1:
                bias_data_size = num_output * 4
                bias_data = np.frombuffer(binf.read(bias_data_size), np.float32)
                weight_dict["bias"] = NcnnWeight(bias_data)
        elif op_type == "PReLU":
            num_slope = layer.params[0].value
            assert isinstance(num_slope, int), "Num slopes must be int"
            slope_data_size = num_slope * 4
            slope_data = np.frombuffer(binf.read(slope_data_size), np.float32)
            weight_dict["slope"] = NcnnWeight(slope_data)
        elif op_type == "Scale":
            scale_data_length = layer.params[0].value
            assert isinstance(scale_data_length, int), "Scale data size must be int"
            if scale_data_length != -233:
                quantize_tag = binf.read(4)
                dtype = DTYPE_DICT[quantize_tag]
                scale_data_size = (
                    scale_data_length * 2
                    if quantize_tag == DTYPE_FP16
                    else scale_data_length * 4
                )
                scale_data = np.frombuffer(binf.read(scale_data_size), dtype)
                weight_dict["weight"] = NcnnWeight(scale_data, quantize_tag)

                has_bias = layer.params[1].value
                if has_bias == 1:
                    bias_data = np.frombuffer(
                        binf.read(scale_data_length * 4), np.float32
                    )
                    weight_dict["bias"] = NcnnWeight(bias_data)

        elif len(layer.params.weight_order) != 0:
            error_msg = f"Load weights not added for {op_type} yet, please report"
            raise ValueError(error_msg)

        return weight_dict

    def write_param(self, filename: Path | str = "") -> str:
        with StringIO() as p:
            p.write(f"{self.magic}\n{self.node_count} {self.blob_count}\n")

            for layer in self.layers:
                if layer.op_type == "ncnnfused":
                    continue

                p.write(
                    f"{layer.op_type:<16}"
                    f" {layer.name:<24}"
                    f" {layer.num_inputs}"
                    f" {layer.num_outputs}"
                )
                if layer.inputs:
                    p.write(f" {' '.join(layer.inputs)}")
                if layer.outputs:
                    p.write(f" {' '.join(layer.outputs)}")
                if layer.params.param_dict:
                    param_str = str(layer.params)
                    if param_str:
                        p.write(f"{param_str}")
                p.write("\n")

            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(p.getvalue())
                return ""
            else:
                return p.getvalue()

    def serialize_weights(self) -> bytes:
        layer_weights = [
            b"".join((w.quantize_tag, np.ndarray.tobytes(w.weight)))
            for l in self.layers
            for w in l.weight_data.values()
            if l.weight_data and l.op_type != "ncnnfused"
        ]

        return b"".join(layer_weights)

    def write_bin(self, filename: Path | str) -> None:
        with open(filename, "wb") as f:
            f.write(self.serialize_weights())

    def interpolate(self, model_b: NcnnModel, alpha: float) -> NcnnModel:
        interp_model = deepcopy(self)

        layer_a_weights = [(i, l) for i, l in enumerate(self.layers) if l.weight_data]
        layer_b_weights = [
            (i, l) for i, l in enumerate(model_b.layers) if l.weight_data
        ]

        assert len(layer_a_weights) == len(
            layer_b_weights
        ), "Models must have same number of layers containing weights"

        weight_bytes_list = []
        for layer_a, layer_b in zip(layer_a_weights, layer_b_weights):
            interp_layer, layer_bytes = NcnnModel.interp_layers(
                layer_a[1], layer_b[1], alpha
            )
            interp_model.layers[layer_a[0]] = interp_layer
            weight_bytes_list.append(layer_bytes)

        return interp_model

    @property
    def bin(self) -> bytes:
        return self.serialize_weights()


class NcnnModelWrapper:
    def __init__(self, model: NcnnModel) -> None:
        self.model: NcnnModel = model
        scale, in_nc, out_nc, nf, fp = NcnnModelWrapper.get_broadcast_data(model)
        self.scale: int = scale
        self.nf: int = nf
        self.in_nc: int = in_nc
        self.out_nc: int = out_nc
        self.fp: str = fp

    def get_nf_and_in_nc(layer: NcnnLayer) -> tuple[int, int]:
        nf = layer.params[0].value
        kernel_w = layer.params[1].value
        try:
            kernel_h = layer.params[11].value
        except KeyError:
            kernel_h = kernel_w
        weight_data_size = layer.params[6].value

        assert (
            isinstance(nf, int)
            and isinstance(kernel_w, int)
            and isinstance(kernel_h, int)
            and isinstance(weight_data_size, int)
        ), "Out nc, kernel width and height, and weight data size must all be ints"
        in_nc = weight_data_size // nf // kernel_w // kernel_h

        return nf, in_nc


def get_broadcast_data(model: NcnnModel) -> tuple[int, int, int, int, str]:
    scale = 1.0
    in_nc = 0
    out_nc = 0
    nf = 0
    fp = "fp32"
    pixel_shuffle = 1
    found_first_conv = False
    current_conv = None

    for i, layer in enumerate(model.layers):
        if layer.op_type == "Interp":
            try:
                if (
                    model.layers[i + 1].op_type != "BinaryOp"
                    and model.layers[i + 1].params[0].value != 0
                ):
                    scale *= checked_cast(float, layer.params[1].value)
            except IndexError:
                scale *= checked_cast(float, layer.params[1].value)
        elif layer.op_type == "PixelShuffle":
            scale *= checked_cast(int, layer.params[0].value)
            pixel_shuffle *= checked_cast(int, layer.params[0].value)
        elif layer.op_type in (
            "Convolution",
            "Convolution1D",
            "ConvolutionDepthWise",
        ):
            if found_first_conv is not True:
                nf, in_nc = NcnnModelWrapper.get_nf_and_in_nc(layer)

                found_first_conv = True

            scale /= checked_cast(int, layer.params[3].value)
            current_conv = layer
        elif layer.op_type in ("Deconvolution", "DeconvolutionDepthWise"):
            if found_first_conv is not True:
                nf, in_nc = NcnnModelWrapper.get_nf_and_in_nc(layer)
                found_first_conv = True

            scale *= checked_cast(int, layer.params[3].value)
            current_conv = layer

    assert current_conv is not None, "Cannot broadcast; model has no Convolution layers"

    out_nc = checked_cast(int, current_conv.params[0].value) // pixel_shuffle**2

    assert scale >= 1, "Models with scale less than 1x not supported"
    assert scale % 1 == 0, f"Model not supported, scale {scale} is not an integer"

    return int(scale), in_nc, out_nc, nf, fp


if __name__ == "__main__":
    try:
        model_param = sys.argv[1]

    except:
        model_param = input("Enter param path here: ")

    model = NcnnModel.load_from_file(model_param)
    scale = get_broadcast_data(model)[0]
    print(scale)


def getNCNNScale(modelPath: str = "") -> int:
    model = NcnnModel.load_from_file(modelPath + ".param")
    scale = get_broadcast_data(model)[0]
    return scale


class UpscaleNCNN:
    def __init__(
        self,
        modelPath: str,
        num_threads: int,
        scale: int,
        gpuid: int = 0,
        width: int = 1920,
        height: int = 1080,
    ):
        # only import if necessary
        

        self.model = UPSCALE(
            gpuid=gpuid, model_str=modelPath, num_threads=num_threads, scale=scale
        )
        self.width = width
        self.height = height
        self.scale = scale
        """model = model + '.param'
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.load_param(model.replace('.bin','.param'))
        self.net.load_model(model.replace('.param','.bin'))
        """

    """def NCNNImageMatFromNP(self, npArray: np.array) -> ncnn.Mat:
        return ncnn.Mat.from_pixels(
            npArray,
            ncnn.Mat.PixelType.PIXEL_BGR,
            self.width,
            self.height,
        )

    def NormalizeImage(self, mat, norm_vals):
        mean_vals = []
        mat.substract_mean_normalize(mean_vals, norm_vals)

    def ClampNPArray(self, nparray: np.array) -> np.array:
        min_val = np.min(nparray)
        max_val = np.max(nparray)
        if min_val < 0 or max_val > 255:
            nparray = ((nparray - min_val) / (max_val - min_val)) * 255
        return nparray

    def ProcessNCNN(self, frame: np.array) -> np.asarray:
        ex = self.net.create_extractor()
        frame = self.NCNNImageMatFromNP(frame)
        # norm
        self.NormalizeImage(mat=frame, norm_vals=[1 / 255.0, 1 / 255.0, 1 / 255.0])
        # render frame
        ex.input("data", frame)
        ret, frame = ex.extract("output")

        # norm
        self.NormalizeImage(mat=frame, norm_vals=[255.0, 255.0, 255.0])

        frame = np.ascontiguousarray(frame)
        frame = self.ClampNPArray(frame)
        frame = frame.transpose(1, 2, 0)
        return np.ascontiguousarray(frame, dtype=np.uint8)"""

    def Upscale(self, imageChunk):
        output = self.model.process_bytes(imageChunk, self.width, self.height, 3)
        return np.ascontiguousarray(
            np.frombuffer(
                output,
                dtype=np.uint8,
            ).reshape(self.height * self.scale, self.width * self.scale, 3)
        )
