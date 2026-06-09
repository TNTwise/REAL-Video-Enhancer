from dataclasses import dataclass
from typing import Literal


@dataclass
class GRLConfig:
    out_proj_type: Literal["linear", "conv2d"] = "linear"
    """
    Type of the output projection in the self-attention modules.
    """
    local_connection: bool = False
    """
    Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.
    """
    euclidean_dist: bool = False
    """
    use Euclidean distance or inner product as the similarity metric. An ablation study.
    """
    double_window: bool = False
    stripe_square: bool = False
    separable_conv_act: bool = False
    use_buffer: bool = False
    """
    Whether to use buffer.
    False: the attention masks, tables, and indices are pre-computed. Huge GPU memory consumption when the window size is large.
    True:
        use_efficient_buffer=False: buffers are not shared. computed for each layer during forward pass. Slow forward pass.
        use_efficient_buffer=True: pre-computed and shared buffers. Small GPU memory consumption, fast forward pass. Need to allocate buffers manually.
    """
    use_efficient_buffer: bool = False
    """
    Whether to use efficient buffer.
    """
