import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet_HDv3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, local_rank=-1,scale=1.0,ensemble=False):
        self.flownet = IFNet(scale,ensemble)
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.version = 4.8
        # self.vgg = VGGPerceptualLoss().to(device)
        if local_rank != -1:
            self.flownet = DDP(
                self.flownet, device_ids=[local_rank], output_device=local_rank
            )

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def half(self):
        self.flownet.half()

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(
                    convert(torch.load("{}/rife4.16-lite.pkl".format(path))), False
                )
            else:
                self.flownet.load_state_dict(
                    convert(
                        torch.load(
                            "{}/rife4.16-lite.pkl".format(path), map_location="cpu"
                        )
                    ),
                    False,
                )

    def inference(self, img0, img1, timestep=0.5):

        return self.flownet(img0, img1, timestep)
