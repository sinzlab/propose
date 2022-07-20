import torch
import torch.backends.cudnn as cudnn

from collections import OrderedDict

import os

from .models.pose_hrnet import PoseHighResolutionNet
from .config import config

import wandb


class HRNet(PoseHighResolutionNet):
    @classmethod
    def from_pretrained(cls, artifact_name=None, config_file=None, **kwargs):
        if not config_file:
            dirname = os.path.dirname(__file__)
            config_file = os.path.join(
                dirname, "experiments/w32_256x256_adam_lr1e-3.yaml"
            )

            config.defrost()
            config.merge_from_file(config_file)
            config.freeze()

        model = cls(config, **kwargs)

        api = wandb.Api()
        artifact = api.artifact(artifact_name, type="model")

        if wandb.run:
            wandb.run.use_artifact(artifact, type="model")

        artifact_dir = artifact.download()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(
            artifact_dir + "/pose_hrnet_w32_256x256.pth",
            map_location=torch.device(device),
        )

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k  # remove module.
            #  print(name,'\t')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    @property
    def device(self):
        return next(self.parameters()).device
