import torch
from torch.nn import Module
import models.diffusion as diffusion
from models.diffusion import VarianceSchedule, D2MP_OB
import numpy as np

class D2MP(Module):
    def __init__(self, config, encoder=None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = encoder
        self.diffnet = getattr(diffusion, config.diffnet)

        self.diffusion = D2MP_OB(
            # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            net=self.diffnet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            var_sched = VarianceSchedule(
                num_steps=100,
                beta_T=5e-2,
                mode='linear'
            ),
            config=self.config
        )

    def generate(self, conds, sample, bestof, flexibility=0.0, ret_traj=False, img_w=None, img_h=None):
        cond_encodeds = []
        for i in range(len(conds)):
            tmp_c = conds[i]
            tmp_c = np.array(tmp_c)
            tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
            tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
            tmp_conds = torch.tensor(tmp_c, dtype=torch.float)
            if len(tmp_conds) != 5:
                pad_conds = tmp_conds[-1].repeat((5, 1))
                tmp_conds = torch.cat((tmp_conds, pad_conds), dim=0)[:5]
            cond_encodeds.append(tmp_conds.unsqueeze(0))
        cond_encodeds = torch.cat(cond_encodeds)
        cond_encodeds = self.encoder(cond_encodeds)
        track_pred = self.diffusion.sample(cond_encodeds, sample, bestof, flexibility=flexibility, ret_traj=ret_traj)
        return track_pred.cpu().detach().numpy()

    def forward(self, batch):
        cond_encoded = self.encoder(batch["condition"]) # B * 64
        loss = self.diffusion(batch["delta_bbox"], cond_encoded)
        return loss