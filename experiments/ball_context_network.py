"""R39 bounded player attention with explicitly observed relative ball geometry."""
import torch
from torch import nn
from masked_player_network import MaskedPlayer


class BallContext(MaskedPlayer):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Linear(5,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())

    def forward(self,xy,observed,teams,ball,ball_observed):
        good=ball_observed.bool()&torch.isfinite(ball).all(-1)
        clean=torch.where(good.unsqueeze(-1),torch.nan_to_num(ball),0.)
        relative=torch.where(good[:,:,None,None],xy-clean[:,:,None],0.)
        flag=good[:,:,None,None].expand(*xy.shape[:-1],1).to(xy.dtype)
        value=torch.cat([xy,relative,flag],dim=-1)
        return super().forward(value,observed,teams)


def probability(model,values):
    model.eval()
    with torch.inference_mode():return torch.sigmoid(model(*values)).numpy()


def step(model,optimizer,values,target):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(*values),target)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();return float(loss.detach())
