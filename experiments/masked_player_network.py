"""Small masked player-set encoder with shared antisymmetric temporal scoring."""
import numpy as np
import torch
from torch import nn


def initialize(seed=42):
    if torch.__version__!='2.5.0':raise ValueError('Pinned existing Torch2.5.0 required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(seed)


class MaskedPlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.null=nn.Parameter(torch.zeros(1,1,32))
        self.spatial=nn.TransformerEncoderLayer(32,4,dim_feedforward=64,dropout=.1,batch_first=True)
        self.project=nn.Conv1d(66,32,1)
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(32,32,3,padding=d,dilation=d),nn.ReLU(),nn.Dropout(.1)) for d in (1,2,4,8)])
        self.output=nn.Linear(32,1)

    def forward(self,xy,observed,teams):
        batch,time,nodes,_=xy.shape
        observed=observed.bool()&torch.isfinite(xy).all(dim=-1)
        xy=torch.where(observed.unsqueeze(-1),torch.nan_to_num(xy),0.)
        tokens=self.stem(xy).reshape(batch*time,nodes,32)
        tokens=torch.cat([tokens,self.null.expand(batch*time,-1,-1)],dim=1)
        padding=torch.cat([~observed.reshape(batch*time,nodes),torch.zeros(batch*time,1,dtype=torch.bool,device=xy.device)],dim=1)
        encoded=self.spatial(tokens,src_key_padding_mask=padding)[:,:nodes].reshape(batch,time,nodes,32)
        pooled=[]
        for team in (0,1):
            keep=observed&(teams==team);count=keep.sum(dim=-1,keepdim=True)
            mean=(encoded*keep.unsqueeze(-1)).sum(dim=2)/count.clamp(min=1)
            pooled.append(torch.cat([mean,count.to(xy.dtype)/11],dim=-1))
        home=torch.cat([pooled[0],pooled[1]],dim=-1);away=torch.cat([pooled[1],pooled[0]],dim=-1)
        value=torch.relu(self.project(torch.cat([home,away],dim=0).transpose(1,2)))
        for block in self.temporal:value=value+block(value)
        score=self.output(value[:,:,time//2])[:,0]
        return score[:batch]-score[batch:]


def step(model,optimizer,xy,observed,teams,target):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xy,observed,teams),target)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    return float(loss.detach())


def probability(model,xy,observed,teams):
    model.eval()
    with torch.inference_mode():return torch.sigmoid(model(xy,observed,teams)).numpy()
