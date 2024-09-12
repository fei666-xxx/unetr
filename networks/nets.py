from typing import Tuple, Union, Sequence
import math
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
import numpy as np
import torch
import torch.nn as nn
import  torch.nn.functional as F

from visualizer import get_local

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.utils import optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")


class SABlock_SPA(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, has_cls=False, gamma=1.0)-> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.has_cls = has_cls
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)


    def forward(self, x):
        B, N, C = x.shape
        
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        
        _, H, _, _ = q.shape
        
        
        ###############
        #基于 masked 的 same pos attention
        ###############
        
        # # torch.full
        # mask = torch.zeros([B,H,N,N],dtype=torch.float,requires_grad=False, device=q.device)
        # NUM = N//4
        # if not self.has_cls:
        #     for kk in range(N):
        #         idx = kk % NUM
        #         if idx>0 :
        #             mask[:,:,kk,0:idx] = float("-inf")
        #         mask[:,:,kk,idx+1:idx+NUM] = float("-inf")
        #         mask[:,:,kk,idx+NUM+1:idx+2*NUM] = float("-inf")
        #         mask[:,:,kk,idx+2*NUM+1:idx+3*NUM] = float("-inf")
        #         if idx+3*NUM+1<N:
        #             mask[:,:,kk,idx+3*NUM+1:N] = float("-inf")
        # else:
        #     for kk in range(1,N):
        #         idx = (kk-1) % NUM 
        #         mask[:,:,kk,0:idx+1] = float("-inf")
        #         mask[:,:,kk,idx+2:idx+NUM+1] = float("-inf")
        #         mask[:,:,kk,idx+NUM+2:idx+2*NUM+1] = float("-inf")
        #         mask[:,:,kk,idx+2*NUM+2:idx+3*NUM+1] = float("-inf")    
        #         if idx+3*NUM+2<N:
        #             mask[:,:,kk,idx+3*NUM+2:N] = float("-inf")
        # att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale + mask).softmax(dim=-1)
        
        
        ###############
        #基于 超参数γ 的 same pos attention
        ###############
        
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        NUM = N//4
        mask = torch.ones([B,H,N,N],dtype=torch.float,requires_grad=False, device=q.device)
        
        if not self.has_cls:
            for kk in range(N):
                idx = kk % NUM
                mask[:,:,kk,idx] = self.gamma
                mask[:,:,kk,idx+NUM] = self.gamma
                mask[:,:,kk,idx+2*NUM] = self.gamma
                mask[:,:,kk,idx+3*NUM] = self.gamma
        else:
            for kk in range(1,N):
                idx = (kk-1) % NUM
                mask[:,:,kk,idx+1] = self.gamma
                mask[:,:,kk,idx+NUM+1] = self.gamma
                mask[:,:,kk,idx+2*NUM+1] = self.gamma
                mask[:,:,kk,idx+3*NUM+1] = self.gamma
        att_mat = att_mat.mul(mask).softmax(dim=-1)
        
        
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)

        return x

 

class SPABlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, has_cls=False, gamma=1.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """
        super().__init__()   

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock_SPA(hidden_size, num_heads, dropout_rate, has_cls=has_cls, gamma=gamma)
        self.norm2 = nn.LayerNorm(hidden_size)
        

    def forward(self, x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x




class SABlock_MCA(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, enhance_rate=0.0, fuse_rate=0.0, has_cls=False, CSA=False)-> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        
        self.CSA = CSA
        self.has_cls = has_cls
        
        self.enhance_rate = enhance_rate
        assert 0 <= enhance_rate <= 1, "enhance_rate must > 0 and <= 1, got {0}".format(enhance_rate)
        self.fuse_rate = fuse_rate
        assert 0 <= fuse_rate <= 1, "fuse_rate must > 0 and <= 1, got {0}".format(fuse_rate)

    @get_local('att_mat')
    def forward(self, x):
        B, N, C = x.shape
        
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        
        _, H, _, _ = q.shape
        
        if self.CSA:
            mask = torch.zeros([B,H,N,N],dtype=torch.float,requires_grad=False, device=q.device)
            NUM = N//4
            
            if self.has_cls:
                for kk in [0,1,2,3]:
                    mask[:,:,kk*NUM+1:(kk+1)*NUM+1,kk*NUM+1:(kk+1)*NUM+1] = float("-inf")
            else:
                for kk in [0,1,2,3]:
                    mask[:,:,kk*NUM:(kk+1)*NUM,kk*NUM:(kk+1)*NUM] = float("-inf")
            att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale + mask).softmax(dim=-1)
            
        else:
            att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        
        fuse_tokens = 0
        enhance_tokens = 0
        if self.fuse_rate > 0 or self.enhance_rate > 0 :
            cls_attn = att_mat[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            if self.fuse_rate > 0 :
                fuse_tokens = math.ceil(self.fuse_rate * (N - 1))
                _, fuse_idx = torch.topk(cls_attn, fuse_tokens, dim=1, largest=False, sorted=True)  # [B, fuse_tokens]
                fuse_index = fuse_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, fuse_tokens, C]
            else:
                fuse_index = None
                fuse_idx = None
                
            if self.enhance_rate > 0 :
                enhance_tokens = math.ceil(self.enhance_rate * (N - 1))
                _, enhance_idx = torch.topk(cls_attn, enhance_tokens, dim=1, largest=True, sorted=True)  # [B, enhance_tokens]
                enhance_index = enhance_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, enhance_tokens, C]
            else:
                enhance_index = None
                enhance_idx = None

            return x, enhance_index, enhance_idx, fuse_index, fuse_idx, cls_attn

        return  x, None, None, None, None, None


 

class MCABlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, enhance_rate=0.0, fuse_rate=0.0, has_cls=False, CSA=False) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """
        super().__init__()   

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock_MCA(hidden_size, num_heads, dropout_rate, enhance_rate=enhance_rate, fuse_rate=fuse_rate, has_cls=has_cls, CSA=CSA)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.enhance_rate = enhance_rate
        if(enhance_rate>0):
            self.enhance = nn.Sequential(nn.Linear(hidden_size, hidden_size*4), nn.GELU())
        self.fuse_rate = fuse_rate

    def forward(self, x):
        B, N, C = x.shape
        
        tmp, enhance_index, enhance_idx, fuse_index, fuse_idx, cls_attn = self.attn(self.norm1(x))
        x = x + tmp
        
        if (enhance_index is not None) or (fuse_index is not None) :
            non_cls = x[:, 1:]
            
            if fuse_idx is not None:
                others_index = complement_idx(fuse_idx, N - 1).unsqueeze(-1).expand(-1, -1, C)
                x_cls = x[:, 0, :].unsqueeze(1)
                x_others = torch.gather(non_cls, dim=1, index=others_index)  # [B, left_tokens, C]
            else:
                x_cls = x[:, 0, :].unsqueeze(1)
                x_others = x[:, 1:, :]
            
            

            if enhance_index is not None:
                
                # x_id = torch.div(enhance_idx, 64, rounding_mode='trunc').int()
                # y_id = torch.div(enhance_idx-x_id*64, 8, rounding_mode='trunc').int()
                # z_id = enhance_idx - x_id*64 - y_id*8
                
                # enhance_idx_new = []
                
                # for X_id in [x_id*2, x_id*2+1]:
                #     for Y_id in [y_id*2, y_id*2+1]:
                #         for Z_id in [z_id*2, z_id*2+1]:
                #             enhance_idx_new.append(X_id*256 + Y_id*16 + Z_id)
                # enhance_idx_new = torch.cat(enhance_idx_new, dim=1)
                
                # enhance_topk = torch.gather(x_p8, dim=1, index=enhance_idx_new.unsqueeze(-1).expand(-1, -1, C))  # [B, enhance_tokens*8, C]
                # x_cls = torch.cat([x_cls, enhance_topk], dim=1)
                
                enhance_topk = torch.gather(non_cls, dim=1, index=enhance_index)
                enhance_topk = self.enhance(enhance_topk)
                enhance_topk = torch.cat(torch.chunk(enhance_topk, 4, dim=-1), dim=1)
                x_cls = torch.cat([x_cls, enhance_topk], dim=1)
                
                
            if fuse_index is not None:
                fuse_topk = torch.gather(non_cls, dim=1, index=fuse_index)  # [B, fuse_tokens, C]
                fuse_topk_attn = torch.gather(cls_attn, dim=1, index=fuse_idx)  # [B, fuse_tokens]
                extra_token = torch.sum(fuse_topk * fuse_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x_others = torch.cat([x_others, extra_token], dim=1)
            x = torch.cat([x_cls, x_others], dim=1)
        
        
        x = x + self.mlp(self.norm2(x))
        return x


class SABlock_CA(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, num_BN: int = 4)-> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_rearrange_kv = Rearrange("b h (kv l d) -> kv b l h d", kv=2, l=num_heads)
        self.input_rearrange_q = Rearrange("b h (l d) -> b l h d", l=num_heads)
        
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        
        self.num_BN = num_BN

    @get_local('att_mat')
    def forward(self, x):
        
        k,v = self.input_rearrange_kv(self.kv(x))
        q = self.input_rearrange_q(self.q(x[:,:self.num_BN,:]))
        
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)

        return x

 

class CABlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, num_BN: int = 4) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """
        super().__init__()   

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_BN = num_BN
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SABlock_CA(hidden_size, num_heads, dropout_rate, num_BN)
        self.norm2 = nn.LayerNorm(hidden_size)
        

    def forward(self, x):
        
        x = x[:,:self.num_BN,:] + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x