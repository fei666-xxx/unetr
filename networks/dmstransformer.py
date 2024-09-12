# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Sequence
import math

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.utils import optional_import

from .nets import SPABlock, MCABlock

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")



def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim #2
    dims = idx.shape # B N-1-toknes
    n_idx = dims[-1]# N-1-toknes
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl




class SABlock(nn.Module):
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


 

class TransformerBlock(nn.Module):
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
        self.attn = SABlock(hidden_size, num_heads, dropout_rate, enhance_rate=enhance_rate, fuse_rate=fuse_rate, has_cls=has_cls, CSA=CSA)
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



class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        enhance_rate = (0.0,),
        fuse_rate = (0.0,),
        
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        
        # self.patch_embedding_p8 = PatchEmbeddingBlock(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=[p//2 for p in patch_size],
        #     hidden_size=hidden_size,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        
        if len(enhance_rate) == 1:
            enhance_rate = enhance_rate * num_layers
        if len(fuse_rate) == 1:
            fuse_rate = fuse_rate * num_layers
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, enhance_rate=enhance_rate[i], fuse_rate=fuse_rate[i]) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        if self.classification:
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore
                

    def forward(self, x):
        #x_p8 = self.patch_embedding_p8(x)
        x = self.patch_embedding(x)
        #B, N, _ = x.shape
        #x_level = np.zeros((B,N),dtype=np.int)
        
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out



class DMSTransformer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int, int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
        modality: int = 4,
        enhance_rate = (0.0,),
        fuse_rate = (0.0,),
        pretrained = False,
        num_classes = 2,
        pretrained_model_name = "UNETR_model_best_acc.pth",
        open_spatten = False,
        has_cls = False,
        open_mcatten = False,
    ) -> None:


        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.hidden_size = hidden_size
        self.modality = modality
        self.pretrained_model_name = pretrained_model_name
        self.open_spatten = open_spatten
        self.open_mcatten = open_mcatten
        self.has_cls = has_cls
        
        self.blocks = nn.ModuleList([ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=False,
            dropout_rate=dropout_rate,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
        ) for _ in range(1)])
        
        
        if self.open_mcatten:
            # self.fusion = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, CSA=True)
            self.fusion = nn.ModuleList([MCABlock(hidden_size, mlp_dim, num_heads, dropout_rate, has_cls=has_cls, CSA=open_mcatten) for _ in range(4)])
        
        if self.open_spatten:
            # self.same_pos_fusion = TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
            self.same_pos_fusion = nn.ModuleList([SPABlock(hidden_size, mlp_dim, num_heads, dropout_rate, has_cls=has_cls, gamma=4.0) for _ in range(4)])
        
        if self.has_cls:
            self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, hidden_size)) for _ in range(4)])
        
        #self.head = nn.Linear(in_features=hidden_size*modality, out_features=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_dim, num_classes))
        
        if pretrained:
            self.load_pretrain_model()



    def load_pretrain_model(self):

        if self.pretrained_model_name=='UNETR_model_best_acc.pth':
            state_dict = torch.load('./pretrained_models/UNETR_model_best_acc.pth')
            cls_param = {'cls_token':state_dict['vit.patch_embedding.cls_token']}
            patch_param = {
                            #'position_embeddings':state_dict['vit.patch_embedding.position_embeddings'],
                            'patch_embeddings.1.weight':state_dict['vit.patch_embedding.patch_embeddings.1.weight'],
                            'patch_embeddings.1.bias':state_dict['vit.patch_embedding.patch_embeddings.1.bias'],}
            layer_param = [{
                            'mlp.linear1.weight':state_dict['vit.blocks.'+str(i)+'.mlp.linear1.weight'],
                            'mlp.linear1.bias':state_dict['vit.blocks.'+str(i)+'.mlp.linear1.bias'],
                            'mlp.linear2.weight':state_dict['vit.blocks.'+str(i)+'.mlp.linear2.weight'],
                            'mlp.linear2.bias':state_dict['vit.blocks.'+str(i)+'.mlp.linear2.bias'],
                            'norm1.weight':state_dict['vit.blocks.'+str(i)+'.norm1.weight'],
                            'norm1.bias':state_dict['vit.blocks.'+str(i)+'.norm1.bias'],
                            'norm2.weight':state_dict['vit.blocks.'+str(i)+'.norm2.weight'],
                            'norm2.bias':state_dict['vit.blocks.'+str(i)+'.norm2.bias'],
                            'attn.out_proj.weight':state_dict['vit.blocks.'+str(i)+'.attn.out_proj.weight'],
                            'attn.out_proj.bias':state_dict['vit.blocks.'+str(i)+'.attn.out_proj.bias'],
                            'attn.qkv.weight':state_dict['vit.blocks.'+str(i)+'.attn.qkv.weight'],}  for i in range(self.num_layers)]
            norm_param = {
                            'weight':state_dict['vit.norm.weight'],
                            'bias':state_dict['vit.norm.bias']}
                        
            fusion_param = {
                            'mlp.linear1.weight':state_dict['vit.blocks.'+str(self.num_layers-1)+'.mlp.linear1.weight'],
                            'mlp.linear1.bias':state_dict['vit.blocks.'+str(self.num_layers-1)+'.mlp.linear1.bias'],
                            'mlp.linear2.weight':state_dict['vit.blocks.'+str(self.num_layers-1)+'.mlp.linear2.weight'],
                            'mlp.linear2.bias':state_dict['vit.blocks.'+str(self.num_layers-1)+'.mlp.linear2.bias'],
                            'norm1.weight':state_dict['vit.blocks.'+str(self.num_layers-1)+'.norm1.weight'],
                            'norm1.bias':state_dict['vit.blocks.'+str(self.num_layers-1)+'.norm1.bias'],
                            'norm2.weight':state_dict['vit.blocks.'+str(self.num_layers-1)+'.norm2.weight'],
                            'norm2.bias':state_dict['vit.blocks.'+str(self.num_layers-1)+'.norm2.bias'],
                            'attn.out_proj.weight':state_dict['vit.blocks.'+str(self.num_layers-1)+'.attn.out_proj.weight'],
                            'attn.out_proj.bias':state_dict['vit.blocks.'+str(self.num_layers-1)+'.attn.out_proj.bias'],
                            'attn.qkv.weight':state_dict['vit.blocks.'+str(self.num_layers-1)+'.attn.qkv.weight'],}
            
            print("Use pretrained weights from UNETR(trained on BTCV dataset)")
            for i in range(len(self.blocks)):
                self.blocks[i].patch_embedding.load_state_dict(patch_param, strict=False)
        
        elif self.pretrained_model_name=='imagenet21k+imagenet2012_ViT-B_16.pth':
            state_dict = torch.load('./pretrained_models/imagenet21k+imagenet2012_ViT-B_16.pth')['state_dict']
            
            cls_param = {'cls_token':state_dict['cls_token']}

            qkv_param = [[state_dict['transformer.encoder_layers.'+str(i)+'.attn.query.weight'].reshape(768,-1),
                          state_dict['transformer.encoder_layers.'+str(i)+'.attn.key.weight'].reshape(768,-1),
                          state_dict['transformer.encoder_layers.'+str(i)+'.attn.value.weight'].reshape(768,-1),
                          ] for i in range(self.num_layers)]
            
            layer_param = [{
                            'mlp.linear1.weight':state_dict['transformer.encoder_layers.'+str(i)+'.mlp.fc1.weight'],
                            'mlp.linear1.bias':state_dict['transformer.encoder_layers.'+str(i)+'.mlp.fc1.bias'],
                            'mlp.linear2.weight':state_dict['transformer.encoder_layers.'+str(i)+'.mlp.fc2.weight'],
                            'mlp.linear2.bias':state_dict['transformer.encoder_layers.'+str(i)+'.mlp.fc2.bias'],
                            'norm1.weight':state_dict['transformer.encoder_layers.'+str(i)+'.norm1.weight'],
                            'norm1.bias':state_dict['transformer.encoder_layers.'+str(i)+'.norm1.bias'],
                            'norm2.weight':state_dict['transformer.encoder_layers.'+str(i)+'.norm2.weight'],
                            'norm2.bias':state_dict['transformer.encoder_layers.'+str(i)+'.norm2.bias'],
                            'attn.out_proj.weight':state_dict['transformer.encoder_layers.'+str(i)+'.attn.out.weight'].reshape(-1,768),
                            'attn.out_proj.bias':state_dict['transformer.encoder_layers.'+str(i)+'.attn.out.bias'],
                            'attn.qkv.weight': torch.cat(qkv_param[i],dim=1).transpose(0,1)}  for i in range(self.num_layers)]
            norm_param = {
                            'weight':state_dict['transformer.norm.weight'],
                            'bias':state_dict['transformer.norm.bias']}
            
            print("pretrained on imagenet1k and imagenet21k")
            
        else:
            raise ValueError("Unsupported pretrained model name: " + str(self.pretrained_model_name))
        
        with torch.no_grad():
            for i in range(len(self.blocks)):
                self.blocks[i].cls_token.copy_(cls_param['cls_token'])
                for j in range(self.num_layers):
                    self.blocks[i].blocks[j].load_state_dict(layer_param[j], strict=False)
                self.blocks[i].norm.weight.copy_(norm_param['weight'])
                self.blocks[i].norm.bias.copy_(norm_param['bias'])
                
            # self.bottlenecks.copy_(cls_param['cls_token'].expand(-1,4,-1))
                

    def forward(self, x_in):
    
        # x_in B C H W D -> 1x4x128x128x128
        output = []
        for i in range(self.modality):
            tmp = x_in[:,i,:,:,:].unsqueeze(1)
            _, tmp = self.blocks[0](tmp)
            output.append(tmp)

        out = []
        for i in range(4):
            fuse_input = []
            for j in range(self.modality):
                fuse_input.append(output[j][2+3*i])
            fuse_input = torch.cat(fuse_input, dim=1)


            if self.has_cls:
                cls_token = self.cls_token[i].expand(x_in.shape[0], -1, -1)
                fuse_input = torch.cat((cls_token, fuse_input), dim=1)

            # # --------------------
            # # spaatten_version_1
            # # --------------------
            # if self.open_spatten:
            #     res = []
            #     for i in range(self.num_patches):
            #         tmp = []
            #         for j in range(self.modality):
            #             tmp.append(output[:, i+j*self.num_patches, :].unsqueeze(1))
            #         tmp = torch.cat(tmp, dim=1)
            #         tmp = self.same_pos_fusion(tmp)
            #         res.append(tmp)
            #     res = torch.stack(res, dim=0)
            #     res = res.permute(2, 1, 0, 3).unbind(0) # N B M D -> M B N D -> M [B N D]
            #     output = torch.cat(res, dim=1)   
        

            # --------------------
            # spaatten_version_2
            # --------------------
            if self.open_mcatten:
                fuse_input = self.fusion[i](fuse_input)
            
            if self.open_spatten:
                fuse_input = self.same_pos_fusion[i](fuse_input)

            out.append(torch.mean(fuse_input,dim=1).unsqueeze(1))
                

        out = torch.cat(out,dim=1)
        out = torch.mean(out,dim=1)
        logits = self.head(out)
        
        return logits
