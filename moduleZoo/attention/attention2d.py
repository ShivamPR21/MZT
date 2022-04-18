from typing import Tuple

import torch
import torch.nn as nn


class SelfAttention2d(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim:int , activation:int):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.rand(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = (self.gamma*out + x)/(1 + self.gamma)
        return out, attention


class SelfAttention1d(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim:int , activation:int):
        super().__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size = 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size = 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size = 1)
        self.gamma = nn.Parameter(torch.rand(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
        """
        m_batchsize,C,n = x.size()
        proj_query = self.query_conv(x).permute(0,2,1) # B X N X (C)
        proj_key = self.key_conv(x) # B X C x (N)

        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)

        proj_value = self.value_conv(x) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))

        out = (self.gamma*out + x)/(1 + self.gamma)
        return out, attention
