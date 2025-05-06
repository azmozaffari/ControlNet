import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, patch_size, in_channels, patch_embed, out_embed, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.in_embed = patch_embed
        self.out_embed = out_embed
        self.patch_embed = nn.Conv2d(in_channels, patch_embed*self.n_heads, patch_size, stride=patch_size)
        self.out_linear_layer = nn.Linear(self.in_embed, self.out_embed)




    def forward(self, x):
        x = self.patch_embed(x)
        batches, features,patch_h, patch_w  = x.size()
        x = x.reshape(batches, self.n_heads, int(features/self.n_heads), patch_h * patch_w)  #(batch,head, feature, sequence)

        query = x.permute(0,1,3,2)
        key = x
        value = x     
        
        weights = query@key  ##(batch,head, sequence, sequence)


        s = torch.matmul(weights, value.permute(0,1,3,2))##(batch,head, sequence)

        













    