import torch.nn as nn

from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        #print("start",x.size())    #start torch.Size([1, 3, 256, 256])
        x = self.proj(x)           #flatten before torch.Size([1, 768, 16, 16])
        B, C, H, W = x.shape
        if H == 16:
            # 2. 取中间 8x8
            x_center = x.clone()[:, :, 4:12, 4:12]  # -> (B, 768, 8, 8)
            # 3. 展平空间维度 (8x8=64)，得到 (B, 64, 768)
            x_center = x_center.flatten(2).permute(0, 2, 1)  # -> (B, 64, 768)
        if self.flatten:
            #print("flatten before",x.size())       #flatten before torch.Size([1, 768, 16, 16])
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            #print("flatten transpose",x.size())   #flatten transpose torch.Size([1, 256, 768])
        if H == 16:
            x_t = self.norm(x_center)
            x = self.norm(x)
            #print("after",x.size())                #after torch.Size([1, 256, 768])
            return x,x_t
        else:
            x = self.norm(x)
            #print("after",x.size())                #after torch.Size([1, 256, 768])
            return x