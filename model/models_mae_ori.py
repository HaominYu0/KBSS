from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
import torch.nn.functional as F
#from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
MAX_ATOMIC_NUM = 100
#import ipdb
#ipdb.set_trace()
#self.MAX_NUM = np.loadtxt('./model/max_data.txt')

class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83).
        self.embeddings = torch.nn.Embedding(MAX_ATOMIC_NUM, emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z)  # -1 because Z.min()=1 (==Hydrogen)
        return h


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    #if cls_token:
     #   pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=1, in_chans=1,
                 embed_dim=8, depth=2, num_heads=2,
                 decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        self.patch_embed_en = nn.Linear(3, embed_dim)

        #PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = 2#self.patch_embed.num_patches
        whole_patch = num_patches*embed_dim*2
        self.barlow = nn.Sequential(
            nn.Linear(whole_patch,int(embed_dim*2)),
            nn.Softplus(),
            nn.Linear(int(embed_dim*2), embed_dim)
        )
        self.fill_token = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
    
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)




        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, 3, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2],1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.decoder_pos_embed.shape[-2],1, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.fill_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, batch_mask, num_atoms, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        len_keep = num_atoms * (1 - mask_ratio).int().tolist()
        
        two_noise = 2*torch.ones_like(noise)
        noise = torch.where(batch_mask[:,:,0]==1, noise,two_noise )

        """
        #import ipdb
        #ipdb.set_trace()
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder_coord(self, x):
        x = self.patch_embed_en(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed#[:, 1:, :]

        # apply Transformer blocks
        #
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #torch.Size([32, 49, 1024]), torch.Size([32, 192]), torch.Size([32, 192])
        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        ## apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        #x = x[:, 1:, :]

        return x

    def forward_loss1(self, imgs, pred, batch_mask, whole_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # (N, L, patch_size**2 *3)
        #pred= pred[:,:,0]
        target = imgs#self.patchify(imgs)
        mask = 1-batch_mask#[:,:,0]
        mask = torch.logical_and(mask, whole_mask).int()

        loss = (pred - target) ** 2
        #loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        #import ipdb
        #ipdb.set_trace()
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    
    def forward_loss(self, imgs, pred, batch_mask, whole_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # (N, L, patch_size**2 *3)

        target = imgs#self.patchify(imgs)
        mask = 1-batch_mask[:,:,0]
        mask = torch.logical_and(mask, whole_mask[:,:,0]).int()
        #if self.norm_pix_loss:
        #    mean = target.mean(dim=-1, keepdim=True)
        #    var = target.var(dim=-1, keepdim=True)
        #    target = (target - mean) / (var + 1.e-6)**.5
        
        
        loss =F.cross_entropy(pred.permute(0,2,1), imgs.type(torch.LongTensor).cuda(), reduction='none') 
        #F.cross_entropy(pred, imags, reduction='none')
        
        #loss = (pred - target) ** 2
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss




    def forward(self, coord_lattice):
        latent1_coord = self.forward_encoder_coord(coord_lattice)

        
        #torch.Size([32, 192, 1])
        



        return  latent1_coord


