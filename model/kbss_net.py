import os
import yaml
import shutil
import sys
import time
import warnings
import numpy as np
from random import sample
from sklearn import metrics
from datetime import datetime
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from model.dimenet_ori import DimeNetPlusPlusWrap
from model.models_mae_ori import MaskedAutoencoderViT
MAX_ATOMIC_NUM=92


EPSILON = 1e-5


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)






class SimpleClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, int(input_features/2))
        self.fc2 = nn.Linear(int(input_features/2), int(input_features/2))
        self.fc3 = nn.Linear(int(input_features/2), num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
                # Apply ReLU activation function after the second fully-connected layer
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

class finetune_ENDE(nn.Module):
    def __init__(self, cutoff, neibor, bin_num):
        super(finetune_ENDE, self).__init__()
        self.latent_dim = 256
        self.hidden_dim = 256
        self.fc_num_layers = 4
        self.fc_graph_lyers = 3
        self.type_sigma_begin = 5.
        self.type_sigma_end=0.01
        self.num_noise_level = 50
        self.device =  'cuda'
        self.encoder = DimeNetPlusPlusWrap(cutoff=cutoff, max_num_neighbors=neibor)#MaskedAutoencoderViT()
        self.MaskedAutoencoder1 = MaskedAutoencoderViT()
        
        self.emb_dim = 64
        self.emb_linear = nn.Linear(128, self.emb_dim)
        
        self.edge_linear = nn.Linear(128, self.emb_dim)
        self.num_targets = self.emb_dim*2+16
        self.JK = "last"
        self.gnn_type = "gin"
        self.num_layer = 5
        self.dropout_ratio=0
        NUM_NODE_ATTR = 119
        self.fc_out_emb_linear = nn.Linear(128, self.emb_dim)
        self.fc_out_c = nn.Linear(128, 64)
        self.fc_out_emb_linear1 = nn.Linear(128, self.emb_dim*2)

        self.fc_out = nn.Sequential(
                nn.Linear(self.num_targets, self.num_targets//2),
                #nn.Softplus(),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//2, self.num_targets//4),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//4, self.num_targets//8),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//8, 1)
            )
        num_classes = bin_num
        self.classifier = SimpleClassifier(self.num_targets, num_classes)

        
    

    def forward(self, batch_gt, mode='other'):
        decoder_text = False
        batch_gt = batch_gt.cuda()
        hidden_atom, hidden, egde_ij, edge_attr, x_att_, x_emb, rbf = self.encoder(batch_gt)
        lattice_coord = batch_gt.scaled_lattice_tensor
        latent1_coord =self.MaskedAutoencoder1(lattice_coord)
        
        
        hidden_atom_mean =  scatter(hidden_atom, batch_gt.batch, dim=0, reduce='mean')
        hidden_atom_max =  scatter(hidden_atom, batch_gt.batch, dim=0, reduce='max')
        hidden_atom = torch.concat([hidden_atom_mean, hidden_atom_max], -1)
        hidden1_loss = torch.concat([hidden_atom, latent1_coord.view(latent1_coord.shape[0],-1)], dim=-1)
        
        target = self.fc_out(hidden1_loss)
        if mode == 'train':
            target_class = self.classifier(hidden1_loss)
            return target, target_class
        else:
            return target
