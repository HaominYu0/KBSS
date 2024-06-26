from __future__ import print_function, division
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from tqdm import tqdm
from typing import List, Tuple, Sequence, Optional
from jarvis.core.atoms import get_supercell_dims
from jarvis.core.specie import Specie
from jarvis.core.utils import random_colors
import numpy as np
import pandas as pd
from collections import OrderedDict
from jarvis.analysis.structure.neighbors import NeighborsAnalysis
from jarvis.core.specie import chem_data, get_node_attributes

import csv
import functools
import json
import os
import random
import warnings
import math
import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from jarvis.core.specie import chem_data, get_node_attributes
from dataset.batch import BatchMasking
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import scipy.sparse as ss
from p_tqdm import p_umap
import pandas as pd
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
#from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
#from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.operations import SymmOp
from pymatgen.transformations.transformation_abc import AbstractTransformation
from .utils import StandardScalerTorch
from dataset.graph import PygGraph

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_val_test_loader(dataset_train, dataset_unseen, dataset_val, dataset_test, collate_fn,batch_size=64,return_test=False, num_workers=1, pin_memory=False,**kwargs):
    #setup_seed(random_seed)
    g = torch.Generator()
    g.manual_seed(0)
    #import ipdb
    #ipdb.set_trace()

    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              num_workers=num_workers,
                              #worker_init_fn=seed_worker,
                              collate_fn=collate_fn,
                              shuffle=True,
                              pin_memory=pin_memory)
    
    unseen_batch_size = math.floor(10000/math.ceil(len(dataset_train)/batch_size))
    useen_train_loader = DataLoader(dataset_unseen, batch_size=unseen_batch_size,
                              num_workers=num_workers,
                              #worker_init_fn=seed_worker,
                              collate_fn=collate_fn,
                              shuffle=True,
                              pin_memory=pin_memory)


    val_loader = DataLoader(dataset_val, batch_size=batch_size,
                            num_workers=num_workers,
                            #worker_init_fn=seed_worker,
                            collate_fn=collate_fn,
                            shuffle=False,
                            pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset_test, batch_size=batch_size,
                                 num_workers=num_workers,
                                 #worker_init_fn=seed_worker,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 pin_memory=pin_memory)
    if return_test:
        return train_loader, useen_train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(batches):
    batches = [x for x in batches]
    batches = BatchMasking.from_data_list(batches)
    return batches





class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, mode, i_num, task, root_dir, goal,max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=28):
        self.root_dir = root_dir
        self.task = task
        self.pygGraph = PygGraph()
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        tail_file = goal  #"_fold"+str(i_num)
        id_prop_file = os.path.join(self.root_dir, 'id_prop_'+mode+'_'+tail_file+'.csv')

        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        if mode == 'train':
            reader = pd.read_csv(id_prop_file)
        else:
            reader = pd.read_csv(id_prop_file)
        titles = reader.columns
        self.id_prop_data = np.array(reader[[titles[0], goal]]).tolist()
        if mode == 'train':
           self.id_goal = reader[goal].tolist()
        if mode == 'val':
            self.id_goal_val = reader[goal].tolist()
        cif_fns  = []
        
        material_id_list = reader[titles[0]].to_list()
        for material_id in material_id_list:
            cif_fns.append(self.root_dir+material_id+'.cif')

        self.cif_data = cif_fns
        self.cached_data = self.preprocess(self.id_prop_data, self.cif_data)
        
        self.add_scaled_lattice_prop(self.cached_data)
        lattice_scaler = self.get_scaler_from_data_list(
                            self.cached_data,
                            'scaled_lattice')
        self.lattice_scaler = lattice_scaler
        #random.seed(random_seed)
        #random.shuffle(self.cif_data)
        atom_init_file = os.path.join('dataset/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

    def  get_scaler_from_data_list(self,data_list, key):
        targets = torch.tensor([d[key] for d in data_list])
        scaler = StandardScalerTorch()
        scaler.fit(targets)
        return scaler


    def lennard_jones_potential(self, crys, epsilon, sigma):
        energy = 0.0
        for i in range(len(crys)):
            for j in range(i + 1, len(crys)):
                distance = crys.get_distance(i, j)
                energy += 4 * epsilon * ((sigma / distance) ** 12 - (sigma / distance) ** 6)
        return energy

    def monte_carlo_step(self, crys, temperature, epsilon, sigma):
        new_crys = crys.copy()
        # 随机选择一个原子移动
        atom_index = np.random.randint(len(new_crys))
        move = np.random.uniform(-step_size, step_size, 3)
        new_crys.translate_sites([atom_index], move, frac_coords=False) 
        # 计算能量差
        old_energy = self.lennard_jones_potential(crys, epsilon, sigma)
        new_energy = self.lennard_jones_potential(new_crys, epsilon, sigma)
        delta_energy = new_energy - old_energy
    
        # Metropolis准则
        if delta_energy < 0 or np.exp(-delta_energy / (temperature * 8.617333262145e-5)) > np.random.rand():
            return new_crys, True  # 接受新结构
        else:
            return crys, False 

    






    def preprocess(self, id_prop_data, cif_data):
        def process_one(id_data, cif_fn):
            cif_id, target = id_data
            cif_id = cif_fn.split('/')[-1].replace('.cif', '')
            crys = Structure.from_file(cif_fn)

            niggli=True
            if niggli:
                crys = crys.get_reduced_structure()

            crys =Structure(
                    lattice=Lattice.from_parameters(*crys.lattice.parameters),
                    species=crys.species,
                    coords=crys.frac_coords,
                    coords_are_cartesian=False)
            graph_arrays = self.crys_structure(crys)

            result_dict = {
                    'mp_id':cif_id,
                    'cif': crys,
                    'target':target,
                    'graph_arrays':graph_arrays,
                    }
            return result_dict

        unordered_results = p_umap(
                process_one,
                [id_prop_data[idx] for idx in range(len(id_prop_data)) ],
                [cif_data[idx] for idx in range(len(cif_data))],
                num_cpus = 60
                )

        mpid_to_results = {result['mp_id']: result for result in unordered_results}
        ordered_results = [mpid_to_results[id_prop_data[idx][0]]
                       for idx in range(len(id_prop_data))]
        #ordered_results = process_one(id_prop_data[0], cif_data[0])

        return ordered_results





    def add_scaled_lattice_prop(self, data_list):
       for dict in data_list:
            graph_arrays = dict['graph_arrays']
            # the indexes are brittle if more objects are returned
            lengths = graph_arrays[2]
            angles = graph_arrays[3]
            num_atoms = graph_arrays[-1]
            assert lengths.shape[0] == angles.shape[0] == 3
            assert isinstance(num_atoms, int)

            #lengths = lengths / float(num_atoms)**(1/3)

            dict['scaled_lattice'] = np.concatenate([lengths, angles])

    def crys_structure(self, crys):
            CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)
            try:

                crystal_graph = StructureGraph.with_local_env_strategy(crys, CrystalNN)
               # graph = self.pygGraph.atom_dgl_multigraph(crys,neighbor_strategy="k-nearest",cutoff=5.0,atom_features="atomic_number",max_neighbors=12,compute_line_graph=False,use_canonize=False,use_lattice=False,use_angle=False,)
                #g = graph
                #z = g.ndata.pop("atom_features")
                #g.ndata["atomic_number"] = z
                #z = z.type(torch.IntTensor).squeeze()
                #f = torch.tensor(features[z]).type(torch.FloatTensor)
                #if g.num_nodes() == 1:
                #    f = f.unsqueeze(0)
                #g.ndata["atom_features"] = f
                #self.prepare_batch = prepare_dgl_batch
                #self.prepare_batch = prepare_line_graph_batch      
                   # print("building line graphs")
                   # self.line_graphs = []
                   # for g in tqdm(graphs):
                #lg = g.line_graph(shared=True)
                #lg.apply_edges(compute_bond_cosines)
                #self.line_graphs.append(lg)


                #import ipdb
                #ipdb.set_trace()


            except (RuntimeError, TypeError, NameError, ValueError):
                print("crystal_error")
                crys = Structure.from_file('/data/cs.aau.dk/haominyu/cdvae/Dataset/MP_DATA_post/mp-1023940.cif')
                crystal_graph = StructureGraph.with_local_env_strategy(crys, CrystalNN)
                #graph = self.pygGraph.atom_dgl_multigraph(crys, neighbor_strategy="k-nearest", cutoff=8.0, atom_features="atomic_number", max_neighbors=12, compute_line_graph=False, use_canonize=False, use_lattice=False, use_angle=False,)

            frac_coords = crys.frac_coords
            atom_types = crys.atomic_numbers
            lattice_parameters = crys.lattice.parameters
            lengths = lattice_parameters[:3]
            angles = lattice_parameters[3:]
            edge_indices, to_jimages = [], []
            #i_choose = 0
            for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
                            edge_indices.append([j, i])
                            to_jimages.append(to_jimage)
                            edge_indices.append([i, j])
                            to_jimages.append(tuple(-tj for tj in to_jimage))

            #-------#
            #edge_indices.append([0, 0])
            #to_jimages.append([0,0])

            graph_edge_attr = torch.zeros((len(edge_indices), 2), dtype=torch.long)
            atom_types = np.array(atom_types)
            num_atoms = atom_types.shape[0]

            lengths, angles = np.array(lengths), np.array(angles)
            scaled_lattice = np.concatenate([lengths, angles])

            edge_indices = np.array(edge_indices)
            to_jimages = np.array(to_jimages)

            return frac_coords, atom_types, lengths, angles,edge_indices, to_jimages, scaled_lattice, graph_edge_attr, num_atoms#graph.x, graph.edge_index, graph.edge_attr, num_atoms

    def graph_file(self, graph_arrays, cif_id, target):
        (frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, scaled_lattice, edge_attr, num_atoms) = graph_arrays


        lattice_tensor = torch.concat([torch.Tensor(lengths).view(1, 1,-1), torch.Tensor(angles).view(1, 1,-1)],1)
        #scaled_lattice = (torch.Tensor(scaled_lattice)-self.lattice_scaler.means)/self.lattice_scaler.stds
        scaled_lattice_tensor = torch.Tensor(scaled_lattice.reshape(1,2,3))

        #import ipdb
        #ipdb.set_trace()
        


        data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                #x = torch.LongTensor(np.array(x)),
                #edge_index = torch.LongTensor(edge_index).contiguous(),
                edge_attr = torch.Tensor(edge_attr),

                #graph_edge_attr = torch.Tensor(graph_edge_attr),
                #frac_tensor = frac_tensor,
                #atom_tensor = atom_tensor,
                lattice_tensor  = lattice_tensor,
                scaled_lattice_tensor = scaled_lattice_tensor,


                lengths=torch.Tensor(lengths).view(1, -1),
                scaled_lattice = torch.Tensor(scaled_lattice).view(1, -1),
                #scaled_mean_std = self.lattice_scaler,
                angles=torch.Tensor(angles).view(1, -1),
               # edge_index=torch.LongTensor(
               # edge_indices.T).contiguous(),
                edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,
                cif_id = cif_id,
                target = target)
        return data

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
            data_dict = self.cached_data[idx] 
            crys = data_dict['cif']
            mp_id = data_dict['mp_id']
            target = data_dict['target']

            data = self.graph_file(data_dict['graph_arrays'], mp_id, target)
            return idx, data
