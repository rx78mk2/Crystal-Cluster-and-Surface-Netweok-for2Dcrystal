from __future__ import print_function, division

import csv
import functools
import json
import os
import random
from random import shuffle
import warnings

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

electronegativity_dict = { "H":2.2, "He":0, "Li":0.98, "Be":1.57, "B":2.04, "C":2.55, "N":3.04, "O":3.44, "F":3.98, "Ne":0,
    "Na":0.93, "Mg":1.31, "Al":1.61, "Si":1.90, "P":2.19, "S":2.58, "Cl":3.16, "Ar":0, "K":0.82, "Ca":1.0,
    "Sc":1.36, "Ti":1.54, "V":1.63, "Cr":1.66, "Mn":1.55, "Fe":1.83, "Co":1.88, "Ni":1.91, "Cu":1.90, "Zn":1.65,
    "Ga":1.81, "Ge":2.01, "As":2.18, "Se":2.55, "Br":2.96, "Kr":0, "Rb":0.82, "Sr":0.95, "Y":1.22, "Zr":1.33,
    "Nb":1.6, "Mo":2.16, "Tc":1.9, "Ru":2.2, "Rh":2.28, "Pd":2.2, "Ag":1.93, "Cd":1.69, "In":1.78, "Sn":1.96,
    "Sb":2.05, "Te":2.1, "I":2.66, "Xe":0, "Cs":0.79, "Ba":0.89, "La":1.1, "Ce":1.12, "Pr":1.13, "Nd":1.14,
    "Pm":1.13, "Sm":1.17, "Eu":1.2, "Gd":1.2, "Tb":1.1, "Dy":1.22, "Ho":1.23, "Er":1.24, "Tm":1.25, "Yb":1.1,
    "Lu":1.27, "Hf":1.3, "Ta":1.5, "W":2.36, "Re":1.9, "Os":2.2, "Ir":2.2, "Pt":2.28, "Au":2.54, "Hg":2.0,
    "Tl":1.62, "Pb":2.33, "Bi":2.02, "Po":2.0, "At":2.2, "Rn":0, "Fr":0.7, "Ra":0.9, "Ac":1.1, "Th":1.3,
    "Pa":1.5, "U":1.38, "Np":1.36, "Pu":1.28, "Am":1.13, "Cm":1.28, "Bk":1.3, "Cf":1.3, "Es":1.3, "Fm":1.3,
    "Md":1.3, "No":1.3, "Lr":0, "Rf":0, "Db":0, "Sg":0, "Bh":0, "Hs":0, "Mt":0, "Ds":0,
    "Rg":0, "Cn":0, "Nh":0, "Fl":0, "Mc":0, "Lv":0, "Ts":0, "Og":0 }

atom_dict = { "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18, "K":19, "Ca":20,
    "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30,
    "Ga":31, "Ge":32, "As":33, "Se":34, "Br":35, "Kr":36, "Rb":37, "Sr":38, "Y":39, "Zr":40,
    "Nb":41, "Mo":42, "Tc":43, "Ru":44, "Rh":45, "Pd":46, "Ag":47, "Cd":48, "In":49, "Sn":50,
    "Sb":51, "Te":52, "I":53, "Xe":54, "Cs":55, "Ba":56, "La":57, "Ce":58, "Pr":59, "Nd":60,
    "Pm":61, "Sm":62, "Eu":63, "Gd":64, "Tb":65, "Dy":66, "Ho":67, "Er":68, "Tm":69, "Yb":70,
    "Lu":71, "Hf":72, "Ta":73, "W":74, "Re":75, "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80,
    "Tl":81, "Pb":82, "Bi":83, "Po":84, "At":85, "Rn":86, "Fr":87, "Ra":88, "Ac":89, "Th":90,
    "Pa":91, "U":92, "Np":93, "Pu":94, "Am":95, "Cm":96, "Bk":97, "Cf":98, "Es":99, "Fm":100,
    "Md":101, "No":102, "Lr":103, "Rf":104, "Db":105, "Sg":106, "Bh":107, "Hs":108, "Mt":109, "Ds":110,
    "Rg":111, "Cn":112, "Nh":113, "Fl":114, "Mc":115, "Lv":116, "Ts":117, "Og":118 }

class CIFData(Dataset):

    def __init__(self, args, max_num_nbr=12, radius=8, dmin=0, step=0.2):
        self.args = args
        self.max_num_nbr, self.radius = max_num_nbr, radius

        all_DF = pd.read_excel(args.id_prop_root)
        self.id_list = list(all_DF['id'])
        self.prop_list = list(all_DF[args.prop_name])
        atom_init_file = args.atom_init_root
        nan_idx = []
        for idx in range(len(self.prop_list)):
            if(str(self.prop_list[idx])=='nan'):
                nan_idx.append(idx)
        self.nan_idx = nan_idx
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.gdf2 = GaussianDistance(dmin=dmin, dmax=4, step=0.1)


    def __len__(self):
        return len(self.id_list)

    def CreateEmptyMap(self,divide):
        matrix = []
        for i in range(divide):
            b = []
            for j in range(divide):
                b.append(0)
            matrix.append(b)
        return matrix

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):

        cif_id, target = self.id_list[idx], self.prop_list[idx]


        crystal = Structure.from_file(self.args.cif_root+ cif_id +'.cif')
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])

        atom_label_list = []
        for i in range(len(crystal)):
            atom_label_list.append(str(crystal[i].specie))  
        ele_fea = []
        for each in atom_label_list:
            ele_fea.append(electronegativity_dict[each])
        ele_fea = np.array(ele_fea)
        ele_fea = self.gdf2.expand(ele_fea)
        ele_fea  = torch.Tensor(ele_fea )

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        x, y, z, e = [], [], [], []
        for nbrlist in all_nbrs:
            for atom in nbrlist:
                e.append(atom_dict[atom.species_string])
                x.append(atom.x)
                y.append(atom.y)
                z.append(atom.z)

        zipped_lists = zip(z, x, y, e)


        sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
        z, x, y, e = map(list, zip(*sorted_lists))

        level = 0
        level_D = 0.3  #z-tolerance
        top_surface = []
        bottom_surface = []
        length = 20
        width = 20# set of matrix scale
        minus = 10# set of matrix scale
        sidelength = minus + max(length, width)
        divide = 30# set of matrix resolution
        resolution = sidelength / divide

        high = z[0]
        max_level = 0
        for i in range(len(z)):
            if (abs(z[i] - high) <= level_D and level <= max_level):
                top_surface.append([e[i], x[i], y[i], z[i], level])
            elif (abs(z[i] - high) > level_D and level < max_level):
                level += 1
                high = z[i]
                top_surface.append([e[i], x[i], y[i], z[i], level])
            else:
                break
        top_surface2 = [x for i, x in enumerate(top_surface) if x not in top_surface[:i]]

        low = z[-1]
        if (low == high):
            bottom_surface2 = top_surface2
        else:
            max_level = 0
            for i in range(len(z)):
                if (abs(z[i] - low) <= level_D and level <= max_level):
                    bottom_surface.append([e[i], x[i], y[i], z[i], level])

            bottom_surface2 = [x for i, x in enumerate(bottom_surface) if x not in bottom_surface[:i]]

        top_matrix_set = []
        for i in range(max_level + 1):
            top_matrix_set.append(self.CreateEmptyMap(divide))


        for ele in top_surface2:
            x = round((ele[1] + minus) / resolution)
            y = round((ele[2] + minus) / resolution)
            if (0 <= x < divide and 0 <= y < divide):
                top_matrix_set[ele[4]][x][y] = ele[0]

        bottom_matrix_set = []
        for i in range(max_level + 1):
            bottom_matrix_set.append(self.CreateEmptyMap(divide))


        for ele in bottom_surface2:
            x = round((ele[1] + minus) / resolution)
            y = round((ele[2] + minus) / resolution)
            if (0 <= x < divide and 0 <= y < divide):
                bottom_matrix_set[ele[4]][x][y] = ele[0]

        total_matrix_set = [top_matrix_set[0], bottom_matrix_set[0]]
        surface_fea = torch.Tensor(total_matrix_set)

        return (atom_fea, nbr_fea, nbr_fea_idx, ele_fea,surface_fea), target, cif_id


'''###------------------------------------------------------------------------------------------------------------###'''


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


'''###------------------------------------------------------------------------------------------------------------###'''


class AtomCustomJSONInitializer(AtomInitializer):

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


'''###------------------------------------------------------------------------------------------------------------###'''


class GaussianDistance(object):

    def __init__(self, dmin, dmax, step, var=None):

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):

        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


'''###------------------------------------------------------------------------------------------------------------###'''


def get_train_val_loader(dataset, collate_fn=default_collate,batch_size=64, num_workers=1, pin_memory=False,
                         val_indices=None, train_indices=None,**kwargs):

    random_seed = 147
    random.seed(random_seed)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)


    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)

    return train_loader, val_loader


'''###------------------------------------------------------------------------------------------------------------###'''


def collate_pool(dataset_list):

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    batch_v_2d_fea=[]
    batch_surface_fea=[]
    base_idx = 0

    for i, ((atom_fea, nbr_fea, nbr_fea_idx,v_2d_fea,surface_fea), target,  cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        batch_v_2d_fea.append(v_2d_fea)
        batch_surface_fea.append(surface_fea)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0), torch.cat(batch_v_2d_fea,dim=0),torch.stack(batch_surface_fea),
            crystal_atom_idx),\
            torch.stack(batch_target, dim=0),\
            batch_cif_ids


'''###------------------------------------------------------------------------------------------------------------###'''