import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn

class E2NN(nn.Module):
    #E(2) equivariant convolution
    def __init__(self, in_channels=1, num_classes=10, N=32,use_bot=False):
        super().__init__()
        self.use_bot=use_bot
        self.r2_act = gspaces.Rot2dOnR2(N=N)


        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * in_channels)

        hidden_size = 16
        out_type_hid=e2nn.FieldType(self.r2_act,
                    3*[self.r2_act.regular_repr])
        out_type=e2nn.FieldType(self.r2_act,
                    6*[self.r2_act.regular_repr])

        self.net = e2nn.SequentialModule(
            e2nn.R2Conv(in_type=in_type,out_type=out_type_hid,kernel_size=3),
            e2nn.InnerBatchNorm(out_type_hid),
            e2nn.ReLU(out_type_hid),
            e2nn.PointwiseMaxPoolAntialiased(out_type_hid, kernel_size=3),

            e2nn.R2Conv(out_type_hid, out_type, kernel_size=3),
            e2nn.ReLU(out_type),

        )

        self.fc1 = nn.Linear(9408, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self,x):
        if (self.use_bot == False):
            x = x[:, 0, :, :].unsqueeze(1)
            x = e2nn.GeometricTensor(x, self.net.in_type)

            # 前向传播
            x = self.net(x)

            # 提取PyTorch张量并展平 (保留批处理维度)
            x=x.tensor.reshape(x.shape[0], -1)
            x=F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))

            return x
        else:
            x1 = x[:, 0, :, :].unsqueeze(1)
            x1 = e2nn.GeometricTensor(x1, self.net.in_type)
            x2 = x[:, 1, :, :].unsqueeze(1)
            x2 = e2nn.GeometricTensor(x2, self.net.in_type)


            x1 = self.net(x1)


            x1=x1.tensor.reshape(x1.shape[0], -1)
            x1=F.tanh(self.fc1(x1))
            x1 = F.tanh(self.fc2(x1))

            x2 = self.net(x2)


            x2 = x2.tensor.reshape(x2.shape[0], -1)
            x2 = F.tanh(self.fc1(x2))
            x2 = F.tanh(self.fc2(x2))

            x = (x1 + x2) / 2

            return x


class LeNet(nn.Module):
    #normal CNN
    def __init__(self, num_classes=32,use_bot=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=1, padding=0)
        #self.fc1 = nn.Linear(576, 256)
        #self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(216, num_classes)
        self.use_bot=use_bot

    def forward(self, x):

        if(self.use_bot==False):
            x = x[:, 0, :, :].unsqueeze(1)
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))


            x = x.view(x.size(0), -1)


            x = F.tanh(self.fc3(x))#serelu
            #x = F.relu(self.fc2(x))

            #x = self.fc3(x)

            return x
        else:
            x1 = x[:, 0, :, :].unsqueeze(1)
            x2 = x[:, 1, :, :].unsqueeze(1)

            x1 = F.relu(F.max_pool2d(self.conv1(x1), 2))
            x1 = F.relu(F.max_pool2d(self.conv2(x1), 2))

            # Flatten the tensor
            x1 = x1.view(x1.size(0), -1)

            # Fully connected layers
            x1 = F.tanh(self.fc3(x1))  # serelu


            x2 = F.relu(F.max_pool2d(self.conv1(x2), 2))
            x2 = F.relu(F.max_pool2d(self.conv2(x2), 2))

            # Flatten the tensor
            x2 = x2.view(x2.size(0), -1)

            # Fully connected layers
            x2 = F.tanh(self.fc3(x2))  # serelu

            x = (x1 + x2) / 2

            return x

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x):
        return self.sp(x) - self.shift


class ClusterCalculator(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len,state_fea_len,middle_size=64):
        super(ClusterCalculator, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.state_fea_len =  state_fea_len
        self.embedding_len=32
        self.middle_size = middle_size

        embed_size_e = (atom_fea_len*2+nbr_fea_len+state_fea_len+state_fea_len)
        embed_size_v = atom_fea_len + nbr_fea_len
        embed_size_u = state_fea_len


        self.MLP_e = nn.Sequential(     
            nn.Linear( embed_size_e, middle_size),
            nn.Tanh(),
            nn.Linear(middle_size, nbr_fea_len*2),

        )

        self.MLP_v = nn.Sequential(
            nn.Linear(embed_size_v, middle_size),
            nn.Tanh(),
            nn.Linear(middle_size, atom_fea_len*2),
            nn.Tanh(),
        )

        self.MLP_u = nn.Sequential(
            nn.Linear(embed_size_u, state_fea_len),

        )



        self.bn1 = nn.BatchNorm1d(2 * self.nbr_fea_len)
        self.bn2 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn3 = nn.BatchNorm1d(self.state_fea_len)




    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx,state_fea, crystal_atom_idx):

        ori_atom_fea=atom_in_fea
        ori_nbr_fea = nbr_fea
        ori_state_fea = state_fea
        N, M = nbr_fea_idx.shape



        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        state_nbr_fea = state_fea[nbr_fea_idx, :]

        state_fea_expand2 = state_fea.unsqueeze(1).expand(N, M, self.state_fea_len)#(total_atom_num,12,128)
        total_edge_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),state_fea_expand2,
              nbr_fea,atom_nbr_fea,state_nbr_fea], dim=2)
        total_edge_fea_out1 = self.MLP_e(total_edge_fea)
        N, M ,t= total_edge_fea_out1.shape
        total_edge_fea_out1 = self.bn1(total_edge_fea_out1.view(
            -1, self.nbr_fea_len*2)).view(N, M, self.nbr_fea_len*2)
        edge_filter,edge_core = total_edge_fea_out1.chunk(2,dim=2)
        total_edge_fea_out = edge_filter*edge_core+ori_nbr_fea



        total_edge_mean = torch.mean(total_edge_fea_out,dim=1)
        total_atom_fea = torch.cat(
            [total_edge_mean,
             atom_in_fea], dim=1)
        total_atom_fea_out1 = self.bn2(self.MLP_v(total_atom_fea))
        atom_filter, atom_core = total_atom_fea_out1.chunk(2, dim=1)
        total_atom_fea_out = atom_filter*atom_core+ori_atom_fea


        total_state_fea = torch.cat(
            [state_fea], dim=1)
        total_state_fea_out = self.bn3(self.MLP_u(total_state_fea))+ori_state_fea



        return  total_edge_fea_out, total_atom_fea_out,total_state_fea_out


    def pooling(self, atom_fea, crystal_atom_idx):

        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)




'''###------------------------------------------------------------------------------------------------------------###'''


class ClusterModel(nn.Module):
    def __init__(self, orig_atom_fea_len, orig_nbr_fea_len,
                 atom_fea_len=32, n_block=3, h_fea_len=128,orig_state_fea_len=41,state_fea_len=32,surface_fea_len=12,
                 nbr_fea_len=32, use_bottom=False,use_E2CNN=False,pretrain=False):


        super(ClusterModel, self).__init__()
        self.pretrain=pretrain
        self.surface_fea_len=surface_fea_len
        self.embedding_atom = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embedding_state = nn.Linear(orig_state_fea_len, state_fea_len)
        self.embedding_nbr = nn.Linear(orig_nbr_fea_len, nbr_fea_len)
        self.ClusterNet = nn.ModuleList([ClusterCalculator(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len,state_fea_len=state_fea_len)
                                    for _ in range(n_block)])

        self.Fc1 = nn.Linear(atom_fea_len+nbr_fea_len+state_fea_len+surface_fea_len, h_fea_len)
        self.tanh = nn.Tanh()



        self.fc_out = nn.Linear(h_fea_len, 1)
        if(use_E2CNN==False):
            self.Surface_conv = LeNet(num_classes=surface_fea_len,use_bot=use_bottom)
        else:
            self.Surface_conv=E2NN(num_classes=surface_fea_len,use_bot=use_bottom)


    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, state_fea, surface_fea,crystal_atom_idx):



        atom_fea = self.embedding_atom(atom_fea)
        state_fea = self.embedding_state(state_fea)
        nbr_fea = self.embedding_nbr(nbr_fea)
        for block in self.ClusterNet:
            nbr_fea,atom_fea,state_fea = block( atom_fea, nbr_fea, nbr_fea_idx,state_fea, crystal_atom_idx)

        nbr_fea= torch.sum(nbr_fea,dim=1)
        crys_fea = torch.cat([nbr_fea, atom_fea, state_fea], dim=1)
        crys_fea = self.pooling( crys_fea ,crystal_atom_idx)

        if(self.pretrain==True):
            surface_fea=torch.zeros(crys_fea.shape[0],self.surface_fea_len)
        else:
            surface_fea = self.Surface_conv(surface_fea)
        crys_fea = torch.cat([crys_fea, surface_fea], dim=1)

        crys_fea = self.Fc1(crys_fea)
        crys_fea = self.tanh(crys_fea)



        out = self.fc_out(crys_fea)

        return out


    def pooling(self, atom_fea, crystal_atom_idx):

        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


'''###------------------------------------------------------------------------------------------------------------###'''