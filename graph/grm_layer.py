""""
Define a generic GRM layer model
"""
from pickletools import decimalnl_short
import  torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph_util import *
from omegaconf import OmegaConf
from global_settings import GPU_ID
from torch.autograd import Variable
from torch.nn import Parameter
import math
#from layers import *

cuda_suffix = 'cuda:' + str(GPU_ID) if len(str(GPU_ID)) == 1 else "cuda"
device = torch.device(cuda_suffix if torch.cuda.is_available() else "cpu")




#Graph Reasoning Module
#Graph convolution
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight.cuda())
        #support = torch.matmul(input, self.weight)
        adj=adj.to(input.cuda())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


#Semantic Mapping Module
class SemanticToLocal(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels):
        super(SemanticToLocal, self).__init__()

        # It is necessary to calculate the mapping weight matrix from 
        #symbol nodes to local features for each image. [?, H*W, M]
        # The W in the paper is as follows
        self.conv1 = nn.Conv2d(256 +256, 1,
                              kernel_size=1, stride=1)
        
        self.relu = nn.ReLU(inplace=False)

    def compute_compat_batch(self, batch_input, batch_evolve):
        # batch_input [H, W, Dl]
        # batch_evolve [M, Dc]
        # [H, W, Dl] => [H * W, Dl] => [H*W, M, Dl]
        H = batch_input.shape[0]
        W = batch_input.shape[1]
        M = batch_evolve.shape[0]
        Dl = batch_input.shape[-1]
        batch_input = batch_input.reshape( H * W, Dl)
        batch_input = batch_input.unsqueeze(1).repeat([1,M,1])
        # [M,Dc] => [H*W, M, Dc]
        batch_evolve = batch_evolve.unsqueeze(0).repeat([H*W, 1, 1])
        # [H*W, M, Dc+Dl] 
        batch_concat = torch.cat([batch_input, batch_evolve], axis=-1)
        # [H*W, M, Dc+Dl] =>[1,H*W, M, Dc+Dl]
        batch_concat = batch_concat[np.newaxis,:,:,:]
        # [H*W, M, Dc+Dl] =>[1,Dc+Dl,H*W, M]
        batch_concat = batch_concat.transpose(2,3).transpose(1,2)
        #print("@@@@@@@batch_concat",batch_concat.size())
        #[1,Dc+Dl,H*W, M] =>[1,1,H*W, M]
        mapping = self.conv1(batch_concat)
        #[1,1,H*W, M] => [1, H*W, M, 1]
        mapping = mapping.transpose(1,2).transpose(2,3)
        #[1,1,H*W, M] => [H*W, M, 1]
        mapping = mapping.view(-1,mapping.size(2),mapping.size(3))
        #[H*W, M,1] => [H*W, M]
        mapping = mapping.view(mapping.size(0), -1)
        mapping = F.softmax(mapping, dim=0)
        return  mapping

    def forward(self, x, evolved_feat):
        # [?, Dl, H, W] , [?, M, Dc]
        input_feat = x 
        evolved_feat = evolved_feat
        # [?, H, W, Dl]
        input_feat = input_feat.transpose(1,2).transpose(2, 3)
        batch_list = []
        for index in range(input_feat.size(0)):
            batch = self.compute_compat_batch(input_feat[index], evolved_feat[index])
            batch_list.append(batch)
        # [?, H*W, M]
        mapping = torch.stack(batch_list, dim=0)
        # [?, M, Dc] => [? * M, Dc] => [? * M, Dl] => [?, M, Dl]
        Dl = input_feat.size(-1)
        M = evolved_feat.size(1)
        H = input_feat.size(1)
        W = input_feat.size(2)
        # [?, M, Dc] => [? * M, Dc]
        #[?, H*W, M] @ [? , M, Dl] => [?, H*W, Dl]
        applied_mapping = torch.bmm(mapping, evolved_feat)
        applied_mapping = self.relu(applied_mapping)
        #[?, H*W, Dl] => [?, H, W, Dl]
        applied_mapping = applied_mapping.reshape(input_feat.size(0), H , W, Dl)
        #[?, H, W, Dl] => [?, Dl, H, W]
        applied_mapping = applied_mapping.transpose(2,3).transpose(1,2)

        return applied_mapping

#overall model layer
class GRMLayer(nn.Module):

    def __init__(self, input_feature_channels,  visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(GRMLayer, self).__init__()

        self.graph_reasoning1 = GraphConvolution(300,256)
        self.graph_reasoning2 = GraphConvolution(256,256)

        self.semantic_to_local = SemanticToLocal(input_feature_channels, visual_feature_channels)
        
        self.graph_adj_mat = torch.FloatTensor(graph_adj_mat).to(device)
        self.visual_feature_channels = visual_feature_channels
        self.fasttest_embeddings = torch.FloatTensor(fasttest_embeddings).to(device)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        #[？，M, H*W]
        visual_feat = x
        
        fasttest_embeddings = self.fasttest_embeddings.unsqueeze(0)
        fasttest_embeddings = fasttest_embeddings.repeat(visual_feat.size(0), 1, 1)
        fasttest_embeddings = fasttest_embeddings.to(visual_feat.cuda())
        
        graph_norm_adj = normalize_adjacency(self.graph_adj_mat)
        
        
        batch_list = []
        for index in range(visual_feat.size(0)):
            batch = self.graph_reasoning1(fasttest_embeddings[index], graph_norm_adj)
            batch = F.relu(batch)
            batch_list.append(batch)
        # [?, M, H*W]
        evolved_feat = torch.stack(batch_list, dim=0)
        #print("#######evolved_feat:",evolved_feat.size())
        batch_list1 = []
        for index in range(evolved_feat.size(0)):
            evolved_feats = F.dropout(evolved_feat[index], 0.3)
            batch1 = self.graph_reasoning2(evolved_feats, graph_norm_adj)
            batch1 = F.relu(batch1)
            batch_list1.append(batch1)
        # [?, M, H*W]
        evolved_feat1 = torch.stack(batch_list1, dim=0)
        
        enhanced_feat = self.semantic_to_local(x, evolved_feat1)

        out = x + enhanced_feat

        return out

if __name__ == "__main__":
   pass