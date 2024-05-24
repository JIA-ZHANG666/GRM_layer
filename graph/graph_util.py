import torch
import pickle
from .global_settings import GPU_ID

cuda_suffix = 'cuda:' + str(GPU_ID) if len(str(GPU_ID)) == 2 else "cuda:0"
#device = torch.device("cuda")

def row_normalize(adj_mat):
    Q = torch.sum(adj_mat, axis=-1).float()
    sQ = torch.rsqrt(Q)
    sQ = torch.diag(sQ)
    return torch.mm(sQ, torch.mm(adj_mat, sQ))

# 输入是tensor
def normalize_adjacency(adj_mat):
    assert adj_mat.shape[0] == adj_mat.shape[1]
    #print('cuda suffix:', cuda_suffix)
    adj_mat += torch.eye(adj_mat.shape[0]).cuda(cuda_suffix)
    adj_mat = adj_mat.float()
    norm_adj_mat = row_normalize(adj_mat)
    return norm_adj_mat

def get_pickle_data(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)
    return data





