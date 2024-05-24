import pickle
import json
import numpy as np

def get_voc_data():
    print('obtaining voc data ...')
    with open("Semantic Consistency/Stored matrices/CM_kg_57_info.json","rb") as f:

        info = json.load(f)
        KF_All_VOC_info = info['KG_VOC_info']
    
        graph_adj_mat = np.asarray(KF_All_VOC_info['S'])

        print('the adj mat is\n',graph_adj_mat)
        print('the type is\n', type(graph_adj_mat))
        print('the shape is\n', graph_adj_mat.shape)
        print('nonzero\n', np.count_nonzero(graph_adj_mat))

        num_symbol_node = graph_adj_mat.shape[0]


    with open("graph/voc_glove_word2vec.pkl","rb") as f:
        fasttest_embeddings = pickle.load(f)
        fasttest_dim = fasttest_embeddings.shape[1]

        print('the fasttest_embeddings is\n',fasttest_embeddings)
        print('the type is\n', type(fasttest_embeddings))
        print('the shape is\n', fasttest_embeddings.shape)
        print('nonzero\n', np.count_nonzero(fasttest_embeddings))

    print('obtained voc data')

    return {"num_symbol_node":num_symbol_node,
            "fasttest_embeddings":fasttest_embeddings,
             "fasttest_dim": fasttest_dim,
            "graph_adj_mat": graph_adj_mat
            }



