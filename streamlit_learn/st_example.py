import streamlit as st
import numpy as np
import pandas as pd

# @st.cache_data
# def load_data(datafile_dir):
#     df_tri = pd.read_csv('benchmarks/' + datafile_dir + '/test2id.txt', sep = ' ', header = None, names = ['h', 't', 'r'], skiprows = 1)
#     df_ent = pd.read_csv('benchmarks/' + datafile_dir + '/entity2id.txt', sep = '\t', header = None, names = ['ent', 'no'], skiprows = 1)
#     df_rel = pd.read_csv('benchmarks/' + datafile_dir + '/relation2id.txt', sep = '\t', header = None, names = ['rel', 'no'], skiprows = 1)
#     head_sets = df_ent[df_ent['no'].isin(df_tri['h'].tolist())]
#     tail_sets = df_ent[df_ent['no'].isin(df_tri['t'].tolist())]
#     rel_sets = df_rel[df_rel['no'].isin(df_tri['r'].tolist())]
#     return df_tri, df_ent, df_rel, head_sets, tail_sets, rel_sets

# @st.cache_data
# def search(src, src_type, dst_type):
#     allowed_type = ['h', 't', 'r']
#     if src_type not in allowed_type or dst_type not in allowed_type or src_type == dst_type:
#         raise ValueError(f"type must be in {allowed_type} and 2 types can not be the same.")
#     if src_type == 'h':
#         no_h = head_sets.loc[head_sets['ent'] == src, 'no'].iloc[0]
#         if dst_type == 't':
#             no_t = df_tri.loc[df_tri['h'] == no_h, 't'].tolist()
#             return tail_sets[tail_sets['no'].isin(no_t)]
#         else:
#             no_r = df_tri.loc[df_tri['h'] == no_h, 'r'].tolist()
#             return rel_sets[rel_sets['no'].isin(no_r)]
#     elif src_type == 't':
#         no_t = tail_sets.loc[tail_sets['ent'] == src, 'no'].iloc[0]
#         if dst_type == 'h':
#             no_h = df_tri.loc[df_tri['t'] == no_t, 'h'].tolist()
#             return head_sets[head_sets['no'].isin(no_h)]
#         else:
#             no_r = df_tri.loc[df_tri['t'] == no_t, 'r'].tolist()
#             return rel_sets[rel_sets['no'].isin(no_r)]
#     elif src_type == 'r':
#         no_r = rel_sets.loc[rel_sets['rel'] == src, 'no'].iloc[0]
#         if dst_type == 'h':
#             no_h = df_tri.loc[df_tri['r'] == no_r, 'h'].tolist()
#             return head_sets[head_sets['no'].isin(no_h)]
#         else:
#             no_t = df_tri.loc[df_tri['r'] == no_r, 't'].tolist()
#             return tail_sets[tail_sets['no'].isin(no_t)]

st.set_page_config(
    page_title='Knowledge Graph Embedding Model',
    page_icon='🧊',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('基于知识图谱嵌入模型的知识图谱补全任务')

st.divider()

# 选择模型、后端、数据集、三元组缺失项
op1, op2, op3, op4, op5 = st.columns(5)

with op1:
    model = st.selectbox(
        '知识图谱嵌入模型:',
        ('TransE', 'TransR', 'RESCAL', 'RotatE')
    )

with op2:
    backend = st.selectbox(
        'TensorLayerX后端:',
        ('TensorFlow', 'Torch', 'Paddle', 'MindSpore')
    )

with op3:
    datafile_dir = st.selectbox(
        '训练和测试数据集:',
        ('FB15K237', 'WN18RR')
    )

with op4:
    triplet = ['Head Entity', 'Relation', 'Tail Entity']
    predict = st.selectbox(
        '三元组缺失项:',
        triplet
    )
    st.write(predict)

with op5:
    st.write(predict)
    st.write(triplet)
    # triplet.remove(predict)
    st.write(triplet)
    key_col = st.selectbox(
        '关键列/索引列:',
        triplet
    )

# st.write(triplet)