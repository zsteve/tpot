import plotly 
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
import itertools
import scipy as sp
import sklearn as sk

import sys
sys.path.append("../tools/partitioned_networks/tools/HyperCOT")
import hypercot
from hypercot import get_hgraph_dual, convert_to_line_graph, get_v, get_omega

marker_size = 10
line_width = 3

def plot_2d(fig, points,color,legend,show_legend = True, row = 1, col =1, **kwargs):
    
    fig.add_trace(go.Scatter(x = points[:,0],
                             y = points[:,1],
                             mode = 'markers',
                             marker = dict(color = color,
                                                   cmax=1,
                                                   cmin=-1,
                                           size = marker_size,
                                           line=dict(width=0.8, 
                                                    color='DarkSlateGrey')
                                           ),
                             name = legend,
                             showlegend= show_legend,
                             legendgroup= legend,
                            
                            ),
                  row = row,
                  col = col,
                  **kwargs
                 )
    return fig


def plot_3d(fig, points,color,legend,show_legend = False, **kwargs):
    # fig = go.Figure()
    fig.add_trace(go.Scatter3d(x = points[:,0],
                             y = points[:,1],
                             z = points[:,2],
                             mode = 'markers',
                             marker = dict(color = color,
                                           cmax=1,
                                           cmin=-1,
                                           size = marker_size,
                                           line=dict(width=0.8, 
                                                    color='DarkSlateGrey')
                  ),
                             name = legend,
                             showlegend= show_legend,
                               legendgroup= legend
                            ),
                  **kwargs
                 )
    return fig

def make_dataframe(barcode):
    births, deaths = np.array(barcode[0]),np.array(barcode[1])
    lengths = deaths - births
    df = pd.DataFrame({"Birth": births, "Death": deaths, "Length": lengths})
    return df

def plot_barcode(barcode,title):
    df = make_dataframe(barcode)
    return px.bar(data_frame = df, x='Length', orientation='h', base='Birth', 
                  hover_data={'Birth': True, 'Death': True,'Length': False},
                 title = title) 


def plot_representative(fig,pointcloud,repre,color,**kwargs):
    
    points = np.array([pointcloud[el-1] for el in repre])
    if pointcloud.shape[1] == 2:
        
        fig.add_trace(go.Scatter(x = points[:,0],
                                 y = points[:,1],
                                 mode = 'markers',
                                 marker = dict(color = color,
                                            size = marker_size+2),
                                 name= 'PH representative',
                                 **kwargs
        )
        )
    if pointcloud.shape[1] == 3:
        
        fig.add_trace(go.Scatter3d(x = points[:,0],
                                 y = points[:,1],
                                 z = points[:,2],
                                 mode = 'markers',
                                 marker = dict(color = color,
                                            size = marker_size+2),
                                 name= 'PH representative',
                                 **kwargs
        )
        )        
        
    return fig

def create_graph_from_PH(barcodes, representatives, nPoints):
    G = nx.Graph()
    G.add_nodes_from(range(nPoints))
    for b, r in zip(np.array(barcodes).T, representatives):
        persistence = abs(b[1] - b[0])

        # add edge for all unique pairs of nodes in this representative
        print(r)
        for k in range(len(r)):
            for j in range(k+1, len(r)):
                # -1 due to zero indexing
                # G.add_edge([x-1 for x in r[k]], [x-1 for x in r[j]], weight=persistence)
                G.add_edge(r[k]-1, r[j]-1, weight=persistence)
    return G


def img_to_pointcloud(X, sigma = 0, size = 500):
    x = X / X.sum()
    _1d_to_2d = list(itertools.product(np.arange(x.shape[0]) / x.shape[0], np.arange(x.shape[1]) / x.shape[1]))
    X_sample = np.array([list(_1d_to_2d[i]) for i in np.random.choice(len(_1d_to_2d), p = np.array(x).flatten(), size = size, replace = False)])
    return X_sample + np.random.randn(*X_sample.shape)*sigma

def get_kernel(X, h, C = None, kernel_type = "gaussian"):
    C = sp.spatial.distance.cdist(X, X, metric = "sqeuclidean") if C is None else C
    bw = h*np.median(np.sqrt(C))
    if kernel_type == "gaussian":
        K = np.exp(-C / bw**2)
    elif kernel_type == "epanech":
        K = np.maximum(1 - C/bw**2, 0)
    elif kernel_type == "flat":
        K = (C < bw**2)*1.0
    else:
        K = None
    return K

def get_eigvec(X, h = 1.0):
    K = get_kernel(X, h)
    _, v = np.linalg.eig(K)
    return np.real(v[:, 1])

def symm(A):
    return (A + A.T)/2

def knn_adj(x, k = 10):
    return sk.neighbors.NearestNeighbors(n_neighbors = k).fit(x).kneighbors_graph()

def knn_cost(x, k = 10):
    A = knn_adj(x, k = k)
    return sp.sparse.csgraph.floyd_warshall(A)**2

# similarity with kernel
def symmetric_laplacian(K):
    d = K.sum(-1)
    A_norm = (K * (d**(-0.5)).reshape(-1, 1) * (d**(-0.5)).reshape(1, -1))
    np.fill_diagonal(A_norm, 0)
    return np.eye(K.shape[0]) - (K * (d**(-0.5)).reshape(-1, 1) * (d**(-0.5)).reshape(1, -1))

def rw_transitions(K):
    d = K.sum(-1)
    return (1/d).reshape(-1, 1)*K

def impute_laplacian(y, L, lamda = 1.0):
    A = np.eye(L.shape[0]) + lamda*L
    return np.linalg.solve(A, y)

# mapper

def heat_kernel(G,t=5):
    L = nx.normalized_laplacian_matrix(G)
    lam,phi = np.linalg.eigh(L.A)
    K = phi @ np.diag(np.exp(-t*lam)) @ phi.T
    return K

def get_nerve(G,K,v):
    """
    Get iterated nerve graph
    
    Parameters:
    G : nx.Graph
    K : heat kernel
    v : (len(G),3) vertex coordinates
    
    Returns:
    
    subG : nx.Graph (nerve graph)
    subv : (len(subG),3) nerve graph vertex coordinates
    memberMat : cover elements
    
    """       
    # Get nerve
    unCov = set(G.nodes)
    memberMat = []
    pts = []
    n = len(G)

    while len(unCov):
        pt = next(iter(unCov)) 
        pts.append(pt)
        h = np.zeros(n)
        h[pt] = 1
        h_ = K @ h
        halfMax = h_.max()/2
        halfVec = h_>= halfMax/2
        memberMat.append(sp.sparse.coo_matrix(halfVec).astype(int))
        halfN = np.where(h_ >= halfMax)[0] #half-max neighbors
        unCov -= set(halfN)

    memberMat = sp.sparse.vstack(memberMat).tocsr()
    # Restrict to new graph
    subG = nx.Graph(memberMat @ memberMat.T) # Nerve graph
    subK = K[pts][:,pts]
    subv = v if v is None else v[pts]
    return subG, subv, memberMat

def process_hg(h, M):
    h_dual = get_hgraph_dual(h)
    l = convert_to_line_graph(h.incidence_dict)
    v = get_v(h.incidence_dict, h_dual.incidence_dict)
    w = incidence(h, M, h.shape[1])
    return w, np.full((w.shape[0], ), 1/w.shape[0]), v

def incidence(g, M, N):
    A = np.zeros((M, N))
    for e in list(g.edges()):
        A[:, int(e.uid)][np.array(list(e.elements))-1] = 1
    return A

# helper functions for generating toy data
def noisy_circle(n_samples, noise_level, center_x, center_y ,radius):
    t = np.linspace(0,2*np.pi,n_samples)
    x = center_x + radius*np.cos(t)
    y = center_y + radius*np.sin(t)
    noise = np.random.rand(n_samples,2)
    data = np.array([x,y]).T + noise_level*radius*noise
    return data

def noisy_ellipses(n_samples, noise_level, radius1, radius2, centre_x, centre_y):
    t = np.linspace(0,2*np.pi,n_samples)
    x = radius1*np.cos(t) 
    y = radius2*np.sin(t) 
    data = np.array([x,y]).T
    data = [el +  np.random.uniform(0,noise_level) for el in data]
    data = [el + [centre_x,centre_y] for el in data]
    data = np.array(data)
    return data

def noisy_disk(n_samples,noise_level,center_x,center_y,radius):
    radius = radius * radius
    t = np.linspace(0,2*np.pi,n_samples)
    r = np.sqrt(radius*np.random.rand(n_samples))
    x = center_x + np.multiply(r,np.cos(t))
    y = center_y + np.multiply(r,np.sin(t))
    noise = np.random.rand(n_samples,2)
    data = np.array([x,y]).T + noise_level*radius*noise
    return data
