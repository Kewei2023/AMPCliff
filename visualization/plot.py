# maintained by kewei li
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
from tqdm import tqdm
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.interpolate import Rbf,RBFInterpolator,LinearNDInterpolator
from scipy import optimize
from scipy.io import savemat
from sklearn.cluster import KMeans
import colorsys
import math

def plot_low_dimension(result, labels=None, savedir=None, alpha=1.0):
    
    os.makedirs(savedir, exist_ok=True)
    if labels is not None:
        label_cat = np.unique(labels)
    decomposition = {
        'tsne': TSNE(n_components=2, random_state=42,perplexity=40,n_iter=500),
        'umap': UMAP(n_components=2, random_state=42)
    }
    
    label_shape_map = {
          'train': 'o',  
          'valid': 'o',  # square
          'test': '^'  # triangle
      }
    
    hidden_states = {}
    for k in decomposition:
        plt.figure()
        flatten_mapping = decomposition[k].fit_transform(result)
        hidden_states[k] = flatten_mapping
        if labels is not None:
            for l in label_cat:
                marker = label_shape_map.get(l, 'o') 
                plt.scatter(flatten_mapping[labels == l, 0], flatten_mapping[labels == l, 1], label=l, alpha=alpha, marker=marker)
        else:
            plt.scatter(flatten_mapping[:, 0], flatten_mapping[:, 1], alpha=alpha)
        plt.legend()
        plt.title(k.upper())
        plt.savefig(f'./{savedir}/LM-{k}-opt_{alpha}.png')
    
    return hidden_states


def plot_3d_scatter_level(x, y,z, fill,name,z_name): # labels,name,z_name):
    """
    Create a 3D scatter plot using Plotly.

    Parameters:
    - x (np.ndarray): The x-coordinates of the points.
    - y (np.ndarray): The y-coordinates of the points.
    - z (np.ndarray): The z-coordinates of the points.

    Returns:
    None: Displays the interactive 3D scatter plot.
    """
    # 使用Z值的范围为点着色
    # norm = np.linalg.norm(z)
    # normalized_z = z / norm

    # 使用HSV colormap将每个值转换为颜色
    
    to_save = {}
    '''
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=5).fit(z.reshape(-1,1))
    
    clusters = kmeans.labels_
    cluster_cat = np.unique(clusters)
    color_cat = [0,60/360,120/360,180/360,240/360]
    color_dict = dict(zip(cluster_cat,color_cat))
    
    hsv_colors = [colorsys.hsv_to_rgb(color_dict[c], 1, 1) for c in clusters]
    rgb_colors = [(int(255*r), int(255*g), int(255*b)) for r, g, b in hsv_colors]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in rgb_colors]
    '''
    # Create the scatter plot
    fig = go.Figure()
    

    
    rbf = Rbf(x, y, z, function='multiquadric', smooth=0.02)
    z_pred = rbf(x,y)
    
    to_save['z_pred']=z_pred
    
    if z_name == 'raw_MIC':
        threshold = 0.2
    if z_name == 'activity':
        threshold = np.log10(5)
    
    # on the surface point
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = rbf(xi,yi)
    
    
    to_save['x_grid']=xi
    to_save['y_grid']=yi
    to_save['z_grid']=zi
    
    color_scale_min = min(2,math.floor(zi.min()))
    color_scale_max = max(9,math.ceil(zi.max()))
    # Add scatter points
    if not fill:
      fig.add_trace(go.Scatter3d(
          x=x,
          y=y,
          z=z,
          mode="markers",
          marker=dict(
                  size=2,
                  color=z,  # 设置颜色为z值
                  colorscale='Jet',  # 选择一个预设的颜色映射
                  colorbar=dict(title='Colorbar'),  # 显示颜色条
                  cmin=color_scale_min,  # 设置颜色尺度最小值
                  cmax=color_scale_max,   # 设置颜色尺度最大值
                  opacity=0.8
                  ),
          # hovertext=labels,
          showlegend=False
          
      ))
    # Interpolate the z-values over the grid
    for dim in range(xi.shape[0]):
        fig.add_trace(go.Scatter3d(z=zi[dim], x=xi[dim], y=yi[dim],mode="lines",
                                   line=dict(color='grey'),   
                                   showlegend=False))
                                   
                                   
    if fill:
      fig.add_trace(go.Surface(z=zi, x=xi, y=yi,colorscale='Jet',cmin=color_scale_min,cmax=color_scale_max))
    xii = xi.T
    yii = yi.T
    zii = zi.T
    
    to_save['x_grid_T']=xii
    to_save['y_grid_T']=yii
    to_save['z_grid_T']=zii
    
    for dim in range(xii.shape[0]):
        fig.add_trace(go.Scatter3d(z=zii[dim], x=xii[dim], y=yii[dim],mode="lines",
                                   line=dict(color='grey'),   
                                   showlegend=False))
    if fill:
      fig.add_trace(go.Surface(z=zii, x=xii, y=yii,colorscale='Jet',cmin=color_scale_min,cmax=color_scale_max))                        
    '''
    # add out grid
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))

    # Y轴
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))

    # Z轴
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    '''
    # linear interpolate
    off_surface_indices = np.where(np.abs(z - z_pred) >= threshold)[0]
    on_surface_indices = np.where(np.abs(z - z_pred) < threshold)[0]
    
    to_save['off_surface']=np.abs(z - z_pred) >= threshold
    
    # draw the outliers
    surfaces = []

    x_surface, y_surface,z_surface = xi.flatten(), yi.flatten(), zi.flatten()
    for idx in off_surface_indices:
        distances = np.sqrt((x_surface - x[idx])**2 + (y_surface - y[idx])**2)
        nearest_indices = distances.argsort()[1:5]  # 从1开始，因为距离为0的点是点本身
        x_neigh = np.append(x_surface[nearest_indices], x[idx])
        y_neigh = np.append(y_surface[nearest_indices], y[idx])
        z_neigh = np.append(z_surface[nearest_indices], z[idx])
        try:

            xi_neigh = np.linspace(min(x_neigh), max(x_neigh), 5)
            yi_neigh = np.linspace(min(y_neigh), max(y_neigh), 5)
            xi_neigh, yi_neigh = np.meshgrid(xi_neigh, yi_neigh)
            z_rbf_interpolated = griddata((x_neigh,y_neigh),z_neigh, (xi_neigh, yi_neigh), method='linear')
            surfaces.append((xi_neigh, yi_neigh, z_rbf_interpolated))
        except:
            x_neigh += np.random.uniform(-1e-4, 1e-4, x_neigh.shape[0])
            y_neigh += np.random.uniform(-1e-4, 1e-4, y_neigh.shape[0])
            xi_neigh = np.linspace(min(x_neigh), max(x_neigh), 5)
            yi_neigh = np.linspace(min(y_neigh), max(y_neigh), 5)
            xi_neigh, yi_neigh = np.meshgrid(xi_neigh, yi_neigh)
            z_rbf_interpolated = griddata((x_neigh,y_neigh),z_neigh, (xi_neigh, yi_neigh), method='linear')
            surfaces.append((xi_neigh, yi_neigh, z_rbf_interpolated))

    for surface in surfaces:
        xi, yi, zi = surface
        for dim in range(xi.shape[0]):
            fig.add_trace(go.Scatter3d(z=zi[dim], x=xi[dim], y=yi[dim],mode="lines",
                                    line=dict(color='grey'),   
                                    showlegend=False))
        if fill:
          fig.add_trace(go.Surface(z=zi, x=xi, y=yi,colorscale='Jet',cmin=color_scale_min,cmax=color_scale_max))
        xii = xi.T
        yii = yi.T
        zii = zi.T
        for dim in range(xii.shape[0]):
            fig.add_trace(go.Scatter3d(z=zii[dim], x=xii[dim], y=yii[dim],mode="lines",
                                    line=dict(color='grey'),   
                                    showlegend=False))
        if fill:
            fig.add_trace(go.Surface(z=zii, x=xii, y=yii,colorscale='Jet',cmin=color_scale_min,cmax=color_scale_max))
    
    # '''
    # Layout adjustments
    fig.update_layout(
        title=dict(text=f"3D {name.upper()} {z_name} Surface", font=dict(size=24)),
        scene=dict(
            bgcolor='rgba(255,255,255,0)',
            # xaxis=dict(title=f"{name}1",title_font=dict(size=24),backgroundcolor='white',showgrid=False,showline=True, showbackground=False,tickfont=dict(size=20)),
            # yaxis=dict(title=f"{name}2",title_font=dict(size=24),backgroundcolor='white',showgrid=False,showline=True, showbackground=False,tickfont=dict(size=20)),
            # zaxis=dict(title=z_name,title_font=dict(size=24),backgroundcolor='white',showgrid=False,showline=True, showbackground=False,tickfont=dict(size=20)),
            
            xaxis=dict(showgrid=False,showline=False, showbackground=False,zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False,showline=False, showbackground=False,zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False,showline=False, showbackground=False,zeroline=False, showticklabels=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4)
        )
        
    )
    # fig.show()
    return fig,xi,yi,zi



def plot_3d_scatter_cluster(x, y,z,labels,name,z_name):
    """
    Create a 3D scatter plot using Plotly.

    Parameters:
    - x (np.ndarray): The x-coordinates of the points.
    - y (np.ndarray): The y-coordinates of the points.
    - z (np.ndarray): The z-coordinates of the points.

    Returns:
    None: Displays the interactive 3D scatter plot.
    """
    to_save = {}
    # seperate the outliers
    rbf = Rbf(x, y, z, function='multiquadric', smooth=0.02)
    z_pred = rbf(x,y)
    
    to_save['z_pred']=z_pred
    
    if z_name == 'raw_MIC':
        threshold = 0.2
    if z_name == 'activity':
        threshold = np.log10(5)
        
    label_cat = np.unique(labels)
    # Create the scatter plot
    fig = go.Figure()
    # Add scatter points
    for l in label_cat:
        fig.add_trace(go.Scatter3d(
            x=x[labels==l],
            y=y[labels==l],
            z=z[labels==l],
            mode="markers",
            marker=dict(size=4),
            name=f'cluster {l}',
            hovertext=f'cluster {l}'
        ))

    # on the surface point
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = rbf(xi,yi)
    
    to_save['x_grid']=xi
    to_save['y_grid']=yi
    to_save['z_grid']=zi
    
    for dim in range(xi.shape[0]):
        fig.add_trace(go.Scatter3d(z=zi[dim], x=xi[dim], y=yi[dim],mode="lines",
                                   line=dict(color='grey'),   
                                   showlegend=False))
    xii = xi.T
    yii = yi.T
    zii = zi.T
    
    to_save['x_grid_T']=xii
    to_save['y_grid_T']=yii
    to_save['z_grid_T']=zii
    
    for dim in range(xii.shape[0]):
        fig.add_trace(go.Scatter3d(z=zii[dim], x=xii[dim], y=yii[dim],mode="lines",
                                   line=dict(color='grey'),   
                                   showlegend=False))
                            
     # add out grid
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), max(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))

    # Y轴
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), min(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), max(yi.flatten())], z=[max(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))

    # Z轴
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[min(yi.flatten()), min(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[min(xi.flatten()), min(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[max(xi.flatten()), max(xi.flatten())], y=[max(yi.flatten()), max(yi.flatten())], z=[min(zi.flatten()), max(zi.flatten())], mode='lines', line=dict(color='black'), showlegend=False))


    off_surface_indices = np.where(np.abs(z - z_pred) > threshold)[0]
    on_surface_indices = np.where(np.abs(z - z_pred) <= threshold)[0]
    
    
    # draw the outliers
    surfaces = []

    x_surface, y_surface,z_surface = xi.flatten(), yi.flatten(), zi.flatten()
    for idx in off_surface_indices:
        distances = np.sqrt((x_surface - x[idx])**2 + (y_surface - y[idx])**2)
        nearest_indices = distances.argsort()[1:5]  # 从1开始，因为距离为0的点是点本身
        x_neigh = np.append(x_surface[nearest_indices], x[idx])
        y_neigh = np.append(y_surface[nearest_indices], y[idx])
        z_neigh = np.append(z_surface[nearest_indices], z[idx])
        try:

            # rbf_interpolator = Rbf(x_neigh, y_neigh, z_neigh, function='thin_plate')
            
            xi_neigh = np.linspace(min(x_neigh), max(x_neigh), 5)
            yi_neigh = np.linspace(min(y_neigh), max(y_neigh), 5)
            xi_neigh, yi_neigh = np.meshgrid(xi_neigh, yi_neigh)

            # z_rbf_interpolated = rbf_interpolator(xi_neigh, yi_neigh)
            z_rbf_interpolated = griddata((x_neigh,y_neigh),z_neigh, (xi_neigh, yi_neigh), method='linear')
            surfaces.append((xi_neigh, yi_neigh, z_rbf_interpolated))
        except:
            x_neigh += np.random.uniform(-1e-4, 1e-4, x_neigh.shape[0])
            y_neigh += np.random.uniform(-1e-4, 1e-4, y_neigh.shape[0])

            # rbf_interpolator = Rbf(x_neigh, y_neigh, z_neigh, function='thin_plate')

            xi_neigh = np.linspace(min(x_neigh), max(x_neigh), 5)
            yi_neigh = np.linspace(min(y_neigh), max(y_neigh), 5)
            xi_neigh, yi_neigh = np.meshgrid(xi_neigh, yi_neigh)

            # z_rbf_interpolated = rbf_interpolator(xi_neigh, yi_neigh)
            z_rbf_interpolated = griddata((x_neigh,y_neigh),z_neigh, (xi_neigh, yi_neigh), method='linear')
            surfaces.append((xi_neigh, yi_neigh, z_rbf_interpolated))

    for surface in surfaces:
        xi, yi, zi = surface
        for dim in range(xi.shape[0]):
            fig.add_trace(go.Scatter3d(z=zi[dim], x=xi[dim], y=yi[dim],mode="lines",
                                    line=dict(color='grey'),   
                                    showlegend=False))
        xii = xi.T
        yii = yi.T
        zii = zi.T
        for dim in range(xii.shape[0]):
            fig.add_trace(go.Scatter3d(z=zii[dim], x=xii[dim], y=yii[dim],mode="lines",
                                    line=dict(color='grey'),   
                                    showlegend=False))
        

    # '''
    # Layout adjustments
    fig.update_layout(
        title=dict(text=f"3D {name.upper()} {z_name} Surface", font=dict(size=24)),
        scene=dict(
            bgcolor='rgba(255,255,255,0)',
            xaxis=dict(title=f"{name}1",title_font=dict(size=24),backgroundcolor='white',showgrid=False,showline=True, showbackground=False,tickfont=dict(size=20)),
            yaxis=dict(title=f"{name}2",title_font=dict(size=24),backgroundcolor='white',showgrid=False,showline=True, showbackground=False,tickfont=dict(size=20)),
            zaxis=dict(title=z_name,title_font=dict(size=24),backgroundcolor='white',showgrid=False,showline=True, showbackground=False,tickfont=dict(size=20)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4)
            # margin=dict(l=0, r=0, b=0, t=0)
        )
        
    )
    fig.show()
    return fig,xi,yi,zi
    
    
    
if __name__=='__main__':
  
  fig,x,y,z = plot_3d_scatter_level(flatten_mapping[:,0], flatten_mapping[:,1],Activity[:avail_seq_num],labels,k,'activity')