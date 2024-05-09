from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import numpy as np
import ipdb
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import os
from tqdm import tqdm
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.interpolate import Rbf,RBFInterpolator,LinearNDInterpolator
from scipy import optimize
from scipy.io import savemat
from sklearn.cluster import KMeans
import colorsys

def plot_3d_scatter_level(x, y,z,labels,name,z_name):
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
    

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=5).fit(z.reshape(-1,1))
    
    clusters = kmeans.labels_
    cluster_cat = np.unique(clusters)
    color_cat = [0,60/360,120/360,180/360,240/360]
    color_dict = dict(zip(cluster_cat,color_cat))
    
    hsv_colors = [colorsys.hsv_to_rgb(color_dict[c], 1, 1) for c in clusters]
    rgb_colors = [(int(255*r), int(255*g), int(255*b)) for r, g, b in hsv_colors]
    hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in rgb_colors]

    # Create the scatter plot
    fig = go.Figure()
    # Add scatter points

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(color=hex_colors, size=5),
        hovertext=labels,
        showlegend=False
        
    ))

    
    rbf = Rbf(x, y, z, function='multiquadric', smooth=0.02)
    z_pred = rbf(x,y)

    if z_name == 'raw_MIC':
        threshold = 0.2
    if z_name == 'activity':
        threshold = 0.2
    
    # on the surface point
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = rbf(xi,yi)
    # Interpolate the z-values over the grid
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

    # linear interpolate
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
        )
        
    )
    fig.show()
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
    # seperate the outliers
    rbf = Rbf(x, y, z, function='multiquadric', smooth=0.02)
    z_pred = rbf(x,y)

    if z_name == 'raw_MIC':
        threshold = 0.2
    if z_name == 'activity':
        threshold = 0.2
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

    dataset = pd.read_csv('./regression/grampa_v2.csv')
    dataset = dataset.dropna()
    dataset = dataset.loc[(dataset.length <= 50) & (dataset.length >= 5)]
    
    print(f'there are {dataset.shape[0]} sequences to analysis')
    dataset_ref = dataset['SMILES'].values
    ids = dataset['ID'].values
    raw_affinity = dataset['value'].values
    Activity = -np.log10(raw_affinity * 1e-6)
    avail_seq_num = sum(raw_affinity < 1000)
    # labels = dataset['bacterium'].values
    output = 'visualization'
    os.makedirs(output,exist_ok=True)



    # visual_type: 'tanimoto'
    fingerprint = []
    radius = 10  # 指纹半径 previous 10
    n_bits = 256  # 指纹长度
    if not os.path.exists(output+"/simularity.pk"):
        dataset_ref = dataset_ref[:avail_seq_num]
        result = np.zeros((ids.shape[0],ids.shape[0]))
        for id1, cpd1 in tqdm(enumerate(dataset_ref),desc='d1'):
            mol1 = Chem.MolFromSmiles(cpd1)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits, useFeatures=True)
            fingerprint.append(fp1)
            for id2, cpd2 in tqdm(enumerate(dataset_ref),desc='d2'):
                
                
                mol2 = Chem.MolFromSmiles(cpd2)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits, useFeatures=True)
                
                # 计算Tanimoto相似度
                
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                result[id1,id2] = similarity

        pickle.dump(result, open(output+"/simularity.pk", 'wb'))
    else:
        result = pickle.load(open(output+"/simularity.pk", 'rb'))
        dataset_ref = dataset_ref[:avail_seq_num]
        for id1, cpd1 in tqdm(enumerate(dataset_ref),desc='d1'):
            mol1 = Chem.MolFromSmiles(cpd1)
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits, useFeatures=True)
            
            fingerprint.append(fp1)


    result = result[:avail_seq_num,:avail_seq_num]
    # labels = labels[:avail_seq_num]
        # ipdb.set_trace()
    
    decomposition = {
        'tsne': TSNE(n_components=2,random_state=42,perplexity=100,early_exaggeration=5),
        'umap': UMAP(n_components=2,random_state=42,n_neighbors=100,min_dist=0.8)
    }

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=5).fit(fingerprint)
    
    labels = kmeans.labels_
    label_cat = np.unique(labels)

    dataset = dataset.iloc[:avail_seq_num,:]
    dataset['cluster'] = labels
    dataset.to_csv('./regression/grampa_v3.csv')

    for k in decomposition:

        flatten_mapping = decomposition[k].fit_transform(result)

        fig,x,y,z = plot_3d_scatter_level(flatten_mapping[:,0], flatten_mapping[:,1], -raw_affinity[:avail_seq_num],labels,k,'raw_MIC')
        fig.write_html(output + f'/landscape-{k}-raw_MIC.html')
        # pd.DataFrame.from_dict({f'{k}1':x, f'{k}2': y,'Raw Affinity':z},orient='index').T.to_csv(k + '_landscape_raw_affinity.csv')
        
        fig,x,y,z = plot_3d_scatter_level(flatten_mapping[:,0], flatten_mapping[:,1],Activity[:avail_seq_num],labels,k,'activity')
        fig.write_html(output + f'/landscape-{k}-activity.html')
        # pd.DataFrame.from_dict({f'{k}1':x, f'{k}2': y,'Activity':z},orient='index').T.to_csv(k + '_landscape_activity.csv')
        fig,x,y,z = plot_3d_scatter_cluster(flatten_mapping[:,0], flatten_mapping[:,1], -raw_affinity[:avail_seq_num],labels,k,'raw_MIC')
        fig.write_html(output + f'/landscape-cluster-{k}-raw_MIC.html')
        # pd.DataFrame.from_dict({f'{k}1':x, f'{k}2': y,'Raw Affinity':z},orient='index').T.to_csv(k + '_landscape_raw_affinity.csv')
        
        fig,x,y,z = plot_3d_scatter_cluster(flatten_mapping[:,0], flatten_mapping[:,1],Activity[:avail_seq_num],labels,k,'activity')
        fig.write_html(output + f'/landscape-cluster-{k}-activity.html')
        # pd.DataFrame.from_dict({f'{k}1':x, f'{k}2': y,'Activity':z},orient='index').T.to_csv(k + '_landscape_activity.csv')
        
        plt.figure()
        for l in label_cat:
            plt.scatter(flatten_mapping[labels==l,0],flatten_mapping[labels==l,1],label=l)
        # colors = [plt.cm.jet(i / label_cat.shape[0]) for i in label_cat]
        # for feat, label,color in zip(features,labels,colors):
        #     # ipdb.set_trace()
        #     plt.text(feat[0],feat[1],str(label),color=color,fontsize=12,ha='center', va='center')
        plt.legend()
        plt.title(k.upper())
        plt.savefig(output + f'/fingerprint-cluster-{k}.png')
    print('Done')