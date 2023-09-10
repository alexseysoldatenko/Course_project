import torch
import segyio
import numpy as np
import matplotlib.pyplot as plt


def load_segy_file(sgy_path: str, format: tuple):
    with segyio.open(sgy_path,  ignore_geometry = True) as f:
        data = f.trace.raw[:].T
    data = cut_data(data, format)
    return data
    
def cut_data(data: np.ndarray, format: tuple):
    new_shape_rows = data.shape[0] // format[0] * format[0]
    new_shape_columns = data.shape[1] // format[1] * format[1]
    return data[:new_shape_rows,:new_shape_columns]

def get_transform_sgy(data: np.ndarray, model: torch.nn.Module, format: tuple):
    view = np.zeros_like(data)
    for i, j in np.ndindex((data.shape[0]//format[0], data.shape[1]//format[1])):
        low_0 = format[0] * i
        high_0 = format[0] * (i+1)
        low_1 = format[1] * j
        high_1 = format[1] * (j+1)
        sub_data = torch.from_numpy(data[low_0:high_0, low_1:high_1]).reshape((1,1,format[0], format[1]))
        sub_data = (sub_data - sub_data.min())/(sub_data.max() - sub_data.min())
        view[low_0:high_0, low_1:high_1] = torch.argmax(model(sub_data),dim =1)[0].detach().numpy()
    return view

def from_numpy_to_png(data: np.ndarray, format: tuple, folder: str, name: str):
    view = np.zeros_like(data)
    for i, j in np.ndindex((data.shape[0]//format[0], data.shape[1]//format[1])):
        low_0 = format[0] * i
        high_0 = format[0] * (i+1)
        low_1 = format[1] * j
        high_1 = format[1] * (j+1)
        sub_data = data[low_0:high_0, low_1:high_1]
        plt.imsave(f"{folder}\\{name}_{i}_{j}.png", sub_data)

def get_list_of_clusters(data: np.ndarray, step: tuple, model: torch.nn.Module, format: tuple):
    list_of_clusters = []

    for i, j in np.ndindex((int((data.shape[0] - format[0]) / step[0]), int((data.shape[1] - format[1]) / step[1]))):
        low_0 = i *step[0]
        high_0 = format[0] +  i*step[0]
        low_1 =  j * step[1]
        high_1 = format[1] +  j*step[1]
        sub_data = torch.from_numpy(data[low_0:high_0, low_1:high_1]).reshape((1,1,format[0], format[1]))
        sub_data_min = sub_data.min()
        sub_data_max = sub_data.max()
        sub_data = (sub_data - sub_data_min)/(sub_data_max - sub_data_min)
        list_of_clusters.append(model(sub_data)[1].detach().numpy() + model(sub_data)[2].detach().numpy())
        print(len(list_of_clusters))
    return list_of_clusters

def view_clasters(data: np.ndarray, step: tuple, model: torch.nn.Module, format: tuple, n_clusters: int = 3, latent_shape: int = 256):
    list_of_clusters = []
    

    for i, j in np.ndindex((int((data.shape[0] - format[0]) / step[0]), int((data.shape[1] - format[1]) / step[1]))):
        low_0 = i *step[0]
        high_0 = format[0] +  i*step[0]
        low_1 =  j * step[1]
        high_1 = format[1] +  j*step[1]
        sub_data = torch.from_numpy(data[low_0:high_0, low_1:high_1]).reshape((1,1,format[0], format[1]))
        sub_data_min = sub_data.min()
        sub_data_max = sub_data.max()
        sub_data = (sub_data - sub_data_min)/(sub_data_max - sub_data_min)
        list_of_clusters.append(model(sub_data)[1].detach().numpy() + model(sub_data)[2].detach().numpy())
        print(len(list_of_clusters))


    list_of_clusters = np.array(list_of_clusters).reshape(-1, latent_shape)
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters=n_clusters,
            assign_labels='discretize',
            random_state=0).fit(list_of_clusters)
    labels = clustering.labels_

    counter = 0
    canvas = np.zeros_like(data)
    for i, j in np.ndindex((int((data.shape[0] - format[0]) / step[0]), int((data.shape[1] - format[1]) / step[1]))):
        low_0 = i *step[0] + format[0] // 2
        high_0 = (i+1)*step[0] + format[0] // 2
        low_1 =  j * step[1] + format[1] // 2
        high_1 = (j+1)*step[1] + format[1] // 2
        canvas[low_0:high_0, low_1:high_1] = labels[counter]
        counter += 1

    return canvas

def prepare_list_of_names(txt_header: str):
    with open(txt_header, 'r') as f:
        names = f.read().splitlines()
    return names