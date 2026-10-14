import argparse
import yaml
import os.path as osp
import errno
import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime 

from .logger import *


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_result_dir(conf):
    result_dir = osp.join(
        conf.result_dir,
        osp.splitext(osp.basename(conf.config))[0],
        conf.name.replace(':', '-'),
        'train-data_split_seed_' + str(conf.seed)
    )
    mkdir_if_missing(result_dir)
    return result_dir


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))


def set_config(config, cfg, save_result_dir):
    config['path_split'] = config['dataset_root_dir'] + config['path_split']
    config['path_feat'] = config['dataset_root_dir'] + config['path_feat']
    config['path_table'] = config['dataset_root_dir'] + config['path_table']
    config['config_file'] = cfg['config']
    config['seed'] = cfg['seed']
    config['name'] = cfg['name']
    config['data_split_seed'] = cfg['seed']
    config['save_result_dir'] = save_result_dir


def init(conf):
    save_result_dir = create_result_dir(conf)
    setup_logger(save_result_dir)
    set_random_seed(conf.seed)
    set_config(conf, conf, save_result_dir)
    torch.set_printoptions(sci_mode=False, precision=4)


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in config:
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def get_device(gpu_id):
    return torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def get_loss_function(loss_function_name):
    if loss_function_name == 'cross_entropy':
        return F.cross_entropy
    else:
        raise NotImplementedError("Please specify a valid loss function.")


def get_init_key_frequency(config):
    key_frequency = {}
    for dataset_name in config['dataset_names']:
        key_frequency[dataset_name] = []
    return key_frequency


def get_current_ensemble_classes(config, seen_tasks):
    class_ensemble_path = config['class_ensemble_path']
    with open(class_ensemble_path) as f:
        prompts = json.load(f)['0']
    classnames = prompts['classnames']
    templates = prompts['templates']

    current_ensemble_classes = {'ensemble_classes': [], 'count': []}
    for k, v in classnames.items():
        for key, val in v.items():
            if k in seen_tasks:
                current_ensemble_classes['count'].append(len(val) * len(templates))
                for name in val:
                    current_ensemble_classes['ensemble_classes'].extend([template.replace('CLASSNAME', name) for template in templates])

    return current_ensemble_classes


def rename_keys(d, prefix_name, sep='/'):
    newd = dict()
    for k, v in d.items():
        newd[prefix_name + sep + k] = v
    return newd


def check_feature_distribution(feature, label, save_dir, type='image', epoch=0):
    if type == 'text':
        if feature.dim() == 3: # text_feature: [N, num_cls, C]
            label = torch.arange(feature.shape[1]).unsqueeze(0).repeat(feature.shape[0], 1).view(-1)
            feature = feature.view(-1, feature.shape[-1])
        else: # text_feature: [num_cls, C]
            label = torch.arange(feature.shape[0])

    print('[epoch {}] check {} feature distribution'.format(epoch, type))

    # NaN
    has_nan = torch.isnan(feature).any()
    if has_nan:
        print("[error] Contains NaN")

    # Inf
    has_inf = torch.isinf(feature).any()
    if has_inf:
        print("[error] Contains Inf")

    # tSNE embeddings
    tsne = TSNE(perplexity=feature.shape[0]-1 if feature.shape[0] <= 30 else 30)
    X_embedded = tsne.fit_transform(feature.numpy())

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    # Define color palette
    NUM_CLUSTER = torch.max(label).item() + 1
    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    palette = sns.color_palette("bright", NUM_CLUSTER)

    # Define custom color mapping for labels
    label_color_mapping = {}
    for i in range(NUM_CLUSTER):
        label_color_mapping[i] = palette[i]

    # Plotting
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=label.numpy(), legend=False,
                    palette=label_color_mapping, ax=ax, s=3, edgecolors='none', linewidths=0)


    if epoch > 0:
        plt.savefig(osp.join(save_dir, '{}_{}.png'.format(type, epoch)), bbox_inches='tight')
        wandb.log({type+'_feature': wandb.Image(plt)})
    else:
        plt.savefig(osp.join(save_dir, '{}_final.png'.format(type)), bbox_inches='tight')
        wandb.log({type+'_feature_final': wandb.Image(plt)})
    plt.close()

def plot_confusion_matrix(cm, save_dir, name='confusion_matrix', idx=0):
    print("[Confusion Matrix] Computing final matrices...")
    results = cm.compute()
    save_matrix(results, osp.join(save_dir, f"{name}_t{idx}.txt"))
    fig, ax = cm.plot(val=results) # Use the metric's plot method
    # You might want more descriptive labels based on seen tasks later
    ax.set_title(f"{name} Task {idx}")
    cm_path = osp.join(save_dir, f"{name}_t{idx}.jpg")
    plt.savefig(cm_path)
    wandb.log({f"{name} task {idx}": wandb.Image(cm_path)})
    plt.close(fig) # Close figure to free memory


def plot_key_matching_heatmap(value, path, dataset='train'):
    value = torch.stack(value, dim=0).numpy()
    plt.figure(dpi=500)
    plt.imshow(value, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(osp.join(path, dataset+'_heatmap.png'))
    wandb.log({dataset+'/heatmap': wandb.Image(plt)})
    plt.close()


def get_current_eval_dataloader(data_loader_dict, seen_tasks:list):
    current_eval_dataloader = {}

    for key, val in data_loader_dict.items():
        if key in seen_tasks:
            current_eval_dataloader[key + '/val'] = val['val']
            current_eval_dataloader[key + '/test'] = val['test']

    return current_eval_dataloader


import scipy
def get_null_space(X, a=None):
    """
    Computes the (approximate) null space of a matrix X using SVD.

    Args:
        X (torch.Tensor): The input matrix (2D tensor, M x N) for which to find the
                          (approximate) null space.
        a (float, optional): Multiplier for the minimum singular value threshold.
            If specified (> 0), computes the approximate null space where singular
            values s satisfy s <= a * s_min.
            If None, uses a standard tolerance based on the largest singular value
            and machine epsilon to identify near-zero singular values.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor whose columns form an orthonormal basis for the
                      computed (approximate) null space. Shape is (N, k),
                      where N = X.shape[1] and k is the dimension of the null space.
                      Returns an empty tensor with shape (N, 0) if the
                      null space is trivial or no singular values meet the criteria.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D PyTorch Tensor.")

    device = X.device
    dtype = X.dtype

    U, S, Vh = torch.linalg.svd(X, full_matrices=True)
    M, N = U.shape[0], Vh.shape[1]
    tol = max(torch.amax(S).item(), 0.) * torch.finfo(S.dtype).eps * max(M, N)
    lambda_min = torch.min(S)
    if lambda_min < tol or a is None:
        indices = torch.where(S <= tol)[0]
    else:
        thredold = min(a * lambda_min, S[-20])
        indices = torch.where(S <= thredold)[0]

    if indices.numel() == 0:
        # No singular values meet the criteria, null space is empty (or only has zero vector)
        null_space_basis = torch.empty((X.shape[1], 0), dtype=dtype, device=device)
    else:
        # Select the corresponding right singular vectors (columns of V)
        # These vectors v satisfy ||Xv|| = ||U S Vh v|| = ||U S e_i|| = S_i which is small
        null_space_basis = Vh[indices, :].t()  # Transpose to get shape (N, k)
    return null_space_basis

def save_matrix(data, filename):
    """
    Saves a matrix (NumPy array or PyTorch Tensor) to a file (TXT or CSV).
    Args:
        data: The matrix data to save. Can be a NumPy array or a PyTorch Tensor.
        filename (str): The full path and filename for the output file.
                        Must end with '.txt' or '.csv'.
    """
    # 1. Check input data type and convert if necessary
    if isinstance(data, np.ndarray):
        data_np = data
    elif isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        raise TypeError("Input data must be a NumPy array or a PyTorch Tensor.")

    # 2. Determine file format and delimiter from filename extension
    _, ext = os.path.splitext(filename)
    ext = ext.lower()  # Make extension check case-insensitive

    if ext == '.csv':
        delimiter = ','
        file_format = 'CSV'
    elif ext == '.txt':
        delimiter = '\t'  # Using tab for better readability in TXT
        file_format = 'TXT'
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. "
                         "Filename must end with '.txt' or '.csv'.")

    # 3. Determine data type for saving format ('fmt')
    dtype = data_np.dtype
    if np.issubdtype(dtype, np.integer):
        # Data contains integers
        fmt = '%d'
        dtype_description = "integer"
    elif np.issubdtype(dtype, np.floating):
        # Data contains floats
        # '%g' is often a good default: uses standard float or scientific
        # notation depending on magnitude and precision.
        # You could use '%.<precision>f' e.g., '%.6f' for fixed precision.
        fmt = '%g'
        dtype_description = "float"
    else:
        # Fallback for other types (e.g., boolean, complex, string)
        # Saving as string might be best general fallback, but check if suitable.
        fmt = '%s'
        dtype_description = f"other ({dtype})"
        print(f"Warning: Data type is {dtype}. Saving as string ('%s'). "
              "Verify output format.")

    # 4. Save the data
    print(f"Saving {dtype_description} data to {file_format} file: {filename}")
    np.savetxt(filename, data_np, fmt=fmt, delimiter=delimiter)