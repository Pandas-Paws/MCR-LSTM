"""
This file is part of the accompanying code to our manuscript:
Y. Wang, L. Zhang, N.B. Erichson, T. Yang. (2025). A Mass Conservation Relaxed (MCR) LSTM Model for Streamflow Simulation
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import add_camels_attributes, rescale_features
from papercode.lstm import LSTM
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list
from papercode.mclstm_modifiedhydrology import MassConservingLSTM # yhwang 20240604
from papercode.mclstm_mr_modifiedhydrology import MassConservingLSTM_MR # yhwang 20240604

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 256,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 64,
    'initial_forget_gate_bias': 3,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 365,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091999', format='%d%m%Y')
}

###############
# Prepare run #
###############


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness", "create_splits"])
    parser.add_argument(
        '--camels_root',
        type=str,
        default='./data/basin_dataset_public_v1p2/',
        help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help="User-selected GPU ID - if none chosen, will default to cpu")
    parser.add_argument(
        '--cache_data', type=str2bool, default=True, help="If True, loads all data into memory")
    parser.add_argument(
        '--num_workers', type=int, default=12, help="Number of parallel threads for data loading")
    parser.add_argument(
        '--no_static',
        type=str2bool,
        default=False,
        help="If True, trains LSTM without static features")
    parser.add_argument(
        '--concat_static',
        type=str2bool,
        default=False,
        help="If True, train LSTM with static feats concatenated at each time step")
    # -- start --
    # yhwang 20240604
    parser.add_argument(
        '--model_name',
        type=str,
        default='mcrlstm',
        help="Choose between ['lstm','mclstm','mcrlstm'].")
    # -- end --
    parser.add_argument(
        '--use_mse',
        type=str2bool,
        default=False,
        help="If True, uses mean squared error as loss function.")
    parser.add_argument(
        '--n_splits',
        type=int,
        default=None,
        help="Number of splits to create for cross validation")
    parser.add_argument(
        '--basin_file',
        type=str,
        default=None,
        help="Path to file containing usgs basin ids. Default is data/basin_list.txt")
    parser.add_argument(
        '--split',
        type=int,
        default=None,
        help="Defines split to use for training/testing in kFold cross validation")
    parser.add_argument(
        '--split_file',
        type=str,
        default=None,
        help="Path to file created from the `create_splits` function.")
    parser.add_argument('--epoch_num', type=int, default=30, help="Epoch number to evaluate")  # yhwang 202408


    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] in ["train", "create_splits"]) and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # GPU selection
    if cfg["gpu"] >= 0:
        device = f"cuda:{cfg['gpu']}"
    else:
        device = 'cpu'

    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=cfg["train_file"],
        basins=basins,
        dates=[cfg["train_start"], cfg["train_end"]],
        with_basin_str=True,
        seq_length=cfg["seq_length"],
        model_name=cfg["model_name"])
    return cfg


################
# Define Model #
################


class LSTM_Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(LSTM_Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        self.lstm = LSTM(
            input_size=input_size_dyn,
            hidden_size=hidden_size,
            initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n
        

###########################
# Train or evaluate model #
###########################

def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["split_file"] is not None:
        with Path(cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[cfg["split"]]["train"]
    else:
        basins = get_basin_list()

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare PyTorch DataLoader
    ds = CamelsH5(
        h5_file=cfg["train_file"],
        basins=basins,
        db_path=cfg["db_path"],
        concat_static=cfg["concat_static"],
        cache=cfg["cache_data"],
        no_static=cfg["no_static"])
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    # create model and optimizer
    input_size_dyn = 5 if (cfg["no_static"] or not cfg["concat_static"]) else 32
    if cfg["model_name"] == 'lstm':
        model = LSTM_Model(
            input_size_dyn=input_size_dyn,
            hidden_size=cfg["hidden_size"],
            initial_forget_bias=cfg["initial_forget_gate_bias"],
            dropout=cfg["dropout"],
            concat_static=cfg["concat_static"],
            no_static=cfg["no_static"]).to(DEVICE)
    elif cfg["model_name"] == 'mclstm':
        model = MassConservingLSTM(in_dim = 1, 
                            aux_dim = input_size_dyn-1, 
                            out_dim = cfg["hidden_size"], 
                            time_dependent=False, 
                            batch_first=True).to(DEVICE)
    elif cfg["model_name"] == 'mcrlstm':
        model = MassConservingLSTM_MR(in_dim = 1, 
                            aux_dim = input_size_dyn-1, 
                            out_dim = cfg["hidden_size"], 
                            time_dependent=False, 
                            batch_first=True).to(DEVICE)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 26: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        x, y, q_stds = data
        x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
        if cfg["model_name"] == 'mclstm':
            xm = x[..., 0:1]
            xa = x[..., 1:]
            # get model predictions
            m_out, c, _ = model(xm, xa)  # [batch size, seq length, hidden size]
            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
            predictions = output[:, -1, :]
            
        elif cfg["model_name"] == 'mcrlstm':
            xm = x[..., 0:1]
            xa = x[..., 1:]
            # get model predictions
            m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)  # [batch size, seq length, hidden size]
            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
            predictions = output[:, -1, :]
            
            
        elif cfg["model_name"] == 'lstm':
            predictions = model(x)[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if user_cfg["split_file"] is not None:
        with Path(user_cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg["split"]]["test"]
    else:
        basins = get_basin_list()

    # get attribute means/stds from trainings dataset
    train_file = user_cfg["run_dir"] / "data/train/train_data.h5"
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    ds_train = CamelsH5(
        h5_file=train_file, db_path=db_path, basins=basins, concat_static=run_cfg["concat_static"])
    means = ds_train.get_attribute_means()
    stds = ds_train.get_attribute_stds()

    # create model
    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else 32
    if run_cfg["model_name"] == 'lstm':
        model = LSTM_Model(
            input_size_dyn=input_size_dyn,
            hidden_size=run_cfg["hidden_size"],
            dropout=run_cfg["dropout"],
            concat_static=run_cfg["concat_static"],
            no_static=run_cfg["no_static"]).to(DEVICE)
    elif run_cfg["model_name"] == 'mclstm':
        model = MassConservingLSTM(in_dim = 1, 
                            aux_dim = input_size_dyn-1, 
                            out_dim = run_cfg["hidden_size"], 
                            time_dependent=False, 
                            batch_first=True).to(DEVICE)
    elif run_cfg["model_name"] == 'mcrlstm':
        model = MassConservingLSTM_MR(in_dim = 1, 
                            aux_dim = input_size_dyn-1, 
                            out_dim = run_cfg["hidden_size"], 
                            time_dependent=False, 
                            batch_first=True).to(DEVICE)

    # load trained model
    epoch_num = user_cfg["epoch_num"]
    weight_file = user_cfg["run_dir"] / f'model_epoch{epoch_num}.pt' # lstm for pub #12 best
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(
            camels_root=user_cfg["camels_root"],
            basin=basin,
            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
            is_train=False,
            seq_length=run_cfg["seq_length"],
            model_name=run_cfg["model_name"],
            with_attributes=True,
            attribute_means=means,
            attribute_stds=stds,
            concat_static=run_cfg["concat_static"],
            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)
        
        if run_cfg["model_name"] == 'lstm':
            preds, obs = evaluate_basin(run_cfg['model_name'], model, loader)
            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
        elif run_cfg["model_name"] == 'mclstm':
            preds, obs, trash_cell, cell_state = evaluate_basin(run_cfg['model_name'], model, loader)
            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten(), 
                                    'trashcell': trash_cell.flatten(), 'cellstate':cell_state.flatten()},
                                    index=date_range)
        elif run_cfg["model_name"] == 'mcrlstm':
            preds, obs, trash_cell, cell_state, MR_flow, out_flow = evaluate_basin(run_cfg['model_name'], model, loader)
            df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten(), 
                                    'trashcell': trash_cell.flatten(), 'cellstate':cell_state.flatten(),
                                    'mrflow': MR_flow.flatten(), 'outflow': out_flow.flatten()},
                                    index=date_range)

        results[basin] = df

    _store_results(user_cfg, run_cfg, results)


def evaluate_basin(model_name: str, model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None
    trash_cell, cell_state, MR_flow, out_flow = None, None, None, None

    with torch.no_grad():
        for data in loader:
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            # lstm
            if model_name == "lstm":
                p = model(x)[0]
            # mclstm
            elif model_name == 'mclstm':
                xm = x[..., 0:1]
                xa = x[..., 1:]
                m_out, c, o = model(xm, xa)
                output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
                p = output[:, -1, :]
                tc = m_out[:, -1, 0]  # [batch size, 1, last hidden node]
                cs = c[:, -1, :].sum(dim=-1, keepdim=True)
                if trash_cell is None:
                    trash_cell = tc.detach().cpu()
                    cell_state = cs.detach().cpu()
                else:
                    trash_cell = torch.cat((trash_cell, tc.detach().cpu()), 0)
                    cell_state = torch.cat((cell_state, cs.detach().cpu()), 0)
            # mcrlstm
            elif model_name == 'mcrlstm':
                xm = x[..., 0:1]
                xa = x[..., 1:]
                m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)
                output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
                p = output[:, -1, :]
                tc = m_out[:, -1, 0]  # [batch size, 1, last hidden node]
                cs = c[:, -1, :].sum(dim=-1, keepdim=True)
                mrf = mr_flow[:, -1, :].sum(dim=-1, keepdim=True)
                outf = o_flow[:, -1, :].sum(dim=-1, keepdim=True)
                if MR_flow is None:
                    trash_cell = tc.detach().cpu()
                    cell_state = cs.detach().cpu()
                    MR_flow = mrf.detach().cpu()
                    out_flow = outf.detach().cpu()
                else:
                    trash_cell = torch.cat((trash_cell, tc.detach().cpu()), 0)
                    cell_state = torch.cat((cell_state, cs.detach().cpu()), 0)
                    MR_flow = torch.cat((MR_flow, mrf.detach().cpu()), 0)
                    out_flow = torch.cat((out_flow, outf.detach().cpu()), 0)
                
            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)
        if model_name == 'lstm':
            preds = rescale_features(preds.numpy(), variable='output')
        else:
            preds = preds.numpy()
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0
        
        if  model_name == 'mclstm':
            trash_cell = trash_cell.numpy()
            cell_state = cell_state.numpy()
            return preds, obs, trash_cell, cell_state
        
        if model_name == 'mcrlstm':
            trash_cell = trash_cell.numpy()
            cell_state = cell_state.numpy()
            MR_flow = MR_flow.numpy()
            out_flow = out_flow.numpy()
            return preds, obs, trash_cell, cell_state, MR_flow, out_flow
            
    return preds, obs


def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"{run_cfg['model_name']}_no_static_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"{run_cfg['model_name']}_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"
        else:
            file_name = user_cfg["run_dir"] / f"ealstm_seed{run_cfg['seed']}_epoch{user_cfg['epoch_num']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
