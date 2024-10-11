import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple, List
import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # This line checks if GPU is available


class MassConservingLSTM_MR(nn.Module):
    """ Pytorch implementation of Mass-Conserving LSTMs with Mass Relaxation. """

    def __init__(self, in_dim: int, aux_dim: int, out_dim: int,
                 in_gate: nn.Module = None, out_gate: nn.Module = None,
                 redistribution: nn.Module = None, MR_gate: nn.Module = None,
                 time_dependent: bool = True,
                 batch_first: bool = False):
        """
        Parameters
        ----------
        in_dim : int
            The number of mass inputs.
        aux_dim : int
            The number of auxiliary inputs.
        out_dim : int
            The number of cells or, equivalently, outputs.
        in_gate : nn.Module, optional
            A module computing the (normalised!) input gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `in_dim` x `out_dim` matrix for every sample.
            Defaults to a time-dependent softmax input gate.
        out_gate : nn.Module, optional
            A module computing the output gate.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` vector for every sample.
        redistribution : nn.Module, optional
            A module computing the redistribution matrix.
            This module must accept xm_t, xa_t and c_t as inputs
            and should produce a `out_dim` x `out_dim` matrix for every sample.
        MR_gate : nn.Module, optional
            A module implementing the mass relaxing gate (MR gate).
            Defaults to a newly created `_MRGate` instance if not provided.
        batch_first : bool, optional
            Expects first dimension to represent samples if `True`,
            Otherwise, first dimension is expected to represent timesteps (default).
        """
        super().__init__()
        self.in_dim = in_dim
        self.aux_dim = aux_dim
        self.out_dim = out_dim
        self._seq_dim = 1 if batch_first else 0

        gate_inputs = aux_dim + out_dim + in_dim
        
        self.forward_count = 0
        self.step_count = 0

        # initialize gates
        if out_gate is None:
            self.out_gate = _Gate(in_features=gate_inputs, out_features=out_dim)
        if in_gate is None:
            self.in_gate = _NormalizedGate(in_features=gate_inputs,
                                           out_shape=(in_dim, out_dim),
                                           normalizer="normalized_sigmoid")
        if redistribution is None:
            self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                                  out_shape=(out_dim, out_dim),
                                                  normalizer="normalized_relu")
        '''
        define MR gate
        yhwang Apr 2024
        '''
        # Define MR gate
        if MR_gate is None:
            self.MR_gate = _MRGate(in_features=gate_inputs, out_features=out_dim)

        self._reset_parameters()

    @property
    def batch_first(self) -> bool:
        return self._seq_dim != 0

    def _reset_parameters(self, out_bias: float = -3.):
        nn.init.constant_(self.out_gate.fc.bias, val=out_bias)

    def forward(self, xm, xa, state=None):
        self.forward_count += 1
        xm = xm.unbind(dim=self._seq_dim)
        xa = xa.unbind(dim=self._seq_dim)

        if state is None:
            state = self.init_state(len(xa[0]))

        hs, cs, os, mrs, o_prime_s, mr_flow_s, o_flow_s = [], [], [], [], [], [], []
        for xm_t, xa_t in zip(xm, xa):  # xm, xa [365, 64, 1]
            # xm xa shape: [batchsize, 1] (i.e., [64,1])
            h, state, o, MR, o_prime, MR_flow, o_flow = self._step(xm_t, xa_t, state)  # h.shape=[64,128], state.shape=[64,128]
            
            hs.append(h)
            cs.append(state)
            os.append(o)
            mrs.append(MR)
            o_prime_s.append(o_prime)
            mr_flow_s.append(MR_flow)
            o_flow_s.append(o_flow)

        hs = torch.stack(hs, dim=self._seq_dim)  # [64, 365, 128]
        cs = torch.stack(cs, dim=self._seq_dim)  # [64, 365, 128]
        os = torch.stack(os, dim=self._seq_dim)
        mrs = torch.stack(mrs, dim=self._seq_dim)
        o_prime_s = torch.stack(o_prime_s, dim=self._seq_dim)
        mr_flow_s = torch.stack(mr_flow_s, dim=self._seq_dim)
        o_flow_s = torch.stack(o_flow_s, dim=self._seq_dim)
        return hs, cs, os, mrs, o_prime_s, mr_flow_s, o_flow_s

    @torch.no_grad()
    def init_state(self, batch_size: int, initial_cell_state=None): # yhwang May 2024
        """ Create the default initial state. """
        device = next(self.parameters()).device
        if initial_cell_state is None:
            return torch.zeros(batch_size, self.out_dim, device=device)
        else:
            return initial_cell_state

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM_MR. """
        # in this version of the MC-LSTM all available data is used to derive the gate activations. Cell states
        # are L1-normalized so that growing cell states over the sequence don't cause problems in the gates.
        self.step_count += 1
        features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-5)], dim=-1)

        # compute gate activations
        i = self.in_gate(features)
        r = self.redistribution(features)
        
        o = self.out_gate(features)
        #print("I am in step, o and o_supp")
        #print(o, o_supp)
        #print("I am in step, o gate bias")
        #print(self.out_gate.fc.bias

        # distribute incoming mass over the cell states
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)

        # reshuffle the mass in the cell states using the redistribution matrix
        
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)

        # compute the new mass states
        m_new = m_in + m_sys


        # compute the mass relaxing gate #TODO
        MR = self.MR_gate(c_0_norm=c / (c.norm(1) + 1e-5), f=1 - o)
        
        # todo
        '''
        # identify the MR's contribution
        # if MR[i] < -o: MR[i] = -o
        # else: MR[i] = MR[i]
        condition = MR < -o
        # Replace elements where condition is True with -o
        modified_MR = torch.where(condition, -o, MR)
        MR_flow = modified_MR * m_new

        # set the constraint that o+MR > 0
        relu = nn.ReLU()
        o_prime = relu(o + MR)
        
        '''
        o_prime = o + MR 
        MR_flow = MR * m_new
        
        
        return o * m_new, (1 - o_prime) * m_new, o, MR, o_prime, MR_flow, o * m_new


class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))


class _MRGate(nn.Module):
    """Utility class to implement a mass relaxing gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_MRGate, self).__init__()
        # Declare learnable parameters
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(out_features))
        self.weight_s_yvm = nn.Parameter(torch.FloatTensor(out_features, in_features)) #yhwang comment this out for test2
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(out_features, in_features)) #yhwang comment this out for test2
        #self.weight_s_yvm = nn.Parameter(torch.FloatTensor(out_features, out_features))
        #self.weight_r_yvm = nn.Parameter(torch.FloatTensor(1, out_features)) # yhwang test2 added this
        
        # Initialize the learnable parameters
        self._reset_parameters()
        # Define a ReLU activation function
        self.relu_v = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features) 

    def _reset_parameters(self):
        # Initialize learnable parameters using specific initialization methods
        nn.init.orthogonal_(self.weight_s_yvm)  # Initialize weight_s_yvm orthogonally
        nn.init.orthogonal_(self.weight_r_yvm)  # Initialize weight_r_yvm orthogonally
        nn.init.zeros_(self.bias_b0_yrm)  # Initialize bias_b0_yrm to zeros

    def forward(self, c_0_norm: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the Mass Relaxing gate."""

        # Compute ov0 using matrix multiplication and exponential function
        ov0 = torch.mm(c_0_norm - self.bias_b0_yrm, torch.exp(self.weight_s_yvm))
        # ov0 shape: [batch size, out_features]
        # f shape: [batch size, out_features]
        # weight_r_yvm shape: [out_features, in_features]
        # weight_s_yvm shape: [out_features, in_features]
        # Compute ov1 using sigmoid and tan activations
        '''
        **** c_0_norm shape:  torch.Size([256, 64]) bias_b0_yrm shape:  torch.Size([64]) weight_s_yvm shape:  torch.Size([64, 96])
        **** ov0 shape:  torch.Size([256, 96]) f shape:  torch.Size([256, 96]) weight_r_yvm shape:  torch.Size([64, 96])
        '''
        ov1 = self.layer_norm(torch.tanh(ov0) @ torch.sigmoid(self.weight_r_yvm.t()))
        #ov1 = self.relu_v(ov0 @ torch.sigmoid(self.weight_r_yvm.t())) # todo
        #ov1 = torch.tanh(self.layer_norm(torch.tanh(ov0) @ torch.sigmoid(self.weight_r_yvm.t()))) #yihwnawang test1 added this
        #ov1 = torch.tanh(ov0) * torch.sigmoid(self.weight_r_yvm) #yhwang test 2 element wise product
        #ov1 = self.layer_norm(ov1)

        
        # Compute the final ov value using ReLU activation
        ov2 = ov1 - self.relu_v(ov1 - f)  # ensure 1-o-MR>=0
        ov = self.relu_v(ov2 + 1 - f) + f - 1  # ensure o+MR>=0 #yihanwang test commented this out
        #ov=ov2 #yihanwang test added this
        
        return ov


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.
    :param model: A torch.nn.Module implementing the MC-LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    loss_list = []
    model.train()
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        
        xm = xs[..., 0:1]
        xa = xs[..., 1:]
        # get model predictions
        m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)  # [batch size, seq length, hidden size]

        #print("======== mr_flow, o_flow, c")
        #print(mr_flow[0,0:5, 0:5])
        #print(o_flow[0,0:5, 0:5])
        #print(c[0,0:5, 0:5])
        
        # y_hat = m_out[:, -1:].sum(dim=-1)
        output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
        # y_hat = output.transpose(0, 1) 
        y_hat = output[:, -1, :]
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()

        #for name, param in model.named_parameters():
        #    print(f"Gradients for {name}: {param.grad}")
        loss_list.append(loss)
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
    loss_ave = np.mean(torch.stack(loss_list).detach().cpu().numpy())
    return loss_ave


def eval_model(model, loader) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    hidden = []
    cell = []
    #out = []
    #MR =[]
    #out_prime = []
    MR_flow_all = []
    out_flow_all = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        COUNT = 0
        # request mini-batch of data from the loader
        for xs, ys in loader:
            COUNT+=1
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            xm = xs[..., 0:1]
            xa = xs[..., 1:]
            # get model predictions
            m_out, c, o, mr, o_prime, mr_flow, o_flow = model(xm, xa)
            #print("cell state end: ", c[0,-1, :].sum(dim=-1))
            #print("out fluxes sum: ", torch.sum(mr_flow[0,:, :]), torch.sum(o_flow[0,:, :]))
            #print("mass in sum: ", torch.sum(xm[0,:,:]))
            #print("mass in snap: ", xm[0,0:5,:])
            #print("=====")

            output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)  # trash cell excluded [batch size, seq length, 1]
            y_hat = output[:, -1, :]
            hidden_state = m_out[:, -1, :]  # [batch size, 1, hidden sizes]
            cell_state = c[:, -1, :].sum(dim=-1, keepdim=True)
            
            #out_gate = o[:, -1, :]
            #MR_gate = mr[:, -1, :]
            #out_prime_gate = o_prime[:, -1, :]
            
            MR_flow = mr_flow[:, -1, :].sum(dim=-1, keepdim=True)
            out_flow = o_flow[:, -1, :].sum(dim=-1, keepdim=True)
            
            obs.append(ys)
            preds.append(y_hat)
            hidden.append(hidden_state)
            cell.append(cell_state)
            #out.append(out_gate)
            #MR.append(MR_gate)
            #out_prime.append(out_prime_gate)
            
            MR_flow_all.append(MR_flow)
            out_flow_all.append(out_flow)
            
        #print("MCR cell state, hidden, and m tot sum: ", torch.cat(cell).sum(dim=0), torch.cat(MR_flow_all).sum(dim=0)+torch.cat(out_flow_all).sum(dim=0), torch.cat(cell).sum(dim=0)+torch.cat(MR_flow_all).sum(dim=0)+torch.cat(out_flow_all).sum(dim=0))
        #print("COUNT: ", COUNT)
            
            
    return torch.cat(obs), torch.cat(preds), torch.cat(hidden), torch.cat(cell), torch.cat(MR_flow_all), torch.cat(out_flow_all) # torch.cat(out), torch.cat(MR), torch.cat(out_prime)


def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_rmse(obs: np.array, sim: np.array) -> float:
    """Calculate Root Mean Squared Error (RMSE).

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: RMSE value.
    """
    # Only consider time steps where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # Check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    # Calculate RMSE
    rmse_val = np.sqrt(np.mean((sim - obs) ** 2))

    return rmse_val


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, sim, obs):
        return torch.sqrt(torch.mean((sim - obs) ** 2))
