import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from datetime import datetime


def extract_state(observation, action_set):
    constraint_features = torch.FloatTensor(observation.row_features)
    edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
    edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1))
    variable_features = torch.FloatTensor(observation.column_features)
    action_set = torch.LongTensor(np.array(action_set, dtype=np.int64))
    action_set_size = action_set.shape[0]

    state = State(constraint_features, edge_index, edge_attr, variable_features, action_set, action_set_size)
    state.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
    return state


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    Takes a 1D tensor, splits it into slices according to pad_sizes, and pads each
    slice  with pad_value to obtain a 2D tensor of size (pad_sizes.shape[0], pad_sizes.max()).

    Parameters
    ----------
    input_ : 1D torch.Tensor
        Tensor to be sliced and padded.
    pad_sizes : 1D torch.Tensor
        Number of elements of the original tensor in each slice.
    pad_value : float (optional)
        Value to pad the tensor with.

    Returns
    -------
    output : 2D torch.Tensor
        Tensor resulting from the slicing + padding operations.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


def log(str, logfile=None):
    """
    Prints the provided string, and also logs it if a logfile is passed.

    Parameters
    ----------
    str : str
        String to be printed/logged.
    logfile : str (optional)
        File to log into.
    """
    str = f'[{datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


class State(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_index, edge_attr, variable_features,
                 action_set, action_set_size):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.variable_features = variable_features
        self.action_set = action_set
        self.action_set_size = action_set_size

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return State(**cuda_values)

class Transition(torch_geometric.data.Data):
    def __init__(self, state, action, reward=None):
        super().__init__()
        self.constraint_features = state.constraint_features
        self.edge_index = state.edge_index
        self.edge_attr = state.edge_attr
        self.variable_features = state.variable_features
        self.action_set = state.action_set
        self.action_set_size = state.action_set_size
        self.num_nodes = state.num_nodes

        self.action = action
        self.reward = reward

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'action_set':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)

    def to(self, device):
        """
        Current version is inplace, which is incoherent with how pytorch's to() function works.
        This overloads it.
        """
        cuda_values = {key: self[key].to(device) if isinstance(self[key], torch.Tensor) else self[key]
                        for key in self.keys}
        return Transition(**cuda_values)
