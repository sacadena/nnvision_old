import numpy as np
import torch
import copy

from mlutils.layers.cores import Stacked2dCore
from mlutils.layers.legacy import Gaussian2d
from mlutils.layers.readouts import PointPooled2d, FullGaussian2d
from mlutils.layers.activations import MultiplePiecewiseLinearExpNonlinearity

from nnfabrik.builder import get_model
from nnfabrik.utility.nn_helpers import set_random_seed, get_dims_for_loader_dict
from torch import nn
from torch.nn import functional as F

from .cores import SE2dCore, TransferLearningCore
from .readouts import MultipleFullGaussian2d, MultiReadout, MultipleSpatialXFeatureLinear, MultiplePointPooled2d, MultipleGaussian2d
from .utility import unpack_data_info, purge_state_dict, get_readout_key_names

try:
    from ..tables.from_nnfabrik import TrainedTransferModel, TrainedModel
    from ..tables.main import Recording
    from nnfabrik.main import Model
except ModuleNotFoundError:
    pass
except:
    print("dj database connection could not be established. no access to pretrained models available.")


def se_core_gauss_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                          hidden_kern=3, layers=3, gamma_input=15.5,
                          skip=0, final_nonlinearity=True, momentum=0.9,
                          pad_input=False, batch_norm=True, hidden_dilation=1,
                          laplace_padding=None, input_regularizer='LaplaceL2norm',
                          init_mu_range=0.2, init_sigma_range=0.5, readout_bias=True,  # readout args,
                          gamma_readout=4, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                          depth_separable=False, linear=False, data_info=None,
                          ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            sample = kwargs["sample"] if 'sample' in kwargs else None
            x = self.readout(x, data_key=data_key, sample=sample)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
                    hidden_channels=hidden_channels,
                    input_kern=input_kern,
                    hidden_kern=hidden_kern,
                    layers=layers,
                    gamma_input=gamma_input,
                    skip=skip,
                    final_nonlinearity=final_nonlinearity,
                    bias=False,
                    momentum=momentum,
                    pad_input=pad_input,
                    batch_norm=batch_norm,
                    hidden_dilation=hidden_dilation,
                    laplace_padding=laplace_padding,
                    input_regularizer=input_regularizer,
                    stack=stack,
                    se_reduction=se_reduction,
                    n_se_blocks=n_se_blocks,
                    depth_separable=depth_separable,
                    linear=linear)

    readout = MultipleGaussian2d(core, in_shape_dict=in_shapes_dict,
                                 n_neurons_dict=n_neurons_dict,
                                 init_mu_range=init_mu_range,
                                 bias=readout_bias,
                                 init_sigma_range=init_sigma_range,
                                 gamma_readout=gamma_readout)

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_full_gauss_readout(dataloaders,
                               seed,
                               hidden_channels=32,
                               input_kern=13,
                               hidden_kern=3,
                               layers=3,
                               gamma_input=15.5,
                               skip=0,
                               final_nonlinearity=True,
                               momentum=0.9,
                               pad_input=False,
                               batch_norm=True,
                               hidden_dilation=1,
                               laplace_padding=None,
                               input_regularizer='LaplaceL2norm',
                               init_mu_range=0.2,
                               init_sigma=1.,
                               readout_bias=True,
                               gamma_readout=4,
                               elu_offset=0,
                               stack=None,
                               se_reduction=32,
                               n_se_blocks=1,
                               depth_separable=False,
                               linear=False,
                               gauss_type='full',
                               grid_mean_predictor=None,
                               share_features=False,
                               share_grid=False,
                               data_info=None,
                               gamma_grid_dispersion=0,
                               ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        isotropic: whether the Gaussian readout should use isotropic Gaussians or not
        grid_mean_predictor: if not None, needs to be a dictionary of the form
            {
            'type': 'cortex',
            'input_dimensions': 2,
            'hidden_layers':0,
            'hidden_features':20,
            'final_tanh': False,
            }
            In that case the datasets need to have the property `neurons.cell_motor_coordinates`
        share_features: whether to share features between readouts. This requires that the datasets
            have the properties `neurons.multi_match_id` which are used for matching. Every dataset
            has to have all these ids and cannot have any more.
        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    source_grids = None
    grid_mean_predictor_type = None
    if grid_mean_predictor is not None:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop('type')
        if grid_mean_predictor_type == 'cortex':
            input_dim = grid_mean_predictor.pop('input_dimensions', 2)
            source_grids = {k: v.dataset.neurons.cell_motor_coordinates[:, :input_dim] for k, v in dataloaders.items()}
        elif grid_mean_predictor_type == 'shared':
            pass

    shared_match_ids = None
    if share_features or share_grid:
        shared_match_ids = {k: v.dataset.neurons.multi_match_id for k, v in dataloaders.items()}
        all_multi_unit_ids = set(np.hstack(shared_match_ids.values()))

        for match_id in shared_match_ids.values():
            assert len(set(match_id) & all_multi_unit_ids) == len(all_multi_unit_ids), \
                'All multi unit IDs must be present in all datasets'

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            sample = kwargs["sample"] if 'sample' in kwargs else None
            x = self.readout(x, data_key=data_key, sample=sample)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
                    hidden_channels=hidden_channels,
                    input_kern=input_kern,
                    hidden_kern=hidden_kern,
                    layers=layers,
                    gamma_input=gamma_input,
                    skip=skip,
                    final_nonlinearity=final_nonlinearity,
                    bias=False,
                    momentum=momentum,
                    pad_input=pad_input,
                    batch_norm=batch_norm,
                    hidden_dilation=hidden_dilation,
                    laplace_padding=laplace_padding,
                    input_regularizer=input_regularizer,
                    stack=stack,
                    se_reduction=se_reduction,
                    n_se_blocks=n_se_blocks,
                    depth_separable=depth_separable,
                    linear=linear)

    readout = MultipleFullGaussian2d(core, in_shape_dict=in_shapes_dict,
                                     n_neurons_dict=n_neurons_dict,
                                     init_mu_range=init_mu_range,
                                     bias=readout_bias,
                                     init_sigma=init_sigma,
                                     gamma_readout=gamma_readout,
                                     gauss_type=gauss_type,
                                     grid_mean_predictor=grid_mean_predictor,
                                     grid_mean_predictor_type=grid_mean_predictor_type,
                                     source_grids=source_grids,
                                     share_features=share_features,
                                     share_grid=share_grid,
                                     shared_match_ids=shared_match_ids,
                                     gamma_grid_dispersion=gamma_grid_dispersion,
                                     )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_point_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                          hidden_kern=3, layers=3, gamma_input=15.5,
                          skip=0, final_nonlinearity=True, momentum=0.9,
                          pad_input=False, batch_norm=True, hidden_dilation=1,
                          laplace_padding=None, input_regularizer='LaplaceL2norm',
                          pool_steps=2, pool_kern=3, init_range=0.2, readout_bias=True,  # readout args,
                          gamma_readout=4, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                          depth_separable=False, linear=False, data_info=None,
                          ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
                    hidden_channels=hidden_channels,
                    input_kern=input_kern,
                    hidden_kern=hidden_kern,
                    layers=layers,
                    gamma_input=gamma_input,
                    skip=skip,
                    final_nonlinearity=final_nonlinearity,
                    bias=False,
                    momentum=momentum,
                    pad_input=pad_input,
                    batch_norm=batch_norm,
                    hidden_dilation=hidden_dilation,
                    laplace_padding=laplace_padding,
                    input_regularizer=input_regularizer,
                    stack=stack,
                    se_reduction=se_reduction,
                    n_se_blocks=n_se_blocks,
                    depth_separable=depth_separable,
                    linear=linear)

    readout = MultiplePointPooled2d(core, in_shape_dict=in_shapes_dict,
                                    n_neurons_dict=n_neurons_dict,
                                    pool_steps=pool_steps,
                                    pool_kern=pool_kern,
                                    bias=readout_bias,
                                    gamma_readout=gamma_readout,
                                    init_range=init_range)

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def stacked2d_core_gaussian_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                                    hidden_kern=3, layers=3, gamma_hidden=0, gamma_input=0.1,
                                    skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                    pad_input=False, batch_norm=True, hidden_dilation=1,
                                    laplace_padding=None, input_regularizer='LaplaceL2norm',
                                    readout_bias=True, init_mu_range=0.2, init_sigma_range=0.5,  # readout args,
                                    gamma_readout=0.1, elu_offset=0, stack=None, isotropic=True, data_info=None,
                                    ):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]
        

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key, **kwargs)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = Stacked2dCore(input_channels=core_input_channels,
                         hidden_channels=hidden_channels,
                         input_kern=input_kern,
                         hidden_kern=hidden_kern,
                         layers=layers,
                         gamma_hidden=gamma_hidden,
                         gamma_input=gamma_input,
                         skip=skip,
                         final_nonlinearity=final_nonlinearity,
                         bias=core_bias,
                         momentum=momentum,
                         pad_input=pad_input,
                         batch_norm=batch_norm,
                         hidden_dilation=hidden_dilation,
                         laplace_padding=laplace_padding,
                         input_regularizer=input_regularizer,
                         stack=stack)

    readout = MultipleFullGaussian2d(core, in_shape_dict=in_shapes_dict,
                                     n_neurons_dict=n_neurons_dict,
                                     init_mu_range=init_mu_range,
                                     init_sigma=init_sigma_range,
                                     bias=readout_bias,
                                     gamma_readout=gamma_readout,
                                     gauss_type=isotropic
                                     )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_gauss_readout(dataloaders, seed,
                           input_channels=1, tr_model_fn='vgg16',  # begin of core args
                           model_layer=11, momentum=0.1, final_batchnorm=True,
                           final_nonlinearity=True, bias=False,
                           init_mu_range=0.4, init_sigma_range=0.6, readout_bias=True,  # begin or readout args
                           gamma_readout=0.002, elu_offset=-1, gauss_type='uncorrelated', data_info=None):
    """
    A Model class of a predefined core (using models from torchvision.models). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]


        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        """
        helper nn class that combines the core and readout into the final model
        """

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key) + self.core.regularizer()

    set_random_seed(seed)

    core = TransferLearningCore(input_channels=core_input_channels,
                                tr_model_fn=tr_model_fn,
                                model_layer=model_layer,
                                momentum=momentum,
                                final_batchnorm=final_batchnorm,
                                final_nonlinearity=final_nonlinearity,
                                bias=bias)

    readout = MultipleGaussian2d(core, in_shape_dict=in_shapes_dict,
                                     n_neurons_dict=n_neurons_dict,
                                     init_mu_range=init_mu_range,
                                     bias=readout_bias,
                                     gamma_readout=gamma_readout,
                                     init_sigma_range=init_sigma_range,
                                 )

    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_full_gauss_readout(dataloaders, seed,
                           input_channels=1, tr_model_fn='vgg16',  # begin of core args
                           model_layer=11, momentum=0.1, final_batchnorm=True,
                           final_nonlinearity=True, bias=False,
                           init_mu_range=0.4, init_sigma_range=0.6, readout_bias=True,  # begin or readout args
                           gamma_readout=0.002, elu_offset=-1, gauss_type='uncorrelated', data_info=None):
    """
    A Model class of a predefined core (using models from torchvision.models). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]

        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):
        """
        helper nn class that combines the core and readout into the final model
        """

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key) + self.core.regularizer()

    set_random_seed(seed)

    core = TransferLearningCore(input_channels=core_input_channels,
                                tr_model_fn=tr_model_fn,
                                model_layer=model_layer,
                                momentum=momentum,
                                final_batchnorm=final_batchnorm,
                                final_nonlinearity=final_nonlinearity,
                                bias=bias)


    readout = MultipleGaussian2d(core, in_shape_dict=in_shapes_dict,
                                 n_neurons_dict=n_neurons_dict,
                                 init_mu_range=init_mu_range,
                                 bias=readout_bias,
                                 init_sigma_range=init_sigma_range,
                                 gamma_readout=gamma_readout)
    
    
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def se_core_spatialXfeature_readout(dataloaders, seed, hidden_channels=32, input_kern=13,  # core args
                                    hidden_kern=3, layers=3, gamma_input=15.5,
                                    skip=0, final_nonlinearity=True, momentum=0.9,
                                    pad_input=False, batch_norm=True, hidden_dilation=1,
                                    laplace_padding=None, input_regularizer='LaplaceL2norm',
                                    init_noise=1e-3, readout_bias=True,  # readout args,
                                    gamma_readout=4, normalize=False, elu_offset=0, stack=None, se_reduction=32, n_se_blocks=1,
                                    depth_separable=False, linear=False, data_info=None):
    """
    Model class of a stacked2dCore (from mlutils) and a spatialXfeature (factorized) readout

    Args:

    Returns: An initialized model which consists of model.core and model.readout
    """

    if data_info is not None:
        n_neurons_dict, in_shapes_dict, input_channels = unpack_data_info(data_info)
    else:
        if "train" in dataloaders.keys():
            dataloaders = dataloaders["train"]


        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders)
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    core_input_channels = list(input_channels.values())[0] if isinstance(input_channels, dict) else input_channels[0]

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)

            x = self.readout(x, data_key=data_key,)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = SE2dCore(input_channels=core_input_channels,
                    hidden_channels=hidden_channels,
                    input_kern=input_kern,
                    hidden_kern=hidden_kern,
                    layers=layers,
                    gamma_input=gamma_input,
                    skip=skip,
                    final_nonlinearity=final_nonlinearity,
                    bias=False,
                    momentum=momentum,
                    pad_input=pad_input,
                    batch_norm=batch_norm,
                    hidden_dilation=hidden_dilation,
                    laplace_padding=laplace_padding,
                    input_regularizer=input_regularizer,
                    stack=stack,
                    se_reduction=se_reduction,
                    n_se_blocks=n_se_blocks,
                    depth_separable=depth_separable,
                    linear=linear)

    readout = MultipleSpatialXFeatureLinear(core, in_shape_dict=in_shapes_dict,
                                            n_neurons_dict=n_neurons_dict,
                                            init_noise=init_noise,
                                            bias=readout_bias,
                                            gamma_readout=gamma_readout,
                                            normalize=normalize
                                            )

    # initializing readout bias to mean response
    if readout_bias and data_info is None:
        for key, value in dataloaders.items():
            _, targets = next(iter(value))
            readout[key].bias.data = targets.mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def simple_core_transfer(dataloaders,
                         seed,
                         transfer_key=dict(),
                         core_transfer_table=None,
                         readout_transfer_table=None,
                         readout_transfer_key=dict(),
                         pretrained_features=False,
                         pretrained_grid=False,
                         pretrained_bias=False,
                         freeze_core=True,
                         **kwargs):

    if not readout_transfer_key and (pretrained_features or pretrained_grid or pretrained_bias):
        raise ValueError("if pretrained features, positions, or bias should be transferred, a readout transfer key "
                         "has to be provided, by passing it to the argument 'readout_transfer_key'")

    # set default values that are in line with parameter expansion
    if core_transfer_table is None:
        core_transfer_table = TrainedTransferModel
    elif core_transfer_table == "TrainedModel":
        core_transfer_table = TrainedModel

    if readout_transfer_table is None:
        readout_transfer_table = TrainedModel

    if kwargs:
        model_fn, model_config = (Model & transfer_key).fetch1("model_fn", "model_config")
        model_config.update(kwargs)
        model = get_model(model_fn=model_fn,
                          model_config=model_config,
                          dataloaders=dataloaders,
                          seed=seed)
    else:
        model = (Model & transfer_key).build_model(dataloaders=dataloaders, seed=seed)
    model_state = (core_transfer_table & transfer_key).get_full_config(include_state_dict=True)["state_dict"]

    core = purge_state_dict(state_dict=model_state, purge_key='readout')
    model.load_state_dict(core, strict=False)

    if freeze_core:
        for params in model.core.parameters():
            params.requires_grad = False

    if readout_transfer_key:
        readout_state = (readout_transfer_table & readout_transfer_key).get_full_config(include_state_dict=True)["state_dict"]
        readout = purge_state_dict(state_dict=readout_state, purge_key='core')
        feature_key, grid_key, bias_key = get_readout_key_names(model)

        if not pretrained_features:
            readout = purge_state_dict(state_dict=readout, purge_key=feature_key)

        if not pretrained_grid:
            readout = purge_state_dict(state_dict=readout, purge_key=grid_key)

        if not pretrained_bias:
            readout = purge_state_dict(state_dict=readout, purge_key=bias_key)


        model.load_state_dict(readout, strict=False)

    return model


def transfer_readout_augmentation(dataloaders,
                                  seed,
                                  transfer_key=dict(),
                                  core_transfer_table=None,
                                  readout_transfer_table=None,
                                  readout_transfer_key=dict(),
                                  model_config_kwargs=dict(),
                                  pretrained_features=False,
                                  pretrained_grid=False,
                                  pretrained_bias=False,
                                  augmented_in=False,
                                  augment_x_start=-1,
                                  augment_x_end=1,
                                  augment_y_start=-1,
                                  augment_y_end=1,
                                  n_augment_x=10,
                                  n_augment_y=10,
                                  train_augmented_pos=False,
                                  train_augmented_features=True):

    if not readout_transfer_key and (pretrained_features or pretrained_grid or pretrained_bias):
        raise ValueError("if pretrained features, positions, or bias should be transferred, a readout transfer key "
                         "has to be provided, by passing it to the argument 'readout_transfer_key'")

    # set default values that are in line with parameter expansion
    if core_transfer_table is None:
        core_transfer_table = TrainedTransferModel

    if readout_transfer_table is None:
        readout_transfer_table = TrainedModel

    if model_config_kwargs:
        model_fn, model_config = (Model & transfer_key).fetch1("model_fn", "model_config")
        model_config.update(model_config_kwargs)
        model = get_model(model_fn=model_fn,
                          model_config=model_config,
                          dataloaders=dataloaders,
                          seed=seed)
    else:
        model = (Model & transfer_key).build_model(dataloaders=dataloaders, seed=seed)
    model_state = (core_transfer_table & transfer_key).get_full_config(include_state_dict=True)["state_dict"]

    core = purge_state_dict(state_dict=model_state, purge_key='readout')
    model.load_state_dict(core, strict=False)

    for params in model.core.parameters():
        params.requires_grad = False

    if readout_transfer_key:
        readout_state = (readout_transfer_table & readout_transfer_key).get_full_config(include_state_dict=True)["state_dict"]
        readout = purge_state_dict(state_dict=readout_state, purge_key='core')
        feature_key, grid_key, bias_key = get_readout_key_names(model)

        if not pretrained_features:
            readout = purge_state_dict(state_dict=readout, purge_key=feature_key)

        if not pretrained_grid:
            readout = purge_state_dict(state_dict=readout, purge_key=grid_key)

        if not pretrained_bias:
            readout = purge_state_dict(state_dict=readout, purge_key=bias_key)


        model.load_state_dict(readout, strict=False)


    grid_augment = []
    for x in np.linspace(augment_x_start, augment_x_end, n_augment_x):
        for y in np.linspace(augment_y_start, augment_y_end, n_augment_y):
            grid_augment.append([x, y])

    grid_augment = torch.tensor(grid_augment)
    neuron_repeats = grid_augment.shape[0]

    total_n_neurons = 0
    for data_key, readout in model.readout.items():
        total_n_neurons += readout.outdims
    n_augmented_units = total_n_neurons * neuron_repeats

    in_shape = model.readout[data_key].in_shape
    gauss_type = model.readout[data_key].gauss_type

    if not augmented_in:
        model.readout["augmentation"] = FullGaussian2d(in_shape,
                                                       outdims=n_augmented_units,
                                                       bias=True,
                                                       gauss_type=gauss_type)

    # Setting the readout values to
    insert_index = 0
    for data_key, readout in model.readout.items():
        if augmented_in:
            if data_key != 'augmentation':
                continue
            for i in range(readout.outdims //neuron_repeats):

                model.readout["augmentation"].mu.data[0, insert_index:insert_index + neuron_repeats, 0,
                :] = grid_augment
                insert_index += neuron_repeats

        else:
            if data_key == 'augmentation':
                continue

            for i in range(readout.outdims):
                features = model.readout[data_key].features.data[:, :, :, i, ]
                bias = model.readout[data_key].bias.data[i]

                model.readout["augmentation"].features.data[:, :, :,
                insert_index:insert_index + neuron_repeats] = features[:, :, :, None]
                model.readout["augmentation"].mu.data[0, insert_index:insert_index + neuron_repeats, 0, :] = grid_augment
                model.readout["augmentation"].bias.data[insert_index:insert_index + neuron_repeats] = bias
                insert_index += neuron_repeats

    model.readout["augmentation"].mu.requires_grad = False
    model.readout["augmentation"].sigma.requires_grad = False

    return model


def se_core_shared_gaussian_readout(dataloaders,
                                    seed,
                                    key=None,
                                    model_fn=None,
                                    model_hash=None,
                                    dataset_fn=None,
                                    dataset_hash=None,
                                    trainer_fn=None,
                                    trainer_hash=None):
    if key is not None:
        dataloaders, model = TrainedModel().load_model(key)
    else:
        dataloaders, model = TrainedModel().load_model(dict(model_hash=model_hash,
                                                        dataset_hash=dataset_hash,
                                                        trainer_hash=trainer_hash, seed=seed), include_dataloader=True)


    data_key = list(model.readout.keys())[0]

    in_shape = model.readout[data_key].in_shape
    init_mu_range = model.readout[data_key].init_mu_range
    init_sigma = model.readout[data_key].init_sigma

    grid_augment = torch.tensor([[0, 0]])

    total_n_neurons = 0
    for data_key, readout in model.readout.items():
        if data_key == 'augmentation':
            continue
        total_n_neurons += readout.outdims

    n_augmented_units = total_n_neurons

    model.readout['augmentation'] = FullGaussian2d(in_shape=in_shape, outdims=n_augmented_units, bias=True,
                                                   init_mu_range=init_mu_range, init_sigma=init_sigma,
                                                   gauss_type='isotropic')
    model.cuda();
    insert_index = 0
    for data_key, readout in model.readout.items():

        if data_key == 'augmentation':
            continue

        for i in range(readout.outdims):
            features = model.readout[data_key].features.data[:, :, :, i]
            bias = model.readout[data_key].bias.data[i]
            sigma = model.readout[data_key].sigma.data[0][i]

            model.readout["augmentation"].features.data[:, :, :, insert_index] = features
            model.readout["augmentation"].bias.data[insert_index] = bias
            model.readout['augmentation'].sigma.data[:, insert_index, :, :] = sigma
            model.readout['augmentation'].mu.data[:, insert_index, :, :] = grid_augment

            insert_index += 1

    sessions = []
    for data_key in model.readout.keys():
        if data_key != 'augmentation':
            sessions.append(data_key)

    for session in sessions:
        model.readout.pop(session)

    return model


def augmented_full_readout(dataloaders=None,
                           seed=None,
                           key=None,
                            mua_in=False,
                            augment_x_start=-.75,
                            augment_x_end=.75,
                            augment_y_start=-.75,
                            augment_y_end=.75,
                            n_augment_x=5,
                            n_augment_y=5,
                            ):

    model = TrainedModel().load_model(key, include_dataloader=False)

    data_key = list(model.readout.keys())[0]

    in_shape = model.readout[data_key].in_shape
    init_mu_range = model.readout[data_key].init_mu_range
    init_sigma = model.readout[data_key].init_sigma

    grid_augment = []
    for x in np.linspace(augment_x_start, augment_x_end, n_augment_x):
        for y in np.linspace(augment_y_start, augment_y_end, n_augment_y):
            grid_augment.append([x, y])
    grid_augment.append([0, 0])
    grid_augment = torch.tensor(grid_augment)
    neuron_repeats = grid_augment.shape[0]

    total_n_neurons = 0
    for data_key, readout in model.readout.items():
        if data_key == 'augmentation':
            continue
        total_n_neurons += readout.outdims - (32 if mua_in else 0)

    n_augmented_units = total_n_neurons * neuron_repeats

    model.readout['augmentation'] = FullGaussian2d(in_shape=in_shape,
                                                   outdims=n_augmented_units,
                                                   bias=True,
                                                   init_mu_range=init_mu_range,
                                                   init_sigma=init_sigma,
                                                   gauss_type='isotropic')
    insert_index = 0
    for data_key, readout in model.readout.items():

        if data_key == 'augmentation':
            continue

        for i in range(readout.outdims - (32 if mua_in else 0)):
            features = model.readout[data_key].features.data[:, :, :, i]
            bias = model.readout[data_key].bias.data[i]
            sigma = model.readout[data_key].sigma.data[0][i]

            model.readout["augmentation"].features.data[:, :, :, insert_index:insert_index + neuron_repeats] = features[:, :, :, None]
            model.readout["augmentation"].bias.data[insert_index:insert_index + neuron_repeats] = bias
            model.readout['augmentation'].sigma.data[0, insert_index:insert_index + neuron_repeats, 0, :] = sigma
            model.readout["augmentation"].mu.data[0, insert_index:insert_index + neuron_repeats, 0, :] = grid_augment

            insert_index += neuron_repeats

    sessions = []
    for data_key in model.readout.keys():
        if data_key != 'augmentation':
            sessions.append(data_key)

    for session in sessions:
        model.readout.pop(session)

    return model

def stacked2d_core_dn_linear_readout(dataloaders, seed, hidden_channels=32, input_kern=13,          # core args
                                    hidden_kern=3, layers=1, gamma_hidden=0, gamma_input=0.1,
                                    skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                    pad_input=False, hidden_dilation=1, batch_norm=True, 
                                    batch_norm_scale=False, independent_bn_bias=False,
                                    laplace_padding=None, input_regularizer='LaplaceL2norm',
                                    spatial_norm_size=1, normalization_comp=1, dil_factor=1,       # dn args
                                    readout_bias=False, normalize=True, init_noise=1e-3, constrain_pos=False, # readout args,
                                    gamma_readout=0.1,  elu_offset=None, stack=None, use_avg_reg=False,
                                    final_nonlin=False, nonlin_bias=True, nonlin_initial_value=0.01, vmin=-3, vmax=6, 
                                     num_bins=50, smooth_reg_weight=0, smoothnes_reg_order=2):
    """
    Model class of a stacked2dCore (from mlutils), a divisive normalization layer and a SpatialXFeatureLinear readout
    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]
        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores, divisive_normalization and
            SpatialXFeatureLinear in mlutils.layers.readouts
    Returns: An initialized model which consists of model.core, model.dn and model.readout (depending on args: model.nonlin)
    
    Thus the divisive normalization code is private, it is stored in a different repo (divn), which must be installed to build this model.
    For access queries please contact @MaxFBurg.
    
    """
    
    from divn.lib.layer import DivisiveNormalizationLayer
    

    # make sure trainloader is being used
    dataloaders = dataloaders.get("train", dataloaders)

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields[:2]

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"
    
    assert not (readout_bias and nonlin_bias), "Readout and Nonlin bias are True, which will lead to a duplication"

    class Encoder(nn.Module):

        def __init__(self, core, dn, readout, nonlin):
            super().__init__()
            self.core = core
            self.dn = dn
            self.readout = readout
            self.nonlin = nonlin

        def forward(self, *args, data_key=None, **kwargs):
            x = args[0]
            x = self.core(x)
            x = self.dn(x)
            x = self.readout(x, data_key=data_key)
            if self.nonlin is None:
                return x
            else:
                return self.nonlin(x, data_key=data_key)

        def regularizer(self, data_key):
            if self.nonlin is None:
                return self.core.regularizer() + self.readout.regularizer(data_key=data_key)
            else:
                return self.core.regularizer() + self.readout.regularizer(data_key=data_key) + self.nonlin.regularizer(data_key=data_key)

        def _readout_regularizer_val(self):
            ret = 0
            with eval_state(model):
                for data_key in model.readout:
                    ret += self.readout.regularizer(data_key).detach().cpu().numpy()
            return ret

        def _core_regularizer_val(self):
            with eval_state(model):
                return self.core.regularizer().detach().cpu().numpy() if model.core.regularizer() else 0
            

        @property
        def tracked_values(self):
            return dict(readout_l1=self._readout_regularizer_val, 
                        core_reg=self._core_regularizer_val)

    set_random_seed(seed)

    # get a stacked2D core from mlutils
    core = Stacked2dCore(input_channels=input_channels[0],
                         hidden_channels=hidden_channels,
                         input_kern=input_kern,
                         hidden_kern=hidden_kern,
                         layers=layers,
                         gamma_hidden=gamma_hidden,
                         gamma_input=gamma_input,
                         skip=skip,
                         final_nonlinearity=final_nonlinearity,
                         elu_xshift=1,
                         elu_yshift=1,
                         bias=core_bias,
                         momentum=momentum,
                         pad_input=pad_input,
                         batch_norm=batch_norm,
                         batch_norm_scale=batch_norm_scale,
                         independent_bn_bias=independent_bn_bias,
                         hidden_dilation=hidden_dilation,
                         laplace_padding=laplace_padding,
                         input_regularizer=input_regularizer,
                         stack=stack,
                         use_avg_reg=use_avg_reg)
    
    #get a DN layer from divn
    dn = DivisiveNormalizationLayer(num_ch=hidden_channels,
                                    spatial_norm_size=spatial_norm_size,
                                    normalization_comp=normalization_comp,
                                    dil_factor=dil_factor,
                                    doDilated=True)
    

    readout = MultipleSpatialXFeatureLinear(nn.Sequential(core, dn),
                                            in_shape_dict=in_shapes_dict,
                                            n_neurons_dict=n_neurons_dict,
                                            bias=readout_bias,
                                            normalize=normalize,
                                            init_noise=init_noise,
                                            constrain_pos=constrain_pos,
                                            gamma_readout=gamma_readout)
    
    if final_nonlin:
    
        nonlin = MultiplePiecewiseLinearExpNonlinearity(n_neurons_dict=n_neurons_dict,
                                                        bias=nonlin_bias,
                                                        initial_value=nonlin_initial_value,
                                                        vmin=vmin,
                                                        vmax=vmax,
                                                        num_bins=num_bins,
                                                        smooth_reg_weight=smooth_reg_weight,
                                                        smoothnes_reg_order=smoothnes_reg_order)
    else:
        nonlin = None

    # initializing readout bias to mean response
    if readout_bias:
        for k in dataloaders:
            readout[k].bias.data = dataloaders[k].dataset[:][1].mean(0)

    model = Encoder(core, dn, readout, nonlin)

    return model
