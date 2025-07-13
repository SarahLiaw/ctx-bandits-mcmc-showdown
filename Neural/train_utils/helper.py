import numpy as np
from torch.optim import SGD
import sys as _sys

# Many downstream agent-construction helpers rely on a long list of custom
# algorithms that may or may not be present in every checkout.  To ensure that
# utility functions like `get_model` work even when some algorithms are absent,
# we try the bulk import and fall back to stubs if missing.
try:
    from algo import (
        LMCTS, LinTS, LinUCB,
        EpsGreedy, NeuralTS, NeuralUCB, NeuralEpsGreedy,
        UCBGLM, GLMTSL, NeuralLinUCB, MALATS,
        FGLMCTS, FGNeuralTS,
        # Commented out modules that don't exist or have issues
        # FGMALATS, SFGMALATS, PrecondLMCTS,
    )
    from algo.langevin import LangevinMC
    # Module doesn't exist
    # from algo.precond_langevin import PrecondLangevinMC
except ModuleNotFoundError as err:
    # Define minimal stubs that raise informative errors only when actually used.
    class _MissingAlgo:  # pylint: disable=too-few-public-methods
        def __init__(self, name):
            self._name = name
        def __getattr__(self, item):
            raise ImportError(f"Algorithm '{self._name}' is unavailable in this build (missing dependency)")
        def __call__(self, *args, **kwargs):
            raise ImportError(f"Algorithm '{self._name}' is unavailable in this build (missing dependency)")

    _missing_names = [
        'LMCTS', 'LinTS', 'LinUCB', 'EpsGreedy', 'NeuralTS', 'NeuralUCB',
        'NeuralEpsGreedy', 'UCBGLM', 'GLMTSL', 'NeuralLinUCB', 'MALATS',
        'FGLMCTS', 'FGNeuralTS', 'FGMALATS', 'SFGMALATS', 'PrecondLMCTS',
    ]
    current_module = globals()
    for _name in _missing_names:
        current_module[_name] = _MissingAlgo(_name)
    # Langevin utilities may also be missing
    LangevinMC = _MissingAlgo('LangevinMC')
    PrecondLangevinMC = _MissingAlgo('PrecondLangevinMC')
from models.classifier import LinearNet, FCN
from models.conv import MiniCNN, MiniConv
from models.linear import LinearModel
from train_utils.dataset import Collector
from .losses import construct_loss


def get_model(config, device):
    """
    Create a model based on the configuration.
    
    Args:
        config: Dictionary containing model configuration
        device: Device to place the model on
        
    Returns:
        model: PyTorch model
    """
    dim_context = config['dim_context']
    
    if config['model'] == 'linear':
        model = LinearModel(dim_context, 1)
    elif config['model'] == 'neural':
        model = FCN(1, dim_context, 
                   layers=config['layers'],
                   act=config['act'])
    elif config['model'] == 'cnn':
        model = MiniCNN()
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
        
    return model.to(device)


def construct_agent_cls(config, device):
    '''
    Construct agent for classification task
    '''
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    algo_name = config['algo']
    batchsize = config['batchsize'] if 'batchsize' in config else None
    decay = config['decay_step'] if 'decay_step' in config else 20
    reduce = config['reduce'] if 'reduce' in config else None
    # Define model
    if 'model' in config:
        if config['model'] == 'linear':
            model = LinearNet(1, dim_context * num_arm)
            model = model.to(device)
        elif config['model'] == 'neural':
            model = FCN(1, dim_context * num_arm,
                        layers=config['layers'],
                        act=config['act'])
            model = model.to(device)
        else:
            raise ValueError('Choose linear or neural for model please')
    else:
        model = None

    if algo_name == 'LinTS':
        nu = config['nu'] * np.sqrt(num_arm * dim_context * np.log(T))
        agent = LinTS(num_arm, num_arm * dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LinUCB':
        nu = config['nu'] * np.sqrt(dim_context * np.log(T))
        agent = LinUCB(num_arm, num_arm * dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LMCTS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        # create Lagevine Monte Carol optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        agent = LMCTS(model, optimizer, criterion,
                      collector,
                      name='LMCTS',
                      batch_size=batchsize,
                      reduce=reduce,
                      decay_step=decay,
                      device=device)
        
        
        #### MALATS
    
    # elif algo_name == "MALATS":
    #     beta_inv = config['beta_inv'] * dim_context * np.log(T)
    #     print(f'Beta inverse: {beta_inv}')

    #     # create optimizer
    #     optimizer = LangevinMC(model.parameters(), lr=config['lr'],
    #                            beta_inv=beta_inv, weight_decay=2.0)
    #     # Define loss function
    #     if 'loss' not in config:
    #         criterion = construct_loss('L2', reduction='sum')
    #     else:
    #         criterion = construct_loss(config['loss'], reduction='sum')

    #     collector = Collector()
        
    #     agent = LMCTS(model, optimizer, criterion,
    #                   collector,
    #                   name='MALATS',
    #                   batch_size=batchsize,
    #                   decay_step=decay,
    #                   device=device)

    ####

    elif algo_name == 'FGLMCTS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        # create Lagevine Monte Carol optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        agent = FGLMCTS(model, optimizer, criterion,
                      collector,
                      name='FGLMCTS',
                      batch_size=batchsize,
                      reduce=reduce,
                      decay_step=decay,
                      device=device)
    ####
    elif algo_name == 'MALATS':
        # compute temperature
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        # build the Langevin optimizer
        optimizer = LangevinMC(
            model.parameters(),
            lr=config['lr'],
            beta_inv=beta_inv,
            weight_decay=2.0
        )
        # loss
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')
        # data collector
        collector = Collector()
        # instantiate MALATS
        agent = MALATS(
            model,
            optimizer,
            criterion,
            collector,
            batch_size=batchsize,
            reduce=reduce,
            decay_step=decay,
            mala_step_size = config.get('mala_step_size', 1e-3),
            mala_n_steps = config.get('mala_n_steps', 10),
            mala_lazy = config.get('mala_lazy', True),
            device=device
        )
    ####
    elif algo_name == 'PrecondLMCTS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        optimizer_dummy = None   # real optimiser is built inside the class
        collector = Collector()
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')
        agent = PrecondLMCTS(
            model,               # first positional arg
            optimizer=optimizer_dummy,   # ignored
            criterion=criterion,
            collector=collector,
            batch_size=batchsize,
            reduce=reduce,
            decay_step=decay,
            lr=config['lr'],
            beta_inv=beta_inv,
            lambda_reg=config.get('lambda_reg', 1.0),
            device=device
        )


    elif algo_name == "FGMALATS":
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        optimizer = LangevinMC(
            model.parameters(),
            lr=config['lr'],
            beta_inv=beta_inv,
            weight_decay=2.0
        )

        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        agent = FGMALATS(
            model, optimizer, criterion, collector,
            feel_good=True,
            fg_mode="hard",
            lambda_fg=config.get('lambda_fg', 1.0),
            b_fg=config.get('b_fg', 1.0),
            batch_size=batchsize,
            decay_step=decay,
            device=device
        )

    elif algo_name == "SFGMALATS":
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        optimizer = LangevinMC(
            model.parameters(),
            lr=config['lr'],
            beta_inv=beta_inv,
            weight_decay=2.0
        )

        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        agent = SFGMALATS(
            model, optimizer, criterion, collector,
            feel_good=True,
            fg_mode="smooth",
            lambda_fg=config.get('lambda_fg', 1.0),
            b_fg=config.get('b_fg', 1.0),
            smooth_s=config.get('smooth_s', 10.0),
            batch_size=batchsize,
            decay_step=decay,
            device=device
        )

    ####
    
    elif algo_name == 'EpsGreedy':
        agent = EpsGreedy(num_arm, config['eps'])
    elif algo_name == 'NeuralTS':

        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context * num_arm,
                         model, optimizer,
                         criterion, collector,
                         config['nu'], reg=config['reg'],
                         batch_size=batchsize,
                         reduce=reduce,
                         device=device)
    elif algo_name == 'FGNeuralTS':
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = FGNeuralTS(num_arm, dim_context * num_arm,
                           model, optimizer,
                           criterion, collector,
                           config['nu'], reg=config['reg'],
                           batch_size=batchsize,
                           reduce=reduce,
                           feel_good=config.get('feel_good', True),
                           fg_mode=config.get('fg_mode', 'hard'),
                           lambda_fg=config.get('lambda_fg', 1.0),
                           b_fg=config.get('b_fg', 1.0),
                           smooth_s=config.get('smooth_s', 10.0),
                           device=device)
    elif algo_name == 'NeuralUCB':
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context * num_arm,
                          model, optimizer,
                          criterion, collector,
                          config['nu'], reg=config['reg'],
                          batch_size=batchsize,
                          reduce=reduce,
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                config['eps'],
                                batch_size=batchsize,
                                reduce=reduce,
                                device=device)
    else:
        raise ValueError(f'{algo_name} is not supported. Please choose from '
                         f'LinTS, LMCTS, NeuralTS, NeuralUCB, EpsGreedy')
    return agent


def construct_agent_sim(config, device):
    '''
    Construct agent for synthetic data (standard bandit setting)
    '''
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    algo_name = config['algo']
    sigma = config['sigma']
    T = config['T']
    batchsize = config['batchsize'] if 'batchsize' in config else None
    decay = config['decay_step'] if 'decay_step' in config else 20

    # Define model
    if 'model' in config:
        if config['model'] == 'linear':
            # model = LinearNet(1, dim_context)
            model = LinearModel(dim_context, 1)
            model = model.to(device)
        elif config['model'] == 'neural':
            model = FCN(1, dim_context,
                        layers=config['layers'],
                        act=config['act'])
            model = model.to(device)
        else:
            raise ValueError('Choose linear or neural for model please')
    else:
        model = None

    if algo_name == 'LinTS':
        nu = sigma * config['nu'] * np.sqrt(dim_context * np.log(T))
        agent = LinTS(num_arm, dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LinUCB':
        nu = config['nu'] * np.sqrt(dim_context * np.log(T))
        agent = LinUCB(num_arm, dim_context, nu, reg=1.0, device=device)
        
    #### LMCTS    
    elif algo_name == 'LMCTS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        print(f'Beta inverse: {beta_inv}')

        # create optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')
        collector = Collector()
        agent = LMCTS(model, optimizer, criterion,
                      collector,
                      name='LMCTS',
                      batch_size=batchsize,
                      decay_step=decay,
                      device=device)
        
    #### MALATS
    
    elif algo_name == "MALATS":
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        print(f'Beta inverse: {beta_inv}')

        # create optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        
        agent = LMCTS(model, optimizer, criterion,
                      collector,
                      name='MALATS',
                      batch_size=batchsize,
                      decay_step=decay,
                      device=device)

    ####
        
    elif algo_name == 'EpsGreedy':
        agent = EpsGreedy(num_arm, config['eps'])
    elif algo_name == 'NeuralTS':
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'], reg=config['reg'],
                         device=device)
    elif algo_name == 'FGNeuralTS':
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = FGNeuralTS(num_arm, dim_context,
                           model, optimizer,
                           criterion, collector,
                           config['nu'], reg=config['reg'],
                           feel_good=config.get('feel_good', True),
                           fg_mode=config.get('fg_mode', 'hard'),
                           lambda_fg=config.get('lambda_fg', 1.0),
                           b_fg=config.get('b_fg', 1.0),
                           smooth_s=config.get('smooth_s', 10.0),
                           device=device)
    elif algo_name == 'NeuralUCB':
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'], reg=config['reg'],
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                config['eps'], device=device)
    elif algo_name == 'UCBGLM':
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # optimizer = LBFGS(model.parameters(), max_iter=config['num_iter'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = UCBGLM(num_arm, dim_context,
                       model, optimizer,
                       criterion, collector,
                       config['nu'],
                       lr=config['lr'],
                       reg=config['reg'],
                       device=device)
    elif algo_name == 'GLMTSL':
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # optimizer = LBFGS(model.parameters(), max_iter=config['num_iter'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = GLMTSL(num_arm, dim_context,
                       model, optimizer,
                       criterion, collector,
                       config['nu'],
                       reg=config['reg'],
                       tao=config['tao'],
                       device=device)
    else:
        raise ValueError(f'{algo_name} is not supported. Please choose from '
                         f'LinTS, LMCTS, NeuralTS, NeuralUCB, EpsGreedy')
    return agent


def construct_agent_image(config, device):
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    algo_name = config['algo']
    T = config['T']
    batchsize = config['batchsize'] if 'batchsize' in config else None
    decay = config['decay_step'] if 'decay_step' in config else 20
    if algo_name == 'LMCTS':
        beta_inv = config['beta_inv'] * np.log(T)
        model = MiniCNN(in_channel=3 * num_arm).to(device)
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')
        collector = Collector()
        agent = LMCTS(model, optimizer, criterion,
                      collector,
                      batch_size=batchsize,
                      reduce=5,
                      decay_step=decay,
                      device=device)
        
        #### MALATS
    
    elif algo_name == "MALATS":
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        model = MiniCNN(in_channel=3 * num_arm).to(device)
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        
        agent = LMCTS(model, optimizer, criterion,
                      collector,
                      name='MALATS',
                      batch_size=batchsize,
                      reduce=5,
                      decay_step=decay,
                      device=device)
    ####
    
    
    elif algo_name == 'NeuralTS':
        model = MiniCNN(in_channel=3 * num_arm).to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'],
                         batch_size=batchsize,
                         image=True,
                         reg=config['reg'],
                         reduce=10,
                         device=device)
    elif algo_name == 'FGNeuralTS':
        model = MiniCNN(in_channel=3 * num_arm).to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = FGNeuralTS(num_arm, dim_context,
                           model, optimizer,
                           criterion, collector,
                           config['nu'],
                           batch_size=batchsize,
                           image=True,
                           reg=config['reg'],
                           reduce=10,
                           feel_good=config.get('feel_good', True),
                           fg_mode=config.get('fg_mode', 'hard'),
                           lambda_fg=config.get('lambda_fg', 1.0),
                           b_fg=config.get('b_fg', 1.0),
                           smooth_s=config.get('smooth_s', 10.0),
                           device=device)
    elif algo_name == 'NeuralUCB':
        model = MiniCNN(in_channel=3 * num_arm).to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'],
                          batch_size=batchsize,
                          image=True,
                          reduce=10,
                          reg=config['reg'],
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        model = MiniCNN(in_channel=3 * num_arm).to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                eps=config['eps'],
                                batch_size=batchsize,
                                reduce=5,
                                device=device)
    elif algo_name == 'NeuralLinUCB':
        model = MiniConv(in_channel=3 * num_arm).to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralLinUCB(num_arm, dim_context,
                             model, optimizer,
                             criterion, collector,
                             config['nu'],
                             batch_size=batchsize,
                             image=True,
                             reduce=10,
                             reg=config['reg'],
                             device=device)

    elif algo_name == 'FGMALATS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        optimizer = LangevinMC(model.parameters(),
                            lr=config['lr'],
                            beta_inv=beta_inv,
                            weight_decay=2.0)
        criterion = construct_loss(config.get('loss', 'L2'), reduction='sum')
        collector = Collector()

        agent = FGMALATS(
            model, optimizer, criterion, collector,
            batch_size=batchsize,
            reduce=reduce,
            decay_step=decay,
            device=device,
            beta_inv=beta_inv,
            accept_reject_step=config.get('accept_reject_step', 0),
            feel_good=config.get('feel_good', False),
            fg_mode=config.get('fg_mode', 'hard'),
            lambda_fg=config.get('lambda_fg', 0.0),
            b_fg=config.get('b_fg', 1.0),
            smooth_s=config.get('smooth_s', 10.0)
        )


    else:
        raise ValueError(f'Invalid algo name {algo_name}')
    return agent
