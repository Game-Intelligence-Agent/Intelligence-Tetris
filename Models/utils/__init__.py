from .dataset import *
from .optimizers import *
from .time_handler import *
from .tensorboard_handler import *


from torch import optim

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def build_optimizer(training_args, params):

    args = objectview(training_args)
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)

    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay, amsgrad = args.amsgrad)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'radam':
        optimizer = optim.RAdam(filter_fn, lr=args.lr, weight_decay=weight_decay)


    if args.lookahead:
        optimizer = Lookahead(optimizer)

    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.gamma)
    elif  args.opt_scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    elif args.opt_scheduler == 'rlop':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = args.mode, factor = args.gamma, patience = args.patience, verbose = 1, threshold = args.threshold, threshold_mode = args.threshold_mode, cooldown = args.cooldown, min_lr = args.min_lr)

    return scheduler, optimizer