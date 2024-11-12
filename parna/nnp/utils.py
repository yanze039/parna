# from aimnet.train.train import train_function
from omegaconf import OmegaConf
import os
import torch
from ignite import distributed as idist
import logging

import sys
sys.path.insert(0, '/home/gridsan/ywang3/Project/Capping/software/aimnet2')

from aimnet.train import utils
from ignite.engine import Engine
from typing import Callable, Dict, Tuple, Union, Optional
import torch
from torch import Tensor


def prepare_batch(batch: Dict[str, Tensor], device='cuda', non_blocking=True) -> Dict[str, Tensor]:
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=non_blocking)
    return batch


_default_config = os.path.join(os.path.dirname(__file__), "..", "data", 'aimnet2_default_train.yaml')
_default_model = os.path.join(os.path.dirname(__file__), "..", "data", 'aimnet2_model_config.yaml')

def train_function(config, model, load, save, args):
    """Train AIMNet2 model.
    By default, will load AIMNet2 model and default train config.
    ARGS are one or more parameters wo overwrite in config in a dot-separated form.
    For example: `train.data=mydataset.h5`.
    """
    logging.basicConfig(level=logging.INFO)

    # model config
    logging.info('Start training')
    logging.info(f'Using model definition: {model}')
    model_cfg = OmegaConf.load(model)
    logging.info('--- START model.yaml ---')
    model_cfg = OmegaConf.to_yaml(model_cfg)
    logging.info(model_cfg)
    logging.info('--- END model.yaml ---')

    # train config
    logging.info(f'Using default training configuration: {_default_config}')
    train_cfg = OmegaConf.load(_default_config)
    for cfg in [config]:
        logging.info(f'Using additional configuration: {cfg}')
        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.load(cfg))
    if args:
        logging.info(f'Overriding configuration:')
        for arg in args:
            logging.info(arg)
        args_cfg = OmegaConf.from_dotlist(args)
        train_cfg = OmegaConf.merge(train_cfg, args_cfg)
    logging.info('--- START train.yaml ---')
    train_cfg = OmegaConf.to_yaml(train_cfg)
    logging.info(train_cfg)
    logging.info('--- END train.yaml ---')

    # launch
    num_gpus = torch.cuda.device_count()
    logging.info(f'Start training using {num_gpus} GPU(s):')
    for i in range(num_gpus):
        logging.info(torch.cuda.get_device_name(i))
    if num_gpus == 0:
        logging.warning('No GPU available. Training will run on CPU. Use for testing only.')
    if num_gpus > 1:
        logging.info('Using DDP training.')
        with idist.Parallel(backend='nccl', nproc_per_node=num_gpus) as parallel:
            parallel.run(run, model_cfg, train_cfg, load, save)
    else:
        run(0, model_cfg, train_cfg, load, save)


def run(local_rank, model_cfg, train_cfg, load, save):
    
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # load configs
    model_cfg = OmegaConf.create(model_cfg)
    train_cfg = OmegaConf.create(train_cfg)

    # build model
    _force_training = 'forces' in train_cfg.data.y
    model = utils.build_model(model_cfg, forces=_force_training)
    model = utils.set_trainable_parameters(model,
            train_cfg.optimizer.force_train,
            train_cfg.optimizer.force_no_train)
    model = idist.auto_model(model)
    
    # load weights
    if load is not None:
        device = next(model.parameters()).device
        logging.info(f'Loading weights from file {load}')
        sd = torch.load(load, map_location=device)
        logging.info(utils.unwrap_module(model).load_state_dict(sd, strict=False))
    
    # data loaders
    train_loader, val_loader = utils.get_loaders(train_cfg.data)

    # optimizer, scheduler, etc
    optimizer = utils.get_optimizer(model, train_cfg.optimizer)
    optimizer = idist.auto_optim(optimizer)
    if train_cfg.scheduler is not None:
        scheduler = utils.get_scheduler(optimizer, train_cfg.scheduler)
    else:
        scheduler = None
    loss = utils.get_loss(train_cfg.loss)
    metrics = utils.get_metrics(train_cfg.metrics)
    metrics.attach_loss(loss)

    # ignite engine
    trainer, validator = utils.build_engine(model, optimizer, scheduler, loss, metrics, train_cfg, val_loader)

    if local_rank == 0 and train_cfg.wandb is not None:
        utils.setup_wandb(train_cfg, model_cfg, model, trainer, validator, optimizer)
        
    trainer.run(train_loader, max_epochs=train_cfg.trainer.epochs)

    if local_rank == 0 and save is not None:
        logging.info(f'Saving model weights to file {save}')
        torch.save(utils.unwrap_module(model).state_dict(), save)


def conformer_trainer(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True) -> Engine:
    def _update(engine: Engine, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> float:
        model.train()
        optimizer.zero_grad()
        x = prepare_batch(batch[0], device=device, non_blocking=non_blocking)
        y = prepare_batch(batch[1], device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred["energy"] = y_pred["energy"] - y_pred["energy"][0]
        y["energy"] = y["energy"] - y["energy"][0]
        loss = loss_fn(y_pred, y)['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
        optimizer.step()
        return loss.item()
    return Engine(_update)


def conformer_evaluator(
        model: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = True) -> Engine:
    def _inference(engine: Engine, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        model.eval()
        x = prepare_batch(batch[0], device=device, non_blocking=non_blocking)
        y = prepare_batch(batch[1], device=device, non_blocking=non_blocking)
        with torch.no_grad():
            y_pred = model(x)
        y_pred["energy"] = y_pred["energy"] - y_pred["energy"][0]
        y["energy"] = y["energy"] - y["energy"][0]
        return y_pred, y
    return Engine(_inference)


def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'energy', key_true: str = 'energy') -> Tensor:
    """MSE loss normalized by the number of atoms.
    """
    x = y_true[key_true]
    y = y_pred[key_pred]  # torch.Size([4])

    # conformer pairwise energy loss
    x = x.view(-1, 1) - x.view(1, -1)
    y = y.view(-1, 1) - y.view(1, -1)
    s = y_pred['_natom'].sqrt()
    if y_pred['_natom'].numel() > 1:
        l = ((x - y).pow(2) / s).mean()
    else:
        l = torch.nn.functional.mse_loss(x, y) / s
    return l

