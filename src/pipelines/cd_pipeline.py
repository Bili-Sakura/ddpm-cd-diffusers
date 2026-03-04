"""
Change Detection pipeline.

Wraps the change detection head (cd_head_v2) with training, evaluation,
and metric-collection logic.
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os

import src.models.networks as networks
from src.models.base_model import BaseModel
from libs.metric_tools import ConfuseMatrixMeter
from libs.torchutils import get_scheduler

logger = logging.getLogger('base')


class CD(BaseModel):
    def __init__(self, opt):
        super(CD, self).__init__(opt)
        self.netCD = self.set_device(networks.define_CD(opt))

        self.loss_type = opt['model_cd']['loss_type']
        if self.loss_type == 'ce':
            self.loss_func = nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()

        if self.opt['phase'] == 'train':
            self.netCD.train()
            optim_cd_params = list(self.netCD.parameters())

            if opt['train']['optimizer']['type'] == 'adam':
                self.optCD = torch.optim.Adam(
                    optim_cd_params, lr=opt['train']['optimizer']['lr']
                )
            elif opt['train']['optimizer']['type'] == 'adamw':
                self.optCD = torch.optim.AdamW(
                    optim_cd_params, lr=opt['train']['optimizer']['lr']
                )
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(
                        opt['train']['optimizer']['type']
                    )
                )
            self.log_dict = OrderedDict()
            self.exp_lr_scheduler_netCD = get_scheduler(
                optimizer=self.optCD, args=opt['train']
            )
        else:
            self.netCD.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

        self.running_metric = ConfuseMatrixMeter(
            n_class=opt['model_cd']['out_channels']
        )
        self.len_train_dataloader = opt['len_train_dataloader']
        self.len_val_dataloader = opt['len_val_dataloader']

    def feed_data(self, feats_A, feats_B, data):
        self.feats_A = feats_A
        self.feats_B = feats_B
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optCD.zero_grad()
        self.pred_cm = self.netCD(self.feats_A, self.feats_B)
        l_cd = self.loss_func(self.pred_cm, self.data['L'].long())
        l_cd.backward()
        self.optCD.step()
        self.log_dict['l_cd'] = l_cd.item()

    def test(self):
        self.netCD.eval()
        with torch.no_grad():
            if isinstance(self.netCD, nn.DataParallel):
                self.pred_cm = self.netCD.module.forward(
                    self.feats_A, self.feats_B
                )
            else:
                self.pred_cm = self.netCD(self.feats_A, self.feats_B)
            l_cd = self.loss_func(self.pred_cm, self.data['L'].long())
            self.log_dict['l_cd'] = l_cd.item()
        self.netCD.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_cm'] = torch.argmax(
            self.pred_cm, dim=1, keepdim=False
        )
        out_dict['gt_cm'] = self.data['L']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netCD.__class__.__name__,
                self.netCD.module.__class__.__name__,
            )
        else:
            net_struc_str = '{}'.format(self.netCD.__class__.__name__)
        logger.info(
            'Change Detection Network structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n
            )
        )
        logger.info(s)

    def save_network(self, epoch, is_best_model=False):
        cd_gen_path = os.path.join(
            self.opt['path']['checkpoint'],
            'cd_model_E{}_gen.pth'.format(epoch),
        )
        cd_opt_path = os.path.join(
            self.opt['path']['checkpoint'],
            'cd_model_E{}_opt.pth'.format(epoch),
        )

        if is_best_model:
            best_cd_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_gen.pth'
            )
            best_cd_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_opt.pth'
            )

        network = self.netCD
        if isinstance(self.netCD, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, cd_gen_path)
        if is_best_model:
            torch.save(state_dict, best_cd_gen_path)

        opt_state = {'epoch': epoch, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optCD.state_dict()
        torch.save(opt_state, cd_opt_path)
        if is_best_model:
            torch.save(opt_state, best_cd_opt_path)

        logger.info('Saved current CD model in [{:s}] ...'.format(cd_gen_path))
        if is_best_model:
            logger.info(
                'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path)
            )

    def load_network(self):
        load_path = self.opt['path_cd']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for CD model [{:s}] ...'.format(
                    load_path
                )
            )
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            network = self.netCD
            if isinstance(self.netCD, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=True)

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optCD.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    def _update_metric(self):
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(),
            gt=self.data['L'].detach().cpu().numpy(),
        )
        return current_score

    def _collect_running_batch_states(self):
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()
        for k, v in scores.items():
            self.log_dict[k] = v

    def _clear_cache(self):
        self.running_metric.clear()

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netCD.step()
