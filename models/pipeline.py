import h5py
import time
import torch
import scipy.io as sio
from pathlib import Path
from datasets.data import create_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.util import AverageMeter
from models import AttnFuseMain
import numpy as np
from collections import defaultdict
from models.loss_utils import get_loss
from utils.load_weight import module_load

def create_model(config, device):
    dataset_name = config.dataset_name
    model_name = config.model_name
    model = None
    if model_name == 'LFormer':
        model = AttnFuseMain(config.pan_dim, config.lms_dim, config.attn_dim,
                             config.hp_dim, config.n_stage, config.patch_merge,
                             config.crop_batch_size, config.patch_size_list, config.scale).to(device)
    else:
        assert f'{model_name} not supported now.'
    return model


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.writer = None
        dataset_name = config.dataset_name
        self.debug = config.debug
        if not self.debug:
            run_time = logger.handlers[0].baseFilename.split('/')[-1][:-4]
            self.run_time = run_time
            print(f"run_time: {run_time}")
            weights_save_path = Path(self.config.weights_path) / dataset_name / run_time
            print(f"weights_save_path: {weights_save_path}")
            weights_save_path.mkdir(exist_ok=True, parents=True)
            self.weights_save_path = weights_save_path
            tb_log_path = Path(self.config.tb_log_path) / run_time
            tb_log_path.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(str(tb_log_path))

        self.epoch_num = config.epoch_num
        self.train_loader, self.val_loader = create_loaders(config)
        base_lr = float(config.base_lr)
        weight_decay = float(config.weight_decay)
        device = torch.device('cuda:0')
        self.device = device
        self.model = create_model(config, device)
        self.model_name = config.model_name

        self.criterion = get_loss('l1ssim').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        step_size, gamma = int(config.step_size), float(config.gamma)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        return

    def train_all(self):
        print('Start training...')
        epoch_time = AverageMeter()
        end = time.time()

        ckpt = self.config.save_epoch
        model, optimizer, device = self.model, self.optimizer, self.device
        for epoch in range(self.epoch_num):
            epoch += 1
            epoch_train_loss = []

            model.train()
            for iteration, batch in enumerate(self.train_loader, 1):
                gt, lms, ms, pan_hp, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    Variable(batch[2]).to(device), \
                    batch[3], \
                    Variable(batch[4]).to(device)
                optimizer.zero_grad()  # fixed
                out, loss = model.train_step(ms, lms, pan, gt, self.criterion)

                epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

                loss.backward()  # fixed
                optimizer.step()  # fixed
            self.scheduler.step()

            t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
            self.logger.info('Epoch: {}/{} training loss:{:.7f}'.format(epoch, self.epoch_num, t_loss))

            if self.writer:
                self.writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
            self.validate()
            if epoch % ckpt == 0 and not self.debug:
                self.save_checkpoint(epoch)
            epoch_time.update(time.time() - end)
            end = time.time()
            remain_time = self.calc_remain_time(epoch, epoch_time)
            self.logger.info(f"remain {remain_time}")
        return

    def validate(self):
        epoch_val_loss = []
        model, device = self.model, self.device

        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_loader, 1):
                gt, lms, ms, _, pan = Variable(batch[0], requires_grad=False).to(device), \
                    Variable(batch[1]).to(device), \
                    Variable(batch[2]).to(device), \
                    batch[3], \
                    Variable(batch[4]).to(device)

                out, loss = model.train_step(ms, lms, pan, gt, self.criterion)
                epoch_val_loss.append(loss.item())
        v_loss = np.nanmean(np.array(epoch_val_loss))
        # writer.add_scalar('val/loss', v_loss, epoch)
        self.logger.info('validate loss: {:.7f}'.format(v_loss))
        return

    def save_checkpoint(self, epoch):
        model_out_path = str(self.weights_save_path / f'LFormer_epoch{epoch}.pth')
        ckpt = {'model': self.model.state_dict(), 'exp_timestamp': self.run_time}
        torch.save(ckpt, model_out_path)
        return

    def calc_remain_time(self, epoch, epoch_time):
        remain_time = (self.epoch_num - epoch) * epoch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        return remain_time


class Tester:
    def __init__(self, config):
        self.config = config
        dataset_name = config.dataset_name
        assert config.test_mode in ('reduced', 'full')
        data_path = Path(config.test_data_path)
        test_data_path = str(data_path)
        self.dataset = h5py.File(test_data_path, 'r')
        if 'gf2' in test_data_path:
            self.max_value = 1023.0
        elif 'cave' in test_data_path:
            self.max_value = 1.0
        else:
            self.max_value = 2047.0

        device = torch.device('cuda:0')
        self.model = create_model(config, device)
        self.model_name = config.model_name
        weight_path = config.test_weight_path
        ckpt = torch.load(weight_path, map_location=device)
        print(f"loading weight: {weight_path}")
        self.model.load_state_dict(ckpt['model'])
        # module_load(weight_path, self.model, device)
        save_path = Path(config.results_path) / f"{dataset_name}/{config.test_mode}/{weight_path.split('/')[-1].strip('.pth')}"
        save_path.mkdir(exist_ok=True, parents=True)
        self.save_path = save_path
        return

    def test(self, analyse_fms=False):
        features = defaultdict(list)

        def get_features(name):
            def hook(model, input, output):
                features[name].append(output.detach().cpu().numpy())

            return hook

        dataset, model = self.dataset, self.model
        keys = list(dataset.keys())
        if analyse_fms:
            rijabs = model.rijabs
            for i in range(model.block_num):
                cur_ln = f'rijab_{i}'
                rijab_i = getattr(rijabs, cur_ln)
                rijab_i.register_forward_hook(get_features(cur_ln))
                rijab_i.sat1.register_forward_hook(get_features(cur_ln + '.sat1'))
                rijab_i.sat3.register_forward_hook(get_features(cur_ln + '.sat3'))
                rijab_i.sat5.register_forward_hook(get_features(cur_ln + '.sat5'))


        if self.config.test_mode == 'reduced':
            ms = np.array(dataset[keys[2]], dtype=np.float32) / self.max_value
            lms = np.array(dataset[keys[1]], dtype=np.float32) / self.max_value
            pan = np.array(dataset[keys[3]], dtype=np.float32) / self.max_value
        else:
            ms = np.array(dataset[keys[1]], dtype=np.float32) / self.max_value
            lms = np.array(dataset[keys[0]], dtype=np.float32) / self.max_value
            pan = np.array(dataset[keys[2]], dtype=np.float32) / self.max_value

        ms = torch.from_numpy(ms).float().cuda()
        lms = torch.from_numpy(lms).float().cuda()
        pan = torch.from_numpy(pan).float().cuda()
        model.eval()
        print(f"save files to {self.save_path}")
        with torch.no_grad():

            for i in range(len(pan)):
                out = model.val_step(ms[i: i+1], lms[i:i+1], pan[i:i+1])
                I_SR = torch.squeeze(out * self.max_value).cpu().detach().numpy()  # BxCxHxW
                sio.savemat(str(self.save_path / f'output_mulExm_{i}.mat'), {'I_SR': I_SR.transpose(1, 2, 0)})
        return
