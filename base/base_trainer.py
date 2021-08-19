# :: dhruv :: #


import torch
from numpy import inf
from abc import abstractmethod
from logger import TensorboardWriter


class BaseTrainer():

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # config to monitor model params and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf


        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir

        # setup vis writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)



    @abstractmethod
    def _train_epoch(self, epoch):
        # training logic for an epoch

        raise NotImplementedError


    def train(self):
        # full training logic

        not_improveD_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged info into log dict
            log = {'epoch' : epoch}
            log.update(result)

            # print logged info to the screen
            for key, value in log.items():
                self.logger.info(f'{str(key):15s}: {value}')

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performacne improved or not, accordint to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or (self.mnt_mode ==
                            'max' and log[self.mnt_metric] >= self.mnt_best)

                except KeyError:
                    self.logger.warning(f'Warning :: Metric {self.mnt_metric} is not found. Model Performance monitoring is disabled.')
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True

                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {self.early_stop} epochs. Training  Stops.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)


    def _save_checkpoint(self, epoch, save_best=False):
        # save checkpoints

        # :param epoch:     current epoch number
        # :param log:       logging info of the epoch
        # :param save_best: if True, rename the saved checkpoint to 'model_best.pth'

        arch = type(self.model).__name__
        state = {
                'arch' : arch,
                'epoch' : epoch,
                'state_dict' : self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'monitor_best' : self.mnt_best,
                'config' : self.config
                 }

        filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f'Saving checkpoint : {filename} ...')

        if save_best:
            best_path = str(self.checkpoint_dir / f'model_best.pth')
            torch.save(state, best_path)
            self.logger.info(f'Saving current best : {best_path} ...')


    def _resume_checkpoint(self, resume_path):
        # resume from saved checkpoints

        # :param resume_path:   Checkpoint path to be resumed

        resume_path = str(resume_path)
        self.logger.info(f'Loading checkpoints : {resume_path} ...')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint
        if checkpoint['config']['arch'] != self.config['arch']:
            self.loger.warning("Warning : Architecture config given in config file is different from the one from checkpoint. This may yield an exception while state_dict is being loaded")

        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning : Optimizer in config file is different from that of checkpoint. Optimizer params not being resumed.")

        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint loaded. Resume training from epoch {self.start_epoch}')


