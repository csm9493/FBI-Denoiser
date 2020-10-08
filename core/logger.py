import sys
import random
import time
import datetime
import numpy as np
import scipy.io as sio

class Logger():
    def __init__(self, n_epochs, batches_epoch, sv_file_name= None):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.save_file_name = sv_file_name
        self.loss_save = {}


    def log(self, losses=None, lr = None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] lr : [%05f] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch, lr))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].data.cpu().numpy()
            else:
                self.losses[loss_name] += losses[loss_name].data.cpu().numpy()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_save:
                    self.loss_save[loss_name] = []
                    self.loss_save[loss_name].append(loss/self.batch)
                    
                else:
                    self.loss_save[loss_name].append(loss/self.batch)
                #Reset losses for next epoch
                self.losses[loss_name] = 0.0
                        
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1



