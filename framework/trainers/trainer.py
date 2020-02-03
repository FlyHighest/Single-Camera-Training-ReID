from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import time
from utils.meters import AverageMeter
from framework.trainers.base_trainer import BaseTrainer


class clsTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer):
        super().__init__(opt, model, optimizer, criterion, summary_writer)

    def _parse_data(self, inputs):
        imgs, pids = inputs[:2]
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, _ = self.model(self.data)
        self.loss = self.criterion(score, self.target)

    def _backward(self):
        self.loss.backward()



class tripletTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer, criterion, summary_writer,need_cam=False):
        super().__init__(opt, model, optimizer, criterion, summary_writer)
        self.need_label = model.module.name in ['cosface', 'arcface', 'arcface_triplet', 'cosface_triplet']
        self.need_cam=need_cam

    def _parse_data(self, inputs):
        imgs, pids,cids = inputs[:3]
        self.data = imgs.cuda()
        self.target = pids.cuda()
        self.cids=cids.cuda()

    def _forward(self):
        if self.need_label:
            feat = self.model(self.data, self.target)
        else:
            feat = self.model(self.data)

        if self.need_cam:
            self.loss = self.criterion(feat, self.target,self.cids,epoch=self.epoch)
        else:
            self.loss = self.criterion(feat, self.target)
    def _backward(self):
        self.loss.backward()



