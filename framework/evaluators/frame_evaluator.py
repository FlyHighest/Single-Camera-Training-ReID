from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict
import numpy as np
import torch
import tqdm
import time
from sklearn.cluster import MiniBatchKMeans
from framework.trainers.base_evaluator import BaseEvaluator
from utils.re_ranking import re_ranking


class FrameEvaluator(BaseEvaluator):
    def __init__(self, model, flip=False):
        super().__init__(model)
        self.loop = 2 if flip else 1
        self.query_features = None
        self.gallery_features = None
        self.q_path=None

    def _parse_data(self, inputs):
        imgs, pids, camids,paths = inputs[:4]
        if torch.cuda.is_available():
            imgs=imgs.cuda()

        return imgs, pids, camids,paths

    def flip_tensor_lr(self, img):
        if torch.cuda.is_available():
            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
        else:
            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def _forward(self, inputs, label=None):
        with torch.no_grad():
            if torch.cuda.is_available() and self.model.module.name in ['cosface', 'arcface']:
                feature = self.model(inputs, label=None)
            else:
                feature = self.model(inputs)
        return feature.cpu()

    def evaluate(self, queryloader, galleryloader, ranks=[1, 5, 10, 20], normalize=True,rerank=False):
        self.model.eval()
        qf, q_pids, q_camids = [], [], []
        q_path=[]
        for batch_idx, inputs in tqdm.tqdm(enumerate(queryloader)):
            inputs, pids, camids,paths = self._parse_data(inputs)
            feature = None
            for i in range(self.loop):
                if i == 1:
                    inputs = self.flip_tensor_lr(inputs)
                f = self._forward(inputs, label=None)
                if feature is None:
                    feature = f
                else:
                    feature += f
            if normalize:
                fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
                feature = feature.div(fnorm.expand_as(feature))
            qf.append(feature)
            q_path.extend(paths)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        self.query_features = qf.numpy()
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        self.q_path=np.asarray(q_path)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        norm=np.array([])
        gf, g_pids, g_camids,g_path = [], [], [],[]
        for batch_idx, inputs in tqdm.tqdm(enumerate(galleryloader)):
            inputs, pids, camids,paths = self._parse_data(inputs)
            feature = None
            for i in range(self.loop):
                if i == 1:
                    inputs = self.flip_tensor_lr(inputs)
                f = self._forward(inputs)
                if feature is None:
                    feature = f
                else:
                    feature += f
            if normalize:
                fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
                norm=np.append(norm,fnorm)
                feature = feature.div(fnorm.expand_as(feature))
            gf.append(feature)
            g_path.extend(paths)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        print(gf.size())
        self.gallery_features = gf.numpy()
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_path=np.asarray(g_path)
        output={'feature':gf,'pid':g_pids,'camid':g_camids,'path':g_path}
        #np.save('output.npy',output)
        
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()
        #np.save('distmat.npy',distmat)
        rank1,mAP = self._compute_final_results(distmat, q_pids, q_camids, g_pids, g_camids, ranks)
        # get visualization
        # self.detailed_eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        if rerank:
            rerank_distmat = self._rerank(qf, gf)
            np.save('rerank_distmat1.npy',rerank_distmat)
            self._compute_final_results(rerank_distmat, q_pids, q_camids, g_pids, g_camids, ranks)
        return rank1,mAP

    def _compute_final_results(self, distmat, q_pids, q_camids, g_pids, g_camids, ranks):
        print("Computing CMC and mAP")

        cmc, mAP = self.eval_func(distmat, q_pids, g_pids, q_camids, g_camids,q_path=self.q_path)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0],mAP

    def _rerank(self, qf, gf):
        query_feature = qf.numpy()
        gallery_feature = gf.numpy()
        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
        print('start re-ranking...')
        since = time.time()
        re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        time_elapsed = time.time() - since
        print('Reranking complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        return re_rank
