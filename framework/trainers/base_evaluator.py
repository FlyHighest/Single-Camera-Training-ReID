from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
class BaseEvaluator(object):
    def __init__(self, model):
        self.model = model

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError

    def evaluate(self, queryloader, galleryloader, ranks):
        raise NotImplementedError

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50,q_path=None):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        r1_ls=[]
        r5_ls,r10_ls,r20_ls=[],[],[]
        querypath=[]
        num_valid_q = 0.  # number of valid query
        r1=0
        apdict=defaultdict(list)
        myr1=0
        r1dict=defaultdict(int)
        for q_idx in tqdm(range(num_q)):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            if q_path is not None:
                querypath.append(q_path[q_idx])
            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            i=0
            falsep=0
            dic=dict()
            for o in order:
                if g_pids[o]==q_pid and g_camids[o]==q_camid:
                    continue
                i=i+1
                if g_pids[o] != q_pid:
                    falsep+=1
                    continue
                if i==1:
                    myr1=myr1+1
                    r1dict[g_camids[o]]=r1dict[g_camids[o]]+1
                g_cid=g_camids[o]    
                if g_cid in dic:
                    dic[g_cid].append(falsep+1+len(dic[g_cid]))
                else:
                    dic[g_cid]=[falsep+1]
            
            for i in dic:
                l=len(dic[i])
                ap=0           # 
                for j in range(1,l+1):
                    ap+=j/dic[i][j-1]
                ap=ap/l
                apdict[i].append(ap)

            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
         
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            if cmc[0]==1:
                r1+=1
            r1_ls.append(cmc[0])
            r5_ls.append(cmc[5])
            r10_ls.append(cmc[10])
            r20_ls.append(cmc[20])
            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        for i in apdict:
            apdict[i]=np.mean(apdict[i])
            r1dict[i]=r1dict[i]/myr1

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
        all_cmc = np.asarray(all_cmc).astype(np.float32)
                
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

    def detailed_eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        q_ucams = np.unique(q_camids)
        g_ucams = np.unique(g_camids)
        pairwise_rank1 = np.zeros((q_ucams.shape[0], g_ucams.shape[0]))
        pairwise_map = np.zeros((q_ucams.shape[0], g_ucams.shape[0]))
        for i, q_cam in enumerate(q_ucams):
            for j, g_cam in enumerate(g_ucams):
                if not q_cam == g_cam:
                    q_indexes = q_camids == q_cam
                    g_indexes = g_camids == g_cam

                    patch_distmat = np.reshape(distmat[np.repeat(np.expand_dims(q_indexes, 1), g_indexes.shape[0], axis=1) & np.repeat(
                        np.expand_dims(g_indexes, 0), q_indexes.shape[0], axis=0)],
                               (np.sum(q_indexes.astype(int)), np.sum(g_indexes.astype(int))))

                    # patch_distmat = distmat[q_indexes, g_indexes]
                    all_cmc, mAP = self.eval_func(patch_distmat, q_pids[q_indexes], g_pids[g_indexes], q_camids[q_indexes], g_camids[g_indexes], max_rank)
                    pairwise_rank1[i, j] = all_cmc[0]
                    pairwise_map[i, j] = mAP
        os.makedirs('./visualization', exist_ok=True)
        f = plt.figure(figsize=(14, 12))
        plt.title('Pairwise Rank-1', y=1.05, size=15)
        sns.heatmap(pairwise_rank1.astype(float), linewidths=0.1, vmax=1.0,
                    square=True, linecolor='white', annot=True, cmap="YlGnBu")
        f.savefig(os.path.join('./visualization', "rank-1.pdf"), bbox_inches='tight')

        f = plt.figure(figsize=(14, 12))
        plt.title('Pairwise mAP', y=1.05, size=15)
        sns.heatmap(pairwise_map.astype(float), linewidths=0.1, vmax=1.0,
                    square=True, linecolor='white', annot=True, cmap="YlGnBu")
        f.savefig(os.path.join('./visualization', "map.pdf"), bbox_inches='tight')

