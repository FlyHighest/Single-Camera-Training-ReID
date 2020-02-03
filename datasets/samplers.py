from __future__ import absolute_import

from collections import defaultdict

import numpy as np
import torch

from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        pid_cam=defaultdict(set) 
        cids=set()
        for index, (_, pid,cid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
            pid_cam[pid].add(cid)
            cids.add(cid)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        l=0 
        for p in pid_cam:
            l+=len(pid_cam[p])
        print('CP：{}'.format(l/len(self.pids)))

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return 20 


class RandomIdentityCameraSampler(Sampler):

    def __init__(self,data_source,num_instances=8,batch_size=256):
        self.data_source=data_source
        self.num_instances=num_instances
        self.pid_dic=defaultdict(list)
        self.camera_dic=defaultdict(list)
        self.pid_cid_dic=defaultdict(list)
        pid_cam=defaultdict(set)
        for index,(_,pid,cid,_) in enumerate(data_source):
            self.pid_dic[pid].append(index)
            self.pid_cid_dic[(pid,cid)].append(index)
            pid_cam[pid].add(cid)
            if not pid in self.camera_dic[cid]:
                self.camera_dic[cid].append(pid)
        self.pids=list(self.pid_dic.keys())
        self.num_identities=len(self.pids)
        self.cids=list(self.camera_dic.keys())
        self.num_cameras=len(self.cids)
        self.batch_size=batch_size
        self.identities_per_batch=batch_size//num_instances
        self.identities_per_cam=self.identities_per_batch//min(8,self.num_cameras)
        l=0       
        print(len(self.pid_dic)) 
        for p in pid_cam:
            l+=len(pid_cam[p])
        print("Camera/Person Value:{}".format(l/self.num_identities))
    def __len__(self):
        return self.num_identities*8 // self.batch_size

    def __iter__(self):
        for x in range(self.__len__()):
            cameras=torch.randperm(self.num_cameras)
            identities=[]
            ret=[]
            for c in cameras:
                cid=self.cids[c]
                t=self.camera_dic[cid]
                if len(t) < self.identities_per_cam:
                    continue
                replace=False  if len(t)>=self.identities_per_cam else True
                t=np.random.choice(t,size=self.identities_per_cam,replace=replace)
                for pid in t:
                    inds=self.pid_dic[pid] #self.pid_cid_dic[(pid,cid)]
                    replace=False if len(inds)>=self.num_instances else True
                    inds=np.random.choice(inds,size=self.num_instances,replace=replace)
                    ret.extend(inds)
                if len(ret)==self.batch_size:
                    break
            assert len(ret)==self.batch_size
            yield ret


class NormalCollateFn():
    def __call__(self, batch):
        N = len(batch)
        img_tensor = [x[0] for x in batch]
        pids = np.array([x[1] for x in batch])
        camids = np.array([x[2] for x in batch])
        paths=np.array([x[5] for x in batch])
        return torch.stack(img_tensor, dim=0), torch.from_numpy(np.array(pids)), torch.from_numpy(
            np.array(camids)),paths
