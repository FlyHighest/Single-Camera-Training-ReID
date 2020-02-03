from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import os


class DefaultConfig(object):
    keep_layer=None

    keepfc=False
    sampler="randomidentity" # or "randomidentitycamera"
    testepoch=90
    drop=0.5
    seed = 0
    stripes=6
    # dataset options
    trainset = 'market'
    testset = 'duke'
    height = 256
    width = 128
    # if load binary image mask
    with_normalize =True
    with_randomerase=False
    # gpu parameters
    num_gpu = 1
    device_ids =[0]

    # optimization options
    optim = 'Adam'
    optstep=(40,80)
    max_epoch = 300
    train_batch = 128
    test_batch = 128
    lr = 0.01
    step_size = 40
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    margin =None
    num_instances = 8
    margin1=0.3
    margin2=0.3
    # model options
    model_name = 'softmax'  # softmax, triplet, softmax_triplet
    last_stride = 1
    pretrained_model = 'resnet50-19c8e357.pth'
    model_path='resnet50-19c8e357.pth'
    # miscs
    print_freq = 30
    eval_step = 30
    save_dir = '/pytorch-ckpt/market'
    workers = 5
    start_epoch = 0

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
                exit()
            setattr(self, k, v)
        if self.num_gpu==0:
            os.environ["CUDA_VISIBLE_DEVICES"]=""
        if self.num_gpu==1:# is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] =str('3')
        if self.num_gpu==2:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
        if self.num_gpu==4:
            os.environ['CUDA_VISIBLE_DEVICES']="4,5,6,7"
        if self.num_gpu==8:
            os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

    def _state_dict_html(self):
        res="<p>"
        for k,_ in DefaultConfig.__dict__.items():
            if k.startswith('_'):
                continue
            res+=str(k)+" : "+str(getattr(self,k))+"<br />"
        res+='</p>'
        return res


opt = DefaultConfig()
