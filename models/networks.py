from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .baseline_model import BaseNetBuilder

user_path='resnet50-19c8e357.pth'
def get_baseline_model(num_classes=None, model='base', last_stride=1, model_path=user_path,stripes=6,drop_prob=0.5):
    if model == 'softmax':
        model = BaseNetBuilder(num_classes, last_stride, model_path,drop_prob)
        optim_policy = model.get_optim_policy()
    elif model == 'triplet':
        model = BaseNetBuilder(None, last_stride, model_path,drop_prob)
        optim_policy = model.get_optim_policy()
    else:
        assert False, 'Unsupport Network'
    # elif model == 'arcface_triplet':
    #     model = ArcFaceNetBuilder(num_classes, last_stride, model_path)
    #     optim_policy = model.get_optim_policy()
    return model, optim_policy
