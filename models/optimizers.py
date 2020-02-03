from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
def get_optimizer_strategy(model='base', optim_policy=None, opt=None):
    lr_scale=4.0
    step=opt.optstep
    step1=int(step[0])
    step2=int(step[1])
    if model in ["softmax"]:
        optimizer = torch.optim.SGD(
            optim_policy, lr=opt.lr, weight_decay=opt.weight_decay,momentum=0.9
        )       
        def adjust_lr(optimizer, ep):
            #ep=ep%90
            if ep < step1:
                lr = 1e-2 * lr_scale
            elif ep< step2:
                lr = 1e-3 * lr_scale
            else:   
                lr = 1e-4 * lr_scale
            for p in optimizer.param_groups:
                p['lr'] = lr
            return lr

    elif model in ["triplet","distance"]   : 
        optimizer = torch.optim.Adam(
            optim_policy, lr=2e-4, weight_decay=opt.weight_decay
        )
    
        def adjust_lr(optimizer, ep):
            lr=2e-4
            if ep >=step1:
                lr = 2e-4 * (0.001 ** (float(ep + 1 - step1)/ (step2 + 1 - step1)))
            for p in optimizer.param_groups:
                p['lr'] = lr
            return lr

    return optimizer, adjust_lr
