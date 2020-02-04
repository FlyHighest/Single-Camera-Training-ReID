from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from os import path as osp
from pprint import pprint
import random
import numpy as np
import torch

from tensorboardX import SummaryWriter

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.samplers import RandomIdentitySampler,RandomIdentityCameraSampler,  NormalCollateFn
from models import get_baseline_model
from models import get_optimizer_strategy
from framework.trainers.trainer import clsTrainer, tripletTrainer
from framework.evaluators import evaluator_manager
from utils.loss import TripletLoss
from utils.serialization import Logger
from utils.serialization import save_checkpoint
from utils.transforms import TrainTransform, TestTransform
from utils.meters import AverageMeter
from utils.loss import DistanceLoss

def train(**kwargs):
    #### Part 1 : Initialization
    opt._parse(kwargs)

    torch.backends.cudnn.deterministic = True
    # set random seed and cudnn benchmark
    #torch.manual_seed(opt.seed)
    #random.seed(opt.seed)
    #np.random.seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    #### Part 2 : Preparing Data
    print('initializing train dataset {}'.format(opt.trainset))
    train_dataset = data_manager.init_dataset(name=opt.trainset)

    print('initializing test dataset {}'.format(opt.testset))
    test_dataset = data_manager.init_dataset(name=opt.testset)

    pin_memory = True if use_gpu else False
    pin_memory=False
    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

   
    collateFn = NormalCollateFn()
    
    if opt.sampler=="randomidentity":
        trainloader = DataLoader(
            data_manager.init_datafolder(opt.trainset, train_dataset.train,
                                     TrainTransform(opt.height, opt.width, random_erase=opt.with_randomerase), if_train=True),
            sampler=RandomIdentitySampler(train_dataset.train, opt.num_instances),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True, collate_fn=collateFn,
        )
    elif opt.sampler=="randomidentitycamera":
        trainloader=DataLoader(
            data_manager.init_datafolder(opt.trainset, train_dataset.train,TrainTransform(opt.height, opt.width, random_erase=opt.with_randomerase), if_train=True),
                batch_sampler=RandomIdentityCameraSampler(train_dataset.train, opt.num_instances,opt.train_batch),
                num_workers=opt.workers,
                pin_memory=pin_memory, collate_fn=collateFn,
         )   

    queryloader = DataLoader(
        data_manager.init_datafolder(opt.testset, test_dataset.query, TestTransform(opt.height, opt.width),
                                     if_train=False),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        data_manager.init_datafolder(opt.testset, test_dataset.gallery, TestTransform(opt.height, opt.width),
                                     if_train=False),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )
    

    #### Part 3 : Preparing Backbone Network
    print('initializing model ...')
    if opt.model_name in ['triplet','distance']:
        model, optim_policy = get_baseline_model(num_classes=None, model='triplet')
    elif opt.model_name in ["softmax"]:
        model, optim_policy = get_baseline_model(train_dataset.num_train_pids, model='softmax',drop_prob=opt.drop)
    else:
        assert False, "unknown model name"
    if (not opt.model_path=='zero') and 'tar' in opt.model_path:
        print('load pretrain reid model......'+opt.model_path)
        ckpt = torch.load(opt.model_path)
    # remove classifer
        tmp = dict()
        for k, v in ckpt['state_dict'].items():
            if opt.keep_layer:
                for i in opt.keep_layer:
                    if 'layer'+str(i) in k :
                        #print(k+" skip....")
                        continue
            if opt.keepfc or ('fc' not in k and 'classifier' not in k):
                tmp[k] = v
        ckpt['state_dict'] = tmp
        model.load_state_dict(ckpt['state_dict'], strict=False)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    #### Part 4: Preparing Loss Functions
    if opt.margin1 is not None:
        distance_loss=DistanceLoss(margin=(opt.margin1,opt.margin2))
    else:
        distance_loss=DistanceLoss()
    tri_loss = TripletLoss(margin=opt.margin)
    xent_loss=nn.CrossEntropyLoss()

    vis=dict()
    vis['tri_acc1']=AverageMeter()
    vis['tri_acc2']=AverageMeter() 
    vis['cls_accuracy']=AverageMeter()
    vis['cls_loss']=AverageMeter()
    def dist_criterion(feat,targets,cameras,model=None,paths=None,epoch=0):
        dis_loss,tri_acc1,tri_acc2=distance_loss(feat,targets,cameras,model,paths,epoch=epoch)
        vis['tri_acc1'].update(float(tri_acc1))
        vis['tri_acc2'].update(float(tri_acc2))
        return dis_loss 
 
    def triplet_criterion(feat, targets):
        triplet_loss, tri_accuracy, _, _ = tri_loss(feat, targets)
        vis['tri_acc1'].update(float(tri_accuracy))
        return triplet_loss
    
    def cls_criterion(cls_scores,targets):
        cls_loss=xent_loss(cls_scores,targets)
        _,preds=torch.max(cls_scores.data,1)
        corrects=float(torch.sum(preds==targets.data))
        vis['cls_accuracy'].update(float(corrects/opt.train_batch))
        vis['cls_loss'].update(float(cls_loss))
        return cls_loss


    #### Part 5: Preparing Optimizer and Trainer
    optimizer, adjust_lr = get_optimizer_strategy(opt.model_name, optim_policy, opt)
    start_epoch = opt.start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    #model=model.cuda()
    # get trainer and evaluatori
    if opt.model_name=="distance": 
        reid_trainer=tripletTrainer(opt,model,optimizer,dist_criterion,summary_writer,need_cam=True)
    elif opt.model_name == 'triplet' or opt.model_name=='triplet_fc':
        reid_trainer = tripletTrainer(opt, model, optimizer, triplet_criterion, summary_writer)
    elif opt.model_name == 'softmax':
        reid_trainer = clsTrainer(opt, model, optimizer, cls_criterion, summary_writer)
    
    else:
        print("Error: Unknown model name {}".format(opt.model_name))
    reid_evaluator = evaluator_manager.init_evaluator(opt.testset, model, flip=True)
    
    #### Part 6 : Training
    best_rank1 = -np.inf
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.step_size > 0:
            current_lr = adjust_lr(optimizer, epoch)

        reid_trainer.train(epoch, trainloader)
        for k ,v in vis.items():
            print("{}:{}".format(k,v.mean))
            v.reset()

        if (epoch+1) ==opt.max_epoch:
            if use_gpu and opt.num_gpu>1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=False, save_dir=opt.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

        # skip if not save model
        if  (opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0  and epoch>=0 or (
                epoch + 1) == opt.max_epoch):  
            #print('Test on '+opt.testset)
            #rank1 = reid_evaluator.evaluate(queryloader, galleryloader,normalize=opt.with_normalize)
            print('Test on '+opt.trainset)
            if use_gpu and opt.num_gpu>1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch + 1,
            }, is_best=False, save_dir=opt.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

            rank1,mAP=reid_evaluator.evaluate(queryloader,galleryloader,normalize=opt.with_normalize)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': epoch + 1,
                    } , is_best=False, save_dir=opt.save_dir, filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achieved at epoch {}'.format(best_rank1, best_epoch))


def test(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)

    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_test_{}_{}.txt'.format(opt.testset,opt.testepoch)))

    if use_gpu:
        print('currently using GPU {}'.format(opt.device_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device_ids)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.testset))
    dataset = data_manager.init_dataset(name=opt.testset)

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        data_manager.init_datafolder(opt.testset, dataset.query, TestTransform(opt.height, opt.width), if_train=False),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        data_manager.init_datafolder(opt.testset, dataset.gallery, TestTransform(opt.height, opt.width),
                                     if_train=False),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('loading model ...')
    model, optim_policy = get_baseline_model(model="softmax",model_path=opt.model_path)
     
    best_model_path = os.path.join(opt.save_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path)==False:
        best_model_path=os.path.join(opt.save_dir, "{}_checkpoint_ep{}.pth.tar".format(opt.testepoch,opt.testepoch))
    
    if torch.cuda.is_available():
        ckpt=torch.load(best_model_path)
    else:
        ckpt = torch.load(best_model_path,map_location="cpu")

    # remove classifer

    tmp = dict()
    for k, v in ckpt['state_dict'].items():
        if 'fc' not in k and 'classifier' not in k:
            tmp[k] = v
    ckpt['state_dict'] = tmp
    print(model)
    print(ckpt)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('model size: {:.5f}M'.format(sum(p.numel()
                                           for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    reid_evaluator = evaluator_manager.init_evaluator(opt.testset, model, flip=True)
    reid_evaluator.evaluate(queryloader, galleryloader,normalize=opt.with_normalize,rerank=False)


if __name__ == '__main__':
    import fire

    fire.Fire()
