# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import cv2
import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from utils.utils import get_world_size, get_rank
from utils.vis import save_batch_image_with_joints
from utils.vis import get_max_preds
from core.evaluate import accuracy

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_loss_joints = AverageMeter()
    ave_loss_inp = AverageMeter()
    ave_acc = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader):
        images, labels, target_weight, _, name,joints,joints_vis = batch
        size = labels.size()
        #cv2.imwrite('groundtruth/gt_'+str(i_iter)+'.png', labels[0].detach().numpy())
        images = images.to(device)
        labels = labels.to(device)
   
        losses, losses_joints, losses_inp, pred = model(images, labels, target_weight) #forward
        #pred = F.upsample(input=pred, size=(size[-2], size[-1]), mode='bilinear')
        #pred = pred.to('cpu')
        #cv2.imwrite('prediction/pred_'+str(i_iter)+'.png',pred[0][0].detach().numpy())
        #print("saved")

        label_joints, _ = get_max_preds(labels[:,0:15,:,:].detach().cpu().numpy())
        pred_joints, _ = get_max_preds(pred[:,0:15,:,:].detach().cpu().numpy())

        _,acc,_,_ = accuracy(pred[:,0:15,:,:].detach().cpu().numpy(),labels[:,0:15,:,:].detach().cpu().numpy())

        save_batch_image_with_joints(images[:,0:3,:,:], label_joints*4, joints_vis,'results/full_RGBD/train/joint_gt/{}_gt.png'.format(i_iter))
        save_batch_image_with_joints(images[:,0:3,:,:], pred_joints*4, joints_vis,'results/full_RGBD/train/joint_pred/{}_pred.png'.format(i_iter))

        labels = F.upsample(input=labels, size=(256, 256), mode='bilinear')
        pred = F.upsample(input=pred, size=(256, 256), mode='bilinear')

        cv2.imwrite('results/full_RGBD/train/depth_gt/{}_gt.png'.format(i_iter),labels[0,15,:,:].detach().cpu().numpy())
        cv2.imwrite('results/full_RGBD/train/depth_pred/{}_pred.png'.format(i_iter),pred[0,15,:,:].detach().cpu().numpy())

        loss = losses.mean()
        loss_joints = losses_joints.mean()
        loss_inp = losses_inp.mean()

        reduced_loss = reduce_tensor(loss)
        reduced_loss_joints = reduce_tensor(loss_joints)
        reduced_loss_inp = reduce_tensor(loss_inp)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())
        ave_loss_joints.update(reduced_loss_joints.item())
        ave_loss_inp.update(reduced_loss_inp.item())
        ave_acc.update(acc)

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            print_loss_joints = ave_loss_joints.average() / world_size
            print_loss_inp = ave_loss_inp.average() / world_size
            print_acc = ave_acc.average() / world_size

            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}, {:.6f}, {:.6f}, Acc: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss, print_loss_joints, print_loss_inp,print_acc)
            logging.info(msg)
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer.add_scalar('train_loss_joint',print_loss_joints,global_steps)
            writer.add_scalar('train_loss_depth',print_loss_inp,global_steps)
            writer.add_scalar('train_accuracy',print_acc,global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, device):
    
    rank = get_rank() #0
    world_size = get_world_size() #1
    model.eval()
    ave_loss = AverageMeter()
    ave_loss_joints = AverageMeter()
    ave_loss_inp = AverageMeter()
    ave_accs = AverageMeter()
    ave_acc = AverageMeter()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))

    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
            image, label, target_weight, _, name, joints, joints_vis = batch
            size = label.size()
            #cv2.imwrite('validation_result/groundtruth/gt_'+str(i_iter)+'.png', label[0].detach().numpy())
            image = image.to(device)
            label = label.to(device)
            
            losses, losses_joints, losses_inp, pred = model(image, label, target_weight)

            #pred = F.upsample(input=pred, size=(64, 64), mode='bilinear')

            label_joints, _ = get_max_preds(label[:,0:15,:,:].detach().cpu().numpy())
            pred_joints, _ = get_max_preds(pred[:,0:15,:,:].detach().cpu().numpy())

            accs,acc,_,_ = accuracy(pred[:,0:15,:,:].detach().cpu().numpy(),label[:,0:15,:,:].detach().cpu().numpy())

            save_batch_image_with_joints(image[:,0:3,:,:], label_joints*4, joints_vis,'results/full_RGBD/val/joint_gt/{}_gt.png'.format(i_iter))
            save_batch_image_with_joints(image[:,0:3,:,:], pred_joints*4, joints_vis,'results/full_RGBD/val/joint_pred/{}_pred.png'.format(i_iter))

            label = F.upsample(input=label, size=(256, 256), mode='bilinear')
            pred = F.upsample(input=pred, size=(256, 256), mode='bilinear')

            cv2.imwrite('results/full_RGBD/val/depth_gt/{}_gt.png'.format(i_iter),label[0,15,:,:].detach().cpu().numpy())
            cv2.imwrite('results/full_RGBD/val/depth_pred/{}_pred.png'.format(i_iter),pred[0,15,:,:].detach().cpu().numpy())
            

            loss = losses.mean()
            loss_joints = losses_joints.mean()
            loss_inp = losses_inp.mean()

            reduced_loss = reduce_tensor(loss)
            reduced_loss_joints = reduce_tensor(loss_joints)
            reduced_loss_inp = reduce_tensor(loss_inp)

            ave_loss.update(reduced_loss.item())
            ave_loss_joints.update(reduced_loss_joints.item())
            ave_loss_inp.update(reduced_loss_inp.item())
            ave_acc.update(acc)
            ave_accs.update(accs)

    print_loss = ave_loss.average()/world_size
    print_loss_joints = ave_loss_joints.average() / world_size
    print_loss_inp = ave_loss_inp.average() / world_size
    print_acc = ave_acc.average() / world_size
    print_accs = ave_accs.average() / world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_loss_joint',print_loss_joints,global_steps)
        writer.add_scalar('valid_loss_depth',print_loss_inp,global_steps)
        writer.add_scalar('valid_accuracy',print_acc,global_steps)
        for i in range(15):
            writer.add_scalar('valid_each_accuracy_'+str(i),print_accs[i],global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, print_loss_joints, print_loss_inp, print_acc
    

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image_rgb, image_depth, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image_rgb, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
