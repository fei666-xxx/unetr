# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time
import datetime

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather


from monai.data import decollate_batch
from torchvision.utils import save_image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torchmetrics import Specificity


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)
 

def softmax(x):

    max = np.max(
        x, axis=1, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=1, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

class MetricRecorder():
    def __init__(self):
        self.metrics = {
            'accuracy':-1.0,
            'precision':-1.0,
            'recall':-1.0,
            'f1score':-1.0,
            'specificity':-1.0,
            'auc':-1.0,
            'confusion matrix':[],}
    
    def update(self, metrics):
        for metric in self.metrics:
            self.metrics[metric] = metrics[metric]
            
    def GetMetrics(self):
        return self.metrics
    
    def __getitem__(self,key):
        return self.metrics[key]
    
    def __setitem__(self,key,value):
        self.metrics[key] = value
        
        

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        
        
        #if args.rank==0:
        #print(data.shape) # B C H W D
        #B,C,H,W,D = data.shape
        #save_image(data[:,:,:,:,64].view(B*C,H,W)[:64].unsqueeze(1),'test2.png',nrow=8,pad_value=1)
        #raise
        
        
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
            
        start_time = time.time()  
    for param in model.parameters():
        param.grad = None
    
    return run_loss.avg


def val_epoch(model, loader, epoch, args, model_inferer=None):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        
        pred_all = []
        target_all = []
        
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            
            if args.distributed:
                logits_list = distributed_all_gather([logits], out_numpy=False, is_valid=idx < loader.sampler.valid_length)
                logits = torch.cat(logits_list[0],dim=0)
                target_list = distributed_all_gather([target], out_numpy=False, is_valid=idx < loader.sampler.valid_length)
                target = torch.cat(target_list[0],dim=0)

            pred = logits.detach().cpu().numpy()
            pred_all.extend(pred.astype(float))
            target = target.detach().cpu().numpy()
            target_all.extend(target.astype(float))    
            
            if idx%5==0 and args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()  


        pred_all = np.stack(pred_all, axis=0)
        target_all = np.stack(target_all, axis=0)
        metrics = {}
        
        metrics['auc'] = roc_auc_score(target_all, pred_all, multi_class='ovo')
        pred_all = np.argmax(pred_all,axis=1)
        target_all = np.argmax(target_all,axis=1)
        
        metrics['f1score'] = f1_score(target_all, pred_all, average='weighted', zero_division=0)
        metrics['precision'] = precision_score(target_all,pred_all, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(target_all,pred_all, average='weighted', zero_division=0)
        
        specificity_score = Specificity(task="multiclass", num_classes=args.num_classes, threshold=1. / args.num_classes, average="weighted")
        metrics['specificity'] = float(specificity_score(torch.tensor(pred_all),torch.tensor(target_all)))
        
        metrics['accuracy'] = accuracy_score(target_all, pred_all)
        metrics['confusion matrix'] = confusion_matrix(target_all, pred_all).tolist()
        
    return metrics


def save_checkpoint(model, epoch, args, filename="model.pt", metrics = {}, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "state_dict": state_dict}
    save_dict.update(metrics)
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,  
    val_loader,
    optimizer,
    loss_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
        
    recorder = MetricRecorder()
    
    
    start_time = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            metrics = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                model_inferer=model_inferer,
                args=args,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "f1score {:.4f}/{:.4f}".format(metrics['f1score'], recorder['f1score']),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_f1score", metrics['f1score'], epoch)
                if metrics['f1score'] > recorder['f1score']:
                    print("new best ({:.6f} --> {:.6f}). ".format(recorder['f1score'], metrics['f1score']))
                    recorder.update(metrics)
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args, metrics=metrics, optimizer=optimizer, scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, filename="model_final.pt", metrics=metrics)
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if args.rank == 0:
        writer.close()
        print("Training Finished ! Best f1score:", recorder["f1score"])
        print('Training time {}'.format(total_time_str))

    recorder['training_time'] = total_time_str
    return recorder.GetMetrics()
