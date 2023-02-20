# code to train world coord regression from RGB Image
# models are saved in checkpoints-wc/

import sys, os
import torch
# import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from torchvision import utils
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

from loaders import doctr_geo_loader
from GeoTr import GeoTr
from seg import U2NETP
from utils import reload_segmodel, get_lr, write_log_file, show_tnsboard

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm

def train(args):

    # Setup Dataloader
    data_path = args.data_path
    t_loader = doctr_geo_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=False)
    v_loader = doctr_geo_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    iter(trainloader).__next__()

    # Setup Model
    model = GeoTr_Seg().cuda()
    reload_segmodel(model.msk, args.Seg_path)

    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # model.cuda()

    # Activation
    htan = nn.Hardtanh(0,1.0)
    
    # Optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    # LR Scheduler 
    sched=CosineLRScheduler(optimizer, t_initial=args.n_epoch, lr_min=1e-4, warmup_t=20, warmup_lr_init=1e-5,warmup_prefix=True)
    # sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()

    epoch_start=0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                        .format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 
    
    #Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    experiment_name='htan_doc3d_l1grad_bghsaugk_scratch' #activation_dataset_lossparams_augmentations_trainstart
    log_file_name=os.path.join(args.logdir,experiment_name+'.txt')
    if os.path.isfile(log_file_name):
        log_file=open(log_file_name,'a')
    else:
        log_file=open(log_file_name,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'logs'), comment=experiment_name)

    best_val_mse = 99999.0
    global_step=0

    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        train_mse=0.0
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            pred=htan(outputs)
            # print(pred.shape, labels.shape)
            loss = loss_fn(pred, labels)
            avg_loss+=float(loss)

            # is this necessary?
            train_mse+=float(MSE(pred, labels).item())

            loss.backward()
            optimizer.step()
            global_step+=1

            if args.tboard:
                writer.add_scalar('L1 Loss/train', loss.item(), global_step)

            if (i+1) % 50 == 0:
                # if args.tboard:
                #     show_tnsboard(global_step, writer,images,labels,pred, 1,'Train/Inputs', 'Train/GTs', 'Train/Preds')
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), loss/50.0))
                avg_loss=0.0

        train_mse=train_mse/len(trainloader)
        avg_loss=avg_loss/len(trainloader)
        # avg_gloss=avg_gloss/len(trainloader)
        print("Training L1:%4f" %(avg_loss))
        print("Training MSE:'{}'".format(train_mse))
        train_losses=[avg_loss, train_mse]

        lrate=get_lr(optimizer)
        write_log_file(log_file_name, train_losses,epoch+1, lrate,'Train')


        model.eval()
        val_loss=0.0
        val_mse=0.0

        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())

                outputs = model(images_val)
                pred_val=htan(outputs)

                pred_val=pred_val
                labels_val=labels_val
                loss = loss_fn(pred_val, labels_val)
                val_loss+=float(loss)

                # same
                val_mse+=float(MSE(pred_val, labels_val))

        if args.tboard:
            show_tnsboard(epoch+1, writer,images_val,labels_val,pred, 1,'Val/Inputs', 'Val/GTs', 'Val/Preds')
            writer.add_scalar('L1 Loss/val', val_loss/(i_val+1), epoch+1)

        val_loss=val_loss/len(valloader)
        val_mse=val_mse/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_loss))
        print("val MSE: {}".format(val_mse))

        val_losses=[val_loss, val_mse]
        write_log_file(experiment_name, val_losses, epoch+1, lrate, 'Val')

        #reduce learning rate
        sched.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse=val_mse
            state = {'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, args.logdir+"{}_{}_{}_{}_best_model.pkl".format(epoch+1,val_mse,train_mse,experiment_name))

        if (epoch+1) % 10 == 0:
            state = {'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, args.logdir+"{}_{}_{}_{}_model.pkl".format(epoch+1,val_mse,train_mse,experiment_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--data_path', nargs='?', type=str, default='./datasets/doc3D', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=288, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=288, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints-geo/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true',
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--Seg_path',  default='./model_pretrained/seg.pth')
    parser.set_defaults(tboard=True)


    args = parser.parse_args()
    train(args)


# CUDA_VISIBLE_DEVICES=1 python train.py --data_path ./datasets/doc3D/ --batch_size 1