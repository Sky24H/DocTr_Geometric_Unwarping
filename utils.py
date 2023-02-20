import torch
import random
import torchvision


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {}".format(phase, lrate, epoch, losses[0], losses[1]))



def show_tnsboard(global_step,writer,images,labels, pred, grid_samples,inp_tag, gt_tag, pred_tag):
    idxs=torch.LongTensor(random.sample(range(images.shape[0]), min(grid_samples,images.shape[0])))
    grid_inp = torchvision.utils.make_grid(images[idxs],normalize=True, scale_each=True)
    writer.add_image(inp_tag, grid_inp, global_step)
    grid_lbl = torchvision.utils.make_grid(labels[idxs],normalize=True, scale_each=True)
    writer.add_image(gt_tag, grid_lbl, global_step)
    grid_pred = torchvision.utils.make_grid(pred[idxs],normalize=True, scale_each=True)
    writer.add_image(pred_tag, grid_pred, global_step)