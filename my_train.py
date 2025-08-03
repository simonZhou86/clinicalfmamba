import argparse
import os
from os.path import exists
import sys
from IPython.core.interactiveshell import validate

sys.path.append("../")
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from model_mamba.clinicalmamba2D import *
from light_training.loss.my_utils import *
from light_training.dataloading.my_dataset import *
from light_training.loss.my_loss import *
from light_training.loss.eval_metric import *

from validation import *

print('finish')

parser = argparse.ArgumentParser(description='parameters for the training script')
parser.add_argument('--dataset', type=str, default="CT-MRI",
                    help="which dataset to use, available option: CT-MRI, MRI-PET, MRI-SPECT")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--lr_decay', type=bool, default=False, help='decay learing rate?')
parser.add_argument('--exp_folder_name', type=str, default="", help='exp folder name')
parser.add_argument('--lambda1', type=float, default=0.5, help='weight for image gradient loss')
parser.add_argument('--lambda2', type=float, default=0.5, help='weight for perceptual loss')
# parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda', default=True)
parser.add_argument('--seed', type=int, default=3407, help='random seed to use')
parser.add_argument('--base_loss', type=str, default='l1_charbonnier',
                    help='which loss function to use for pixel-level (l2 or l1 charbonnier)')

opt = parser.parse_args()

######### whether to use cuda ####################
device = torch.device("cuda:0")
#################################################

########## seeding ##############
seed_val = opt.seed
random_seed(seed_val, True)
################################

############ making dirs########################
res_dir = "/content/drive/MyDrive/SegMamba/my_res"
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
# model_dir = os.path.join(res_dir, "pretrained_models")
# if not os.path.exists(model_dir):
#     os.mkdir(model_dir)
# if not os.path.exists(test_data_dir):
#     os.mkdir(test_data_dir)
lr_decay = True
EPOCH = 200
val_every = 50
################################################

def normalize(im):
    # normalize to [0,1]
    mins = [im[idx].min() for idx in range(len(im))]
    maxes = [im[idx].max() for idx in range(len(im))]

    for idx in range(len(im)):
        min_val = mins[idx]
        max_val = maxes[idx]

        if min_val == max_val:
            im[idx] = torch.zeros(im[idx].shape)
        else:
            im[idx] = (im[idx] - min_val)/(max_val - min_val)

####### loading dataset ####################################
data_dir = "./my_data" #"/content/drive/MyDrive/SegMamba/my_data/SPECT-MRI"
# ct_file, mri_file = get_common_file(data_dir)
# assert len(ct_file) == len(mri_file)
# train_ct, train_mri = load_data(ct_file, data_dir)
# # torch.save(test_ct, os.path.join(config.test_data_dir, "ct_test.pt"))
# # torch.save(test_mri, os.path.join(config.test_data_dir, "mri_test.pt"))
# total_files = len(ct_file)
# cv_datasets = get_cv_dataset(total_files, fold = 5, shuffle = True)
# # print(train_ct.shape, train_mri.shape, test_ct.shape, test_mri.shape)
# # print(cv_datasets)
# print("finish creating cv datasets")
#train_total = torch.cat((train_ct, train_mri), dim=0).to(device)

# these loaders return index, not the actual image
# train_loader, val_loader = get_loader(train_ct, train_mri, config.train_val_ratio, opt.batch_size)
# print("train loader length: ", len(train_loader), " val loder length: ", len(val_loader))

# check the seed is working
# for batch_idx in train_loader:
#     batch_idx = batch_idx.view(-1).long()
#     print(batch_idx)
# print("validation index")
# for batch_idx in val_loader:
#     batch_idx = batch_idx.view(-1).long()
#     print(batch_idx)
# sys.exit()
############################################################

############ loading model #####################
        
###################################################

##### downloading pretrained vgg model ##################
# vgg = vgg16_bn(pretrained=True)
########################################################


my_loss_fn = fusion_loss_med3D()

########################################
NUM_EXP = 3
exp_folder_name = opt.exp_folder_name
####### loading dataset ####################################
# model = MambaFuse(dim = 128, H = 32, W = 32, outchannel = 256, depth=5).cuda()
#U2Net(16, 1, 1).cuda()
# optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# if lr_decay:
#     # stepLR = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#     stepLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.000001)

target_dir = data_dir
ct, mri = get_common_file(target_dir)
ct_left = ct.copy()
tsize = 0
if "SPECT" in opt.dataset:
    tsize = 50
else:
    tsize = 30
    
for exp in range(NUM_EXP):
    model = MambaFuse3D(dim = 128, H = 16, W = 16, Z = 16, outchannel = 128, depth=5).cuda()
    # #U2Net(16, 1, 1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if lr_decay:
        #stepLR = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        stepLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0.000001)
    
    test_ind = np.random.choice(len(ct_left), size=tsize, replace = False)
    print(test_ind)
    test = []
    for ind in test_ind:
        test.append(ct_left[ind])
    for fn in test:
        ct_left.remove(fn)
    print(f"ct_left len: {len(ct_left)}")
    
    if "SPECT" in opt.dataset:
        train_sp, train_mri, test_sp, test_mri = load_data_MRSPECT(ct, target_dir, test) # for CT/MRI change to load_data2
    else:
        train_sp, train_mri, test_sp, test_mri = load_data2(ct, target_dir, test)
    
    fold_path = f"{res_dir}/{opt.dataset}/{exp_folder_name}_{exp}"
    os.makedirs(fold_path, exist_ok=True)
    model_dir = f"{res_dir}/{opt.dataset}/{exp_folder_name}_{exp}/pretrained_models"
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(test_sp, os.path.join(fold_path, "sp_test.pt"))
    torch.save(test_mri, os.path.join(fold_path, "mri_test.pt"))
    print(train_sp.shape, train_mri.shape, test_sp.shape, test_mri.shape)
    assert test_sp.shape[0] != 0 or test_mri.shape[0] != 0, "empty test set!"
    # train_total = torch.cat((train_sp, train_mri), dim=0).to(device)
    
    # print(train_mri.shape)
    (train_loader_ct, val_loader_ct), (train_loader_mri, val_loader_mri) = get_loader_seperate(train_sp, train_mri, 0.8, opt.batch_size)
    # train_loader = DataLoader(train_ind, batch_size=4, num_workers=0, shuffle=True, drop_last=False)
    # val_loader = DataLoader(val_ind, batch_size=1, num_workers=0, shuffle=True, drop_last=False)
    # train_total = torch.cat((train_sp, train_mri), dim=0).to(device)

    # train_loader, val_loader = get_loader(train_sp, train_mri, 0.8, opt.batch_size)
    # print("val num: ", len(val_loader_ct))
    train_loss = []
    val_loss = []
    t = trange(opt.epochs, desc='Training progress...', leave=True)
    lowest_val_loss = int(1e9)
    best_ssim = 0
    use_mixed_precision = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    for i in t:
        print("\n new epoch {} starts for exp {}!".format(i, exp))
        # clear gradient in model
        model.zero_grad()
        b_loss = 0
        # train model
        model.train()
        for j, batch_idx in enumerate(train_loader_ct): #train_loader
            # clear gradient in optimizer
            optimizer.zero_grad()
            batch_idx = batch_idx.view(-1).long()
            
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                # 3d data: may need to add .unsqueeze(1)
                img1 = train_mri[batch_idx].cuda()
                img2 = train_sp[batch_idx].cuda()
                # img = train_total[batch_idx].unsqueeze(1).cuda()
                img_out, _, _ = model(img1, img2)
                # compute loss
                loss, p1, p2, p3 = my_loss_fn(img1, img2, img_out)
                # loss = l1_loss(img, img_out)
                # compute loss
                # loss, _, _, _ = my_loss_fn(vgg, img_out, img, opt.lambda1, opt.lambda2, config.block_idx, device)
                # back propagate and update weights
                # print(loss, p1, p2, p3)
                # print("batch reg, grad, percep loss: ", reg_loss.item(), img_grad.item(), percep.item())
                # loss = loss / NUM_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()

                # if ((j + 1) % NUM_ACCUMULATION_STEPS == 0) or (j + 1 == len(train_loader)):
            # optimizer.step()
            b_loss += loss.item()
            # wandb.log({"loss": loss})

        # store loss
        ave_loss = b_loss / len(train_loader_ct)
        train_loss.append(ave_loss)
        print("epoch {}, training loss is: {}".format(i, ave_loss))

        # validation
        val_loss = []
        val_display_img = []
        with torch.no_grad():
            b_loss = 0
            # eval model, unable update weights
            model.eval()
            for k, batch_idx in enumerate(val_loader_ct):
                batch_idx = batch_idx.view(-1).long()
                img1 = train_mri[batch_idx].cuda()
                img2 = train_sp[batch_idx].cuda()
                img_out, _, _ = model(img1, img2)
                #img = train_total[batch_idx].unsqueeze(1).cuda()
                #img_out = model(img) #model(img1, img2)  _, _              
                # compute loss
                loss, p1, p2, p3 = my_loss_fn(img1, img2, img_out)
                #loss = l1_loss(img, img_out)
                # print(loss, p1, p2, p3)
                b_loss += loss.item()

        ave_val_loss = b_loss / len(val_loader_ct)
        val_loss.append(ave_val_loss)
        print("epoch {}, validation loss is: {}".format(i, ave_val_loss))

        # save model
        if ave_val_loss < lowest_val_loss:
            # torch.save(model.state_dict(), model_dir + "/model_at_{}.pt".format(i))
            lowest_val_loss = ave_val_loss
            print("model is saved in epoch {}".format(i))

            # Evaluate during training
            # Save the current model
            torch.save(model.state_dict(), model_dir + f"/tmp_best_at_{i}.pt")
    

        if (i+1) % val_every == 0:
            print(f"validate at val_every {val_every}")
            torch.save(model.state_dict(), model_dir + f"/val_every_at_{i}.pt")
            val_psnr, val_ssim, val_nmi, val_mi, val_fsim, val_en = validate(model_dir + f"/val_every_at_{i}.pt", fold_path, exp_folder_name, exp, is_spect=True)
            
        if i == opt.epochs - 1:
            torch.save(model.state_dict(), model_dir + "/last.pt".format(i))

        # lr decay update
        if opt.lr_decay:
            stepLR.step()
        torch.cuda.empty_cache()
    ########################################
