from light_training.loss.eval_metric import *
import time
import os
import argparse
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
# from model import *
# from our_utils import *
from model_mamba.clinicalmamba3D import *
from light_training.loss.my_loss import *

# test_folder = './testset'
save_folder = './res/fused_image'
output_filename = None
cuda = True

########### gpu ###############
device = torch.device("cuda:0" if cuda else "cpu")
###############################

######### make dirs ############
# if not os.path.exists(save_folder):
#     os.mkdir(save_folder)
###############################

####### loading pretrained model ########

#########################################

########### loading test set ###########
# test_ct = torch.load(os.path.join(test_folder, 'ct_test.pt')).to(device)
# test_mri = torch.load(os.path.join(test_folder, 'mri_test.pt')).to(device)

########################################
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



def process(out, cb, cr):
    out_img_y = out
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    cb = cb.clip(0,255)
    cr = cr.clip(0,255)
    # print(out_img_y.shape, cb.shape, cr.shape)
    out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')
    out_img_cb = Image.fromarray(np.uint8(cb), mode = "L")
    out_img_cr = Image.fromarray(np.uint8(cr), mode = "L")
    # out_img_cb = cb#cb.resize(out_img_y.size, Image.BICUBIC)
    # out_img_cr = cr#cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img


# psnr = PeakSignalNoiseRatio()

# for strategy in [ "addition", "average", "FER", "L1NW", "AL1NW", "FL1N"]:
# for strategy in ["average", "max_val", "FER", "FL1N"]:

res_dir = "./my_res"
def validate(model_pt, test_folder, foler_name, exp, is_spect = False):
    model = MambaFuse(dim = 128, H = 32, W = 32, outchannel = 256, depth=5).cuda()
    print("model total params: {}", sum(p.numel() for p in model.parameters()))
    model.load_state_dict(torch.load(model_pt, map_location=device))

    model.eval()
    test_ct = torch.load(os.path.join(test_folder, 'ct_test.pt')).to('cpu')

    test_mri = torch.load(os.path.join(test_folder, 'mri_test.pt')).to('cpu')

    psnrs, ssims, ents = [], [], [], [], [], []
    for slice in range(test_ct.shape[0]):
        ct_slice = test_ct[slice, :, :, :].unsqueeze(0)
        mri_slice = test_mri[slice, :, :, :].unsqueeze(0)

        if is_spect:
            ct_slice = test_ct[slice,0,:,:]      
            cb0 = test_ct[slice,1,:,:]
            cr0 = test_ct[slice,2,:,:]
            ct_slice = ct_slice.detach().cpu().numpy()
            cb0 = cb0.detach().cpu().numpy()
            cr0 = cr0.detach().cpu().numpy()
            out = process(ct_slice, cb0, cr0)
            mri_slice = mri_slice.squeeze(0).squeeze(0).detach().cpu()
            # out.save(f"./my_res/MRISPECT/{foler_name}_{exp}/inf_images/SPECT_{slice}.jpg")
            # io.imsave(f"./my_res/MRISPECT/{foler_name}_{exp}/inf_images/mri_{slice}.jpg", (mri_slice.numpy() * 255).astype(np.uint8))

            ct_slice = test_ct[slice, 0, :, :].unsqueeze(0).unsqueeze(0)
            mri_slice = test_mri[slice, :, :, :].unsqueeze(0)
        # ct_fe = model.fe(ct_slice)
        # print(ct_fe.shape)
        # mri_fe = model.fe(mri_slice)
        start = time.time()
        with torch.no_grad():
            final,_,_ = model(mri_slice, ct_slice)
        print("one slice done, total time: ", time.time() - start)
        # fused = fusion_strategy(ct_fe, mri_fe, device=device, strategy=strategy)
        # final = model.recon(fused)
        final = final.squeeze(0).squeeze(0).detach().cpu().clamp(min=0, max=1.)
        gt1 = ct_slice.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1.)
        gt2 = mri_slice.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1.)
        
        if is_spect:
            out_f = process(final, cb0, cr0)
        # if i > 50:
            out_f.save(os.path.join(f"./my_res/{foler_name}_{exp}/inf_images/", "fused_{}.jpg".format(slice)))
        else:
            # io.imsave(os.path.join(f"./my_res/{foler_name}_{exp}/inf_images", "epoch_{}_mri_{}.jpg".format(i, slice)), (gt2.numpy() * 255).astype(np.uint8))
            # io.imsave(os.path.join(f"./my_res/{foler_name}_{exp}/inf_images", "epoch_{}_ct_{}.jpg".format(i, slice)), (gt1.numpy() * 255).astype(np.uint8))
            # final.save(os.path.join(f"{test_folder}/inf_images", "epoch_{}_fused_{}.jpg".format(i, slice)), (final.numpy() * 255).astype(np.uint8))
            io.imsave(os.path.join(f"./my_res/{foler_name}_{exp}/inf_images", "fused_{}.jpg".format(slice)), (final.numpy() * 255).astype(np.uint8))
        
        psnr_val1 = psnr(final, gt1)
        psnr_val2 = psnr(final, gt2)
        psnr_val = (psnr_val1 + psnr_val2) / 2
        psnrs.append(psnr_val)

        ssim_val1 = ssim(final.unsqueeze(0).unsqueeze(0), gt1.unsqueeze(0).unsqueeze(0))
        ssim_val2 = ssim(final.unsqueeze(0).unsqueeze(0), gt2.unsqueeze(0).unsqueeze(0))
        ssim_val = (ssim_val1 + ssim_val2) / 2
        ssims.append(ssim_val)
        
        ent = en(final)
        ents.append(ent)

    # print(len(psnrs))
    # print(strategy)
    # print(f"Average PSNR: {np.mean(psnrs)}")
    # print(f"Average SSIM: {np.mean(ssims)}")
    # print(f"Average NMI: {np.mean(nmis)}")
    # print(f"Average MI: {np.mean(mis)}")
    # print("---------------------")
    val_psnr = np.mean(psnrs)
    val_ssim = np.mean(ssims)
    val_en = np.mean(ents)
    return val_psnr, val_ssim, val_en


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate the model')
    parser.add_argument('--model_pt', type=str, required=True, help='Path to the model .pt file')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name for saving results')
    parser.add_argument('--exp', type=int, required=True, help='Experiment number')
    parser.add_argument('--is_spect', action='store_true', help='Whether the input is SPECT data')

    args = parser.parse_args()

    metrics = validate(args.model_pt, args.test_folder, args.folder_name, args.exp, is_spect=args.is_spect)
    
# python validation.py --model_pt ./model/dep5_epo200_exp.pt --test_folder ./my_data --folder_name "dep5_epo200_exp" --exp 0 --is_spect