import os
import copy
import argparse
from collections import OrderedDict
from PIL import Image
import numpy as np
import time
import lpips

import torch
import torch.optim as optim
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from util import *

from model import ShadowFormer
from timm.utils import NativeScaler


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location="cuda")
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


parser = argparse.ArgumentParser(description='RGB refinement')
# options
parser.add_argument('--result_dir', default='./result', type=str, help='Directory for results')
parser.add_argument('--dataset', type=str, default='ISTD', help='image dataset to test, [ISTD, Proposed]')
parser.add_argument('--prior_checkpoint', type=str, default='/data/add_disk1/shilin/Data/ShadowRemovalRefine/ISTD_plus_model_latest.pth', help='Prior model checkpoint')
parser.add_argument('--sam_model', type=str, default='vit_b', help='SAM model type')
parser.add_argument('--sam_checkpoint', type=str, default='/data/add_disk1/shilin/ShadowRemovalRefine/finetune_SAM/pointmultimaskseploss_lr0.0001_bs1_lossdicemse_ep100_nspi32/60th_epoch.pth', help='SAM model checkpoint')
parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
# hyperparameter
parser.add_argument('--refine_method', type=str, default='select_all', help='refine method to use, [full, select_all, select_per]')
parser.add_argument('--loss_control', type=str, default='all', help='which losses to use, [all, distance, distribution, nonshadow, patch]')
parser.add_argument('--lr_refine', type=float, default=0.00001, help='refine learning rate')
parser.add_argument('--num_iter', type=int, default=10, help='refine number of iter')
parser.add_argument('--update_policy', type=str, default='whole', help='how to update the network')
parser.add_argument('--num_patch', type=int, default=8, help='number of sampled patches')
parser.add_argument('--size_patch', type=int, default=16, help='size of sampled patches')
# shadowformer
parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')

args = parser.parse_args()


print(args.dataset, args.refine_method, args.loss_control)
if args.refine_method == 'select_all':
    mkdir(os.path.join(args.result_dir, 'refineall'+'_'+args.dataset+args.loss_control+args.update_policy+str(args.num_iter)+str(args.lr_refine)))
elif args.refine_method == 'select_per':
    mkdir(os.path.join(args.result_dir, 'refineper'+'_'+args.dataset+args.loss_control+args.update_policy+str(args.num_iter)+str(args.lr_refine)))
elif args.refine_method == 'full':
    mkdir(os.path.join(args.result_dir, 'full'+'_'+args.dataset+args.loss_control+args.update_policy+str(args.num_iter)+str(args.lr_refine)))

sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint)
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.90,
    stability_score_thresh=0.90,
    min_mask_region_area=500,
)

if args.dataset == 'ISTD':
    path = '/data/add_disk1/shilin/Data/ISTD_Dataset/test/test_A'
elif args.dataset == 'Proposed':
    path = '/data/add_disk1/shilin/Data/ShadowRemovalRefine/ProposedDataset/test_A'

torch.manual_seed(321)
model_initial = ShadowFormer(img_size=args.train_ps,embed_dim=args.embed_dim,win_size=args.win_size, 
                                 token_projection=args.token_projection,token_mlp=args.token_mlp)
load_checkpoint(model_initial, args.prior_checkpoint)
loss_scaler = NativeScaler()

if args.refine_method != 'full':
    loss_fn_alex = lpips.LPIPS(net='alex')

num_img = 0
train_time = 0
sample_time = 0
preprocess_time = 0
print('Start...')

for file in os.listdir(path):
    if file.endswith('.png') or file.endswith('.jpg'):
        if args.dataset == 'ISTD':
            img_path = os.path.join(path, file)
            msk_path = img_path.replace('test_A', 'test_B')
            trg_path = img_path.replace('test_A', 'test_C_fixed_official')
        if args.dataset == 'Proposed':
            img_path = os.path.join(path, file)
            msk_path = img_path.replace('test_A', 'test_B')
            msk_path = msk_path[:-4] + '.png'

        preprocess_start = time.time()
        w, h = 640, 480
        img = np.array(Image.open(img_path).resize((w,h)).convert('RGB')).astype(np.float32) / 255
        msk = np.array(Image.open(msk_path).resize((w,h)).convert('L')).astype(np.float32) / 255
        msk[msk>=0.5] = 1
        msk[msk<=0.5] = 0

        rgb_noisy = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        mask = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)
        
        model_restoration = copy.deepcopy(model_initial).to("cuda")
        image = np.array(Image.open(img_path).resize((w,h)))

        preprocess_interstop = time.time()
        sampling_start = time.time()

        if args.refine_method == 'select_per':
            outside, innside, outside_m, innside_m, _, _ = SAM_findedge_edit(mask_generator, image, msk)
        else:
            if args.refine_method == 'select_all':
                stepout, stepin, outside_m, innside_m, _, _ = SAM_findedge(mask_generator, image, msk, args.size_patch)
            else:
                stepout, stepin, _, _ = full_findedge(msk)
            x,y = np.ones((h,w)).nonzero()
            pixel_idxs = list(zip(x,y))
            edgein_idx = []
            edgeot_idx = []
            for i in range(len(pixel_idxs)):
                if stepin[pixel_idxs[i][0],pixel_idxs[i][1]] == 1:
                    edgein_idx.append(i)
                if stepout[pixel_idxs[i][0],pixel_idxs[i][1]] == 1:
                    edgeot_idx.append(i)

        sampling_end = time.time()
        preprocess_interstart = time.time()

        rgb_noisy = rgb_noisy.to("cuda")
        mask = mask.to("cuda")
        if args.update_policy == 'whole':
            update_params = model_restoration.parameters()
        else:
            update_params = model_restoration.decoderlayer_2.parameters()
        optimizer = optim.AdamW(update_params, lr=args.lr_refine, betas=(0.9, 0.999),eps=1e-8, weight_decay=0.02)

        preprocess_stop = time.time()
        start_time = time.time()
        
        model_restoration.train()
        for i in range(args.num_iter):
            rgb_restored = model_restoration(rgb_noisy, mask)
            rgb_restored = torch.clamp(rgb_restored, 0, 1)
            
            if args.refine_method == 'full':
                sdr_values = rgb_restored.view(1,3,-1)[:,:,edgein_idx].transpose(1,2)
                sfr_values = rgb_noisy.view(1,3,-1)[:,:,edgeot_idx].transpose(1,2).detach()
                loss1 = RGBdist(sdr_values,sfr_values)
                loss2 = distribution_loss(sdr_values,sfr_values)
                loss_p = 0
            elif args.refine_method == 'select_per':
                loss1, loss2, loss_p = 0,0,0
                for j in range(len(outside)):
                    sdr_values = rgb_restored[:,:,innside[j][0],innside[j][1]].transpose(1,2)
                    sfr_values = rgb_noisy[:,:,outside[j][0],outside[j][1]].transpose(1,2).detach()
                    loss1 += RGBdist(sdr_values,sfr_values)
                    loss2 += distribution_loss(sdr_values,sfr_values)
                    loss_p += per_patch_pips_loss(loss_fn_alex, 8, outside_m[j], innside_m[j], rgb_restored)
                loss1 /= j+1
                loss2 /= j+1
                loss_p /= j+1
            else:
                sdr_values = rgb_restored.view(1,3,-1)[:,:,edgein_idx].transpose(1,2)
                sfr_values = rgb_noisy.view(1,3,-1)[:,:,edgeot_idx].transpose(1,2).detach()
                loss1 = RGBdist(sdr_values,sfr_values)
                loss2 = distribution_loss(sdr_values,sfr_values)
                loss_p = all_patch_pips_loss(loss_fn_alex, args.num_patch, args.size_patch, outside_m, innside_m, rgb_restored)
            
            loss3 = non_shadow_loss(rgb_restored, rgb_noisy, 1-mask)

            if args.loss_control == 'all':
                loss = 1*loss1+1*loss2+10*loss3+0.1*loss_p
            elif args.loss_control == 'distribution':
                loss = 1*loss1+10*loss3+0.1*loss_p
            elif args.loss_control == 'distance':
                loss = 1*loss2+10*loss3+0.1*loss_p
            elif args.loss_control == 'nonshadow':
                loss = 1*loss1+1*loss2+0.1*loss_p
            elif args.loss_control == 'patch':
                loss = 1*loss1+1*loss2+10*loss3
            
            loss_scaler(loss, optimizer, parameters=update_params)
        
        end_time = time.time()
        train_time += end_time - start_time
        sample_time += sampling_end - sampling_start
        preprocess_time += (preprocess_stop - preprocess_interstart) + (preprocess_interstop - preprocess_start)
        num_img += 1
        
        with torch.no_grad():
            model_restoration.eval()
            rgb_restored = model_restoration(rgb_noisy, mask)
            rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        rgb_restored = (rgb_restored*255).astype(np.uint8)
        refine_img = Image.fromarray(rgb_restored).resize((w,h))
        if args.refine_method == 'select_all':
            refine_img.save(os.path.join(args.result_dir, 'refineall'+'_'+args.dataset+args.loss_control+args.update_policy+str(args.num_iter)+str(args.lr_refine), file))
        elif args.refine_method == 'select_per':
            refine_img.save(os.path.join(args.result_dir, 'refineper'+'_'+args.dataset+args.loss_control+args.update_policy+str(args.num_iter)+str(args.lr_refine), file))
        elif args.refine_method == 'full':
            refine_img.save(os.path.join(args.result_dir, 'full'+'_'+args.dataset+args.loss_control+args.update_policy+str(args.num_iter)+str(args.lr_refine), file))
        
        if args.verbose:
            print(file)

print("Done!")
print("Avg train time: %.3f" % (train_time/num_img))
print("Avg sample time: %.3f" % (sample_time/num_img))
print("Avg preprocess time: %.3f" % (preprocess_time/num_img))
