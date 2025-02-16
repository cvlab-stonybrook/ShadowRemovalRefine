import os
import numpy as np
import cv2
import random
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg"])

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [640, 480], interpolation=cv2.INTER_AREA)
    return img

def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [640, 480], interpolation=cv2.INTER_AREA)
    return img

def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = torch.sum(y_true * y_pred, dim=(1,2,3))
    sum_of_squares_pred = torch.sum(torch.square(y_pred), dim=(1,2,3))
    sum_of_squares_true = torch.sum(torch.square(y_true), dim=(1,2,3))
    dice = 1 - (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
    return dice.mean()


class DataLoaderTrain(Dataset):
    def __init__(self, istd_dir, srd_dir, num_sample_per_image=20, sam_model=None, device=None):
        super(DataLoaderTrain, self).__init__()
        self.sam_model = sam_model
        self.device = device
        self.num_sample_per_image = num_sample_per_image

        clean_dir1 = 'train_C_fixed_official'
        input_dir1 = 'train_A'
        clean_files1 = sorted(os.listdir(os.path.join(istd_dir, clean_dir1)))
        noisy_files1 = sorted(os.listdir(os.path.join(istd_dir, input_dir1)))

        clean_dir2 = 'shadow_free'
        input_dir2 = 'shadow'
        clean_files2 = sorted(os.listdir(os.path.join(srd_dir, clean_dir2)))
        noisy_files2 = []
        for file in clean_files2:
            file = file.split('_no_shadow')[0]
            file = file + '.jpg'
            noisy_files2.append(file)

        self.clean_filenames = [os.path.join(istd_dir, clean_dir1, x) for x in clean_files1 if is_img_file(x)] + [os.path.join(srd_dir, clean_dir2, x) for x in clean_files2 if is_img_file(x)]
        self.noisy_filenames = [os.path.join(istd_dir, input_dir1, x) for x in noisy_files1 if is_img_file(x)] + [os.path.join(srd_dir, input_dir2, x) for x in noisy_files2 if is_img_file(x)]

        # get the size of target
        self.tar_size = len(self.noisy_filenames)
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        num_p = 16
        wvalues = np.linspace(0, 640, num_p, endpoint=False) 
        hvalues = np.linspace(0, 480, num_p, endpoint=False)
        ww, hh = np.meshgrid(wvalues[1:], hvalues[1:])
        self.positions = np.column_stack([ww.ravel(), hh.ravel()]).astype(int)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        noisy = load_img(self.noisy_filenames[tar_index])
        clean = load_img(self.clean_filenames[tar_index])
        original_image_size = noisy.shape[:2]

        input_points_ = []
        select_pos = np.stack(random.choices(self.positions, k=self.num_sample_per_image))
        for (w,h) in select_pos:
            input_point = np.array([[w, h]])
            input_label = np.array([1])
            point_coord = self.transform.apply_coords(input_point, original_image_size)
            coord_torch = torch.as_tensor(point_coord, dtype=torch.float, device=self.device)
            label_torch = torch.as_tensor(input_label, dtype=torch.int, device=self.device)
            input_points_.append((coord_torch, label_torch))

        # mask = load_mask(self.mask_filenames[tar_index])
        # coords_shad = np.column_stack(np.where(mask == 255))
        # coords_free = np.column_stack(np.where(mask == 0))

        # input_points_ = []
        # for _ in range(self.num_sample_per_image):
        #     if random.random() < self.thres:
        #         y, x = random.choice(coords_shad)
        #     else:
        #         y, x = random.choice(coords_free)
        #     input_point = np.array([[x, y]])
        #     input_label = np.array([1])

        #     point_coord = self.transform.apply_coords(input_point, original_image_size)
        #     coord_torch = torch.as_tensor(point_coord, dtype=torch.float, device=self.device)
        #     label_torch = torch.as_tensor(input_label, dtype=torch.int, device=self.device)
        #     input_points_.append((coord_torch, label_torch))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Transform
        input_noisy = self.transform.apply_image(noisy)
        input_noisy_torch = torch.as_tensor(input_noisy, device=self.device)
        transformed_noisy = input_noisy_torch.permute(2, 0, 1).contiguous()
        input_noisy = self.sam_model.preprocess(transformed_noisy)

        input_clean = self.transform.apply_image(clean)
        input_clean_torch = torch.as_tensor(input_clean, device=self.device)
        transformed_clean = input_clean_torch.permute(2, 0, 1).contiguous()#[None, :, :, :]
        input_clean = self.sam_model.preprocess(transformed_clean)
        input_size = tuple(transformed_noisy.shape[-2:])
        
        return input_noisy, input_points_, input_clean, input_size, original_image_size, clean_filename, noisy_filename


parser = argparse.ArgumentParser(description='Fine tuning SAM')
parser.add_argument('--istd_dir', default='/data/add_disk1/shilin/Data/ISTD_Dataset/train/', type=str, help='ISTD data path')
parser.add_argument('--srd_dir', default='/data/add_disk1/shilin/Data/SRD_Train/', type=str, help='SRD data path')
parser.add_argument('--save_dir', default='/data/add_disk1/shilin/ShadowRemovalRefine/finetune_SAM', type=str, help='model save path')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--default_samckpt', default='/data/add_disk1/shilin/Data/ShadowRemovalRefine/sam_vit_b_01ec64.pth', type=str, help='default sam checkpoint')
parser.add_argument('--default_samtype', default='vit_b', type=str, help='default sam modeltype')
parser.add_argument('--lossfn', default='dicemse', type=str, help='loss function')
parser.add_argument('--lr', default=1e-4, type=float, help='finetune learning rate')
parser.add_argument('--num_sample_per_image', default=32, type=int, help='number of samples in each image')
parser.add_argument('--batch_size', default=1, type=int, help='finetune batch size')
parser.add_argument('--epoch', default=100, type=int, help='finetune training epochs')
parser.add_argument('--save_epoch', default=20, type=int, help='save model every X epochs')
args = parser.parse_args()

experimentname = 'pointmultimaskseploss'+'_lr'+str(args.lr)+'_bs'+str(args.batch_size)+'_loss'+str(args.lossfn)+'_ep'+str(args.epoch)+'_nspi'+str(args.num_sample_per_image)
mkdir(os.path.join(args.save_dir, experimentname))
print('Experiment start:\n%s' % experimentname)

device = 'cuda:%s' % args.gpus if torch.cuda.is_available() else 'cpu'
sam_model = sam_model_registry[args.default_samtype](checkpoint=args.default_samckpt)
sam_model.to(device)
sam_model.train()

fix_model = sam_model_registry[args.default_samtype](checkpoint=args.default_samckpt)
fix_model.to(device)
print('Model initialized on:\n%s' % args.gpus)

optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

full_traindata = DataLoaderTrain(args.istd_dir,args.srd_dir,num_sample_per_image=args.num_sample_per_image,sam_model=sam_model, device=device)
data_loader = DataLoader(full_traindata, batch_size=args.batch_size, shuffle=True)
print('Num of samples: %s' % len(data_loader))
print('Num of points per image: %s' % args.num_sample_per_image)

num_epochs = args.epoch
losses = []
best_loss = 1

for epoch in range(num_epochs):
    epoch_losses = []
    for data in data_loader:
        input_noisy = data[0]
        point_torch_ = data[1]
        input_clean = data[2]
        input_size = data[3]
        original_image_size = data[4]

        loss = 0
        noisy_masks = []
        clean_masks = []
        iou_noisy = []
        iou_clean = []
        with torch.no_grad():
            noisy_embedding = fix_model.image_encoder(input_noisy)
            clean_embedding = fix_model.image_encoder(input_clean)
        for i in range(len(point_torch_)):
            point_torch = point_torch_[i]
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = fix_model.prompt_encoder(
                    points=point_torch,
                    boxes=None,
                    masks=None,
                )
                low_res_masks_clean, iou_pred_clean = fix_model.mask_decoder(
                    image_embeddings=clean_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                )

            low_res_masks_noisy, iou_pred_noisy = sam_model.mask_decoder(
                image_embeddings=noisy_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            noisy_masks.append(low_res_masks_noisy[0])
            clean_masks.append(low_res_masks_clean[0])
            iou_noisy.append(iou_pred_noisy[0])
            iou_clean.append(iou_pred_clean[0])

        noisy_stack = torch.stack(noisy_masks, dim=0)
        clean_stack = torch.stack(clean_masks, dim=0)
        iou_noisy_stack = torch.stack(iou_noisy, dim=0)
        iou_clean_stack = torch.stack(iou_clean, dim=0)
        upscaled_noisy_masks = sam_model.postprocess_masks(noisy_stack, input_size, original_image_size)
        upscaled_noisy_masks = torch.nn.Sigmoid()(upscaled_noisy_masks)
        upscaled_clean_masks = sam_model.postprocess_masks(clean_stack, input_size, original_image_size)
        upscaled_clean_masks = torch.nn.Sigmoid()(upscaled_clean_masks)

        loss += (dice_loss(upscaled_noisy_masks[:,0:1,:,:], upscaled_clean_masks[:,0:1,:,:].detach()) + \
                 dice_loss(upscaled_noisy_masks[:,1:2,:,:], upscaled_clean_masks[:,1:2,:,:].detach()) + \
                 dice_loss(upscaled_noisy_masks[:,2:3,:,:], upscaled_clean_masks[:,2:3,:,:].detach())) /3
        if 'mse' in args.lossfn:
            loss += 0.1* (torch.nn.L1Loss()(iou_noisy_stack, iou_clean_stack.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    losses.append(epoch_losses)
    scheduler.step()
    print(f'EPOCH: {epoch+1}')
    print(f'Mean loss: {np.mean(epoch_losses)}')
    if (epoch+1) % args.save_epoch == 0:
        if np.mean(epoch_losses) < best_loss:
            torch.save(sam_model.state_dict(), os.path.join(args.save_dir, experimentname, 'best_epoch.pth'))
            best_loss = np.mean(epoch_losses)
        torch.save(sam_model.state_dict(), os.path.join(args.save_dir, experimentname, '%sth_epoch.pth' % str(epoch+1)))
        np.save(os.path.join(args.save_dir, experimentname, 'losses.npy'), losses)
