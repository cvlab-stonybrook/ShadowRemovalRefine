import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F


def RGBdist(inside, outside):
    shadowed_values = inside[:, :10000,:]
    shadfree_values = outside[:,:10000,:]
    loss = 0
    dist_min = torch.min(torch.cdist(shadowed_values, shadfree_values, p=2), dim=-1)[0][0]
    loss += dist_min.mean()
    return loss

def differentiable_histogram(x, bins=256, min=0.0, max=1.0):
    x = x.transpose(1,2).squeeze(0)
    n_chns, n_pixs = x.shape
    hist_torch = torch.zeros(n_chns, bins).to(x.device)
    delta = (max - min) / bins
    bin_table = torch.arange(start=0, end=bins+1, step=1).view(-1,1,1) * delta
    bin_table = bin_table.to(x.device)
    for bin in range(0, bins, 1):
        h_r = bin_table[bin].item() if bin > 0 else 0
        h_r_sub_1 = bin_table[bin - 1].item() if bin > 0 else 0
        h_r_plus_1 = bin_table[bin + 1].item()

        mask_sub = ((h_r > x) & (x >= h_r_sub_1)).float().to(x.device)
        mask_plus = ((h_r_plus_1 > x) & (x >= h_r)).float().to(x.device)
        hist_torch[:,bin] = torch.sum(((x - h_r_sub_1) * mask_sub), dim=-1) + torch.sum(((h_r_plus_1 - x) * mask_plus), dim=-1)
    return (hist_torch / hist_torch.sum(axis=-1, keepdim=True)).view(-1)

def EMDLoss(pred, target):
    return torch.mean(torch.square(torch.cumsum(target, dim=-1) - torch.cumsum(pred, dim=-1)))

def distribution_loss(inside, outside):
    output = inside
    bg = outside
    total = 0
    for i in range(output.shape[2]):
        hcdiff_in = differentiable_histogram(output[:,:,i:i+1], bins=256, min=0.0, max=1.0)
        hcdiff_out = differentiable_histogram(bg[:,:,i:i+1], bins=256, min=0.0, max=1.0)
        total += EMDLoss(hcdiff_in,hcdiff_out)
    return total

def non_shadow_loss(out, img, msk):
    out_non = out * msk
    img_non = img * msk
    return torch.nn.MSELoss()(out_non, img_non)

def all_patch_pips_loss(loss_fn, num_patch, size_patch, outside_m, innside_m, img):
    loss_total = 0
    num_m = 0
    len_crop = size_patch // 2
    for coord_out, coord_inn in zip(outside_m, innside_m):
        num_m += 1
        if coord_out is None or coord_inn is None:
            loss_total += 0
        else:
            randp_out = random.choices(coord_out, k=num_patch)
            randp_inn = random.choices(coord_inn, k=num_patch)
            patch_inn = torch.zeros(num_patch,3,64,64)
            patch_out = torch.zeros(num_patch,3,64,64)
            for i in range(num_patch):
                p_i = img[:,:,randp_inn[i][0]-len_crop:randp_inn[i][0]+len_crop,randp_inn[i][1]-len_crop:randp_inn[i][1]+len_crop]
                p_i = p_i*2-1
                p_i = F.interpolate(p_i,64)
                patch_inn[i] = p_i
                p_o = img[:,:,randp_out[i][0]-len_crop:randp_out[i][0]+len_crop,randp_out[i][1]-len_crop:randp_out[i][1]+len_crop]
                p_o = p_o*2-1
                p_o = F.interpolate(p_o,64)
                patch_out[i] = p_o
            inn = patch_inn.repeat_interleave(num_patch,dim=0)
            out = patch_out.repeat(num_patch,1,1,1)
            loss = loss_fn(inn,out.detach())
            loss_total += loss.view(num_patch,num_patch).min(dim=1).values.mean()
    return loss_total / num_m

def per_patch_pips_loss(loss_fn, num_patch, coord_out, coord_inn, img):
    if coord_out is None or coord_inn is None:
        return 0
    else:
        randp_out = random.choices(coord_out, k=num_patch)
        randp_inn = random.choices(coord_inn, k=num_patch)
        patch_inn = torch.zeros(num_patch,3,64,64)
        patch_out = torch.zeros(num_patch,3,64,64)
        for i in range(num_patch):
            p_i = img[:,:,randp_inn[i][0]-8:randp_inn[i][0]+8,randp_inn[i][1]-8:randp_inn[i][1]+8]
            p_i = p_i*2-1
            p_i = F.interpolate(p_i,64)
            patch_inn[i] = p_i
            p_o = img[:,:,randp_out[i][0]-8:randp_out[i][0]+8,randp_out[i][1]-8:randp_out[i][1]+8]
            p_o = p_o*2-1
            p_o = F.interpolate(p_o,64)
            patch_out[i] = p_o
        inn = patch_inn.repeat_interleave(num_patch,dim=0)
        out = patch_out.repeat(num_patch,1,1,1)
        loss = loss_fn(inn,out.detach())
        return loss.view(num_patch,num_patch).min(dim=1).values.mean()

def minimum_filter(n, img):
  size = (n, n)
  shape = cv2.MORPH_ELLIPSE
  kernel = cv2.getStructuringElement(shape, size)
  imgResult = cv2.erode(img, kernel)
  return imgResult

def maximum_filter(n, img):
  size = (n, n)
  shape = cv2.MORPH_ELLIPSE
  kernel = cv2.getStructuringElement(shape, size)
  imgResult = cv2.dilate(img, kernel)
  return imgResult

def remove_outlier(selected_edge, threshold=100):
    edge = (selected_edge*255).astype(np.uint8)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < threshold:
            cv2.fillPoly(selected_edge, [cnt], 0)
    return selected_edge
    
def SAM_findedge(mask_generator, image, msk, size_patch):
    extend = maximum_filter(7,msk)
    shrink = minimum_filter(7,msk)
    out = maximum_filter(15,msk)
    inn = minimum_filter(15,msk)
    ot_ring = out - extend
    in_ring = shrink - inn
    ot_ring = remove_outlier(ot_ring)
    in_ring = remove_outlier(in_ring)
    tmp1 = maximum_filter(23,in_ring)+ot_ring
    tmp2 = maximum_filter(23,ot_ring)+in_ring
    ot_ring = np.where(tmp1==2, 1, 0)
    in_ring = np.where(tmp2==2, 1, 0)
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    outer = np.zeros_like(msk)
    inner = np.zeros_like(msk)
    outside_m = []
    innside_m = []
    len_crop = size_patch // 2
    boundary = np.zeros_like(msk)
    boundary[len_crop:480-len_crop,len_crop:640-len_crop] = 1
    for i in range(len(masks)):
        sam_mask = masks[i]['segmentation'].astype(np.float32)
        if masks[i]['area'] < 0.005*msk.shape[0]*msk.shape[1] or masks[i]['area'] > 0.95*msk.shape[0]*msk.shape[1]:
            break
        overlap_out = np.where(ot_ring==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        overlap_inn = np.where(in_ring==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        overlap_out = remove_outlier(overlap_out, threshold=10)
        overlap_inn = remove_outlier(overlap_inn, threshold=10)
        overlap_innext = maximum_filter(23,overlap_inn)+overlap_out
        overlap_outsel = np.where(overlap_innext==2, 1, 0)
        overlap_outext = maximum_filter(23,overlap_out)+overlap_inn
        overlap_innsel = np.where(overlap_outext==2, 1, 0)
        if np.sum(overlap_outsel) == 0 or np.sum(overlap_outsel) == 0:
            continue
        outer[overlap_outsel==1] = 1
        inner[overlap_innsel==1] = 1
        matt_out = np.where((1-msk)==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        matt_inn = np.where(msk==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        matt_out = minimum_filter(15,matt_out)*boundary
        matt_inn = minimum_filter(15,matt_inn)*boundary
        if np.sum(matt_out) < 256 or np.sum(matt_inn) < 256:
            outside_m.append(None)
            innside_m.append(None)
        else:
            outside_m.append(np.column_stack(np.where(matt_out == 1)))
            innside_m.append(np.column_stack(np.where(matt_inn == 1)))
    if np.sum(outer) == 0 or np.sum(inner) == 0:
        outer = ot_ring
        inner = in_ring
        matt_out = minimum_filter(15,(1-msk))*boundary
        matt_inn = minimum_filter(15,msk)*boundary
        if np.sum(matt_out) < 256 or np.sum(matt_inn) < 256:
            outside_m.append(None)
            innside_m.append(None)
        else:
            outside_m.append(np.column_stack(np.where(matt_out == 1)))
            innside_m.append(np.column_stack(np.where(matt_inn == 1)))
    return outer, inner, outside_m, innside_m, extend, shrink

def SAM_findedge_edit(mask_generator, image, msk):
    extend = maximum_filter(7,msk)
    shrink = minimum_filter(7,msk)
    out = maximum_filter(15,msk)
    inn = minimum_filter(15,msk)
    ot_ring = out - extend
    in_ring = shrink - inn
    ot_ring = remove_outlier(ot_ring)
    in_ring = remove_outlier(in_ring)
    tmp1 = maximum_filter(23,in_ring)+ot_ring
    tmp2 = maximum_filter(23,ot_ring)+in_ring
    ot_ring = np.where(tmp1==2, 1, 0)
    in_ring = np.where(tmp2==2, 1, 0)
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    outside = []
    innside = []
    outside_m = []
    innside_m = []
    boundary = np.zeros_like(msk)
    boundary[8:472,8:632] = 1
    for i in range(len(masks)):
        sam_mask = masks[i]['segmentation'].astype(np.float32)
        if masks[i]['area'] < 0.005*msk.shape[0]*msk.shape[1] or masks[i]['area'] > 0.95*msk.shape[0]*msk.shape[1]:
            break
        overlap_out = np.where(ot_ring==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        overlap_inn = np.where(in_ring==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        overlap_out = remove_outlier(overlap_out, threshold=10)
        overlap_inn = remove_outlier(overlap_inn, threshold=10)
        overlap_innext = maximum_filter(23,overlap_inn)+overlap_out
        overlap_outext = maximum_filter(23,overlap_out)+overlap_inn
        overlap_outsel = np.where(overlap_innext==2, 1, 0)
        overlap_innsel = np.where(overlap_outext==2, 1, 0)
        if np.sum(overlap_outsel) == 0 or np.sum(overlap_outsel) == 0:
            continue
        outside.append(np.where(overlap_outsel==1))
        innside.append(np.where(overlap_innsel==1))
        matt_out = np.where((1-msk)==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        matt_inn = np.where(msk==sam_mask, np.ones_like(msk), np.zeros_like(msk))*sam_mask
        matt_out = minimum_filter(15,matt_out)*boundary
        matt_inn = minimum_filter(15,matt_inn)*boundary
        if np.sum(matt_out) < 256 or np.sum(matt_inn) < 256:
            outside_m.append(None)
            innside_m.append(None)
        else:
            outside_m.append(np.column_stack(np.where(matt_out == 1)))
            innside_m.append(np.column_stack(np.where(matt_inn == 1)))
    if len(outside) == 0 or len(innside) == 0:
        outside.append(np.where(ot_ring==1))
        innside.append(np.where(in_ring==1))
        matt_out = minimum_filter(15,(1-msk))*boundary
        matt_inn = minimum_filter(15,msk)*boundary
        if np.sum(matt_out) < 256 or np.sum(matt_inn) < 256:
            outside_m.append(None)
            innside_m.append(None)
        else:
            outside_m.append(np.column_stack(np.where(matt_out == 1)))
            innside_m.append(np.column_stack(np.where(matt_inn == 1)))
    return outside, innside, outside_m, innside_m, out, inn

def full_findedge(msk):
    inring = minimum_filter(7,msk)
    ouring = maximum_filter(7,msk)
    shrinkmsk = minimum_filter(15,msk)
    extendmsk = maximum_filter(15,msk)
    stepin = inring - shrinkmsk
    stepout= extendmsk - ouring
    return stepout, stepin, extendmsk, shrinkmsk
