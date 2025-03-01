import os
import numpy as np
from PIL import Image

def EMDLoss(pred, target):
    return np.mean(np.square(np.cumsum(target, axis=-1) - np.cumsum(pred, axis=-1)))

def distribution_loss(inside, outside):
    output = inside
    bg = outside
    total = 0
    for i in range(output.shape[0]):
        hcdiff_in, _ = np.histogram(output[i:i+1,:], bins=256, range=(0,256), density=True)
        hcdiff_out, _ = np.histogram(bg[i:i+1,:], bins=256, range=(0,256), density=True)
        hcdiff_in = hcdiff_in.reshape(1,-1)
        hcdiff_out = hcdiff_out.reshape(1,-1)
        total += EMDLoss(hcdiff_in, hcdiff_out)
    return total / (i + 1)

img_path = 'PATH/TO/test'
msk_path = 'PATH/TO/check'
result_path = 'PATH/TO/results'

diff_cuhk = []
diff_sbu = []

for file in os.listdir(img_path):
    
    img = np.array(Image.open(os.path.join(img_path, file))).astype(np.float32)
    check_msk = np.array(Image.open(os.path.join(msk_path, file[:-4] + '.png'))).astype(np.float32)
    check_msk[check_msk > 0] = 255
    ori_h, ori_w = img.shape[:2]
    result = np.array(Image.open(os.path.join(result_path, file)).resize((ori_w, ori_h))).astype(np.float32)

    x, y = np.ones((ori_h, ori_w)).nonzero()
    pixel_idxs = list(zip(x, y))
    edgein_idx = []
    edgeot_idx = []
    
    for i in range(len(pixel_idxs)):
        if check_msk[pixel_idxs[i][0], pixel_idxs[i][1], 0] == 255:
            edgein_idx.append(i)
        if check_msk[pixel_idxs[i][0], pixel_idxs[i][1], 1] == 255:
            edgeot_idx.append(i)

    img_list = result.transpose(2, 0, 1).reshape(3, -1)
    right_ed = img.transpose(2, 0, 1).reshape(3, -1)
    sdr_values = img_list[:, edgein_idx]
    sfr_values = right_ed[:, edgeot_idx]
    diff = distribution_loss(sdr_values, sfr_values)
    
    if file.startswith('web') or file.startswith('USR'):
        diff_cuhk.append(diff)
    else:
        diff_sbu.append(diff)

print('CUHK mean: %.4f' % np.mean(diff_cuhk))
print('CUHK std: %.4f' % np.std(diff_cuhk))
print('SBU mean: %.4f' % np.mean(diff_sbu))
print('SBU std: %.4f' % np.std(diff_sbu))