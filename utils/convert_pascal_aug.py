import os.path as osp
import os
import scipy.io as scio
import cv2
from PIL import Image
from tqdm import tqdm



def parse_pascal_voc_aug(pth):
    out_pth = osp.join(pth, 'VOC_AUG')
    ds_pth = osp.join(pth, 'benchmark_RELEASE/dataset')
    labels_pth = osp.join(ds_pth, 'cls')

    lbmats = os.listdir(labels_pth)
    print('converting mat files to png images')
    for lbmat in tqdm(lbmats):
        mat_pth = osp.join(labels_pth, lbmat)
        mat = scio.loadmat(mat_pth,
                mat_dtype = True,
                squeeze_me = True,
                struct_as_record = False)
        lb_arr = mat['GTcls'].Segmentation
        lb_name = osp.splitext(lbmat)[0]
        lb_fn = '{}.png'.format(lb_name)
        lb_save_pth = osp.join(out_pth, 'labels', lb_fn)
        lb = Image.fromarray(lb_arr)
        lb.save(lb_save_pth)


if __name__ == '__main__':
    parse_pascal_voc_aug('./data')
