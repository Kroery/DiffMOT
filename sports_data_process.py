import os.path as osp
import os
import numpy as np


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


seq_root = '/mnt/8T/home/estar/data/sportsmot/sportsmot_publish/dataset/train'
label_root = '/mnt/8T/home/estar/data/sportsmot/sportsmot_publish/dataset/trackers_gt_t/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
for seq in seqs:
    print(seq)
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    idx = np.lexsort(gt.T[:2, :])
    gt = gt[idx, :]

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, cls, vis in gt:
        if mark == 0 or not cls == 1:
            continue
        fid = int(fid)
        tid = int(tid)

        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(tid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            fid, x / seq_width, y / seq_height, w / seq_width, h / seq_height, vis)
        with open(label_fpath, 'a') as f:
            f.write(label_str)
