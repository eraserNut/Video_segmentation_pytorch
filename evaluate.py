import numpy as np
import os
from PIL import Image
from misc import check_mkdir, cal_precision_recall_mae, AvgMeter, cal_fmeasure, crf_refine

root_path = '/home/ext/chenzhihao/code/video_shadow/models/PDBM/predict'
save_path = '/home/ext/chenzhihao/code/video_shadow/models/PDBM/predict_fuse'
gt_path = '/home/ext/chenzhihao/Datasets/saliency_dataset/DAVIS_val/labels'
input_path = '/home/ext/chenzhihao/Datasets/saliency_dataset/DAVIS_val/images'
CRF_able = 'True'
if CRF_able:
    save_path = save_path + '_CRF'

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()

video_list = os.listdir(root_path)
for video in video_list:
    img_list = os.listdir(os.path.join(root_path, video))  # include overlap images
    img_set = list(set([img.split('_', 1)[0] for img in img_list]))  # remove repeat
    for img_prefix in img_set:
        imgs = [img for img in img_list if img.split('_', 1)[0] == img_prefix]  # imgs waited for fuse
        fuse = []
        for img_path in imgs:
            img = np.array(Image.open(os.path.join(root_path, video, img_path)).convert('L')).astype(np.float32)
            fuse.append(img)
        fuse = (sum(fuse) / len(imgs)).astype(np.uint8)
        # CRF
        if CRF_able:
            input = Image.open(os.path.join(input_path, video, img_prefix+'.jpg'))
            fuse = crf_refine(np.array(input.convert('RGB')), fuse)
        # calculate metric
        gt = np.array(Image.open(os.path.join(gt_path, video, img_prefix+'.png')))
        precision, recall, mae = cal_precision_recall_mae(fuse, gt)
        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)
        # save image
        check_mkdir(os.path.join(save_path, video))
        save_name = '{}.png'.format(img_prefix)
        print(os.path.join(save_path, video, save_name))
        Image.fromarray(fuse).save(
            os.path.join(save_path, video, save_name))
fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                        [rrecord.avg for rrecord in recall_record])

print('MAE:{}, F-beta:{}'.format(mae_record.avg, fmeasure))


