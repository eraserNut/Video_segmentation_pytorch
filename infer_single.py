import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms

from config import DAVIS_validation_root
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from networks.PDBM_single import PDBM_single

torch.cuda.set_device(0)

ckpt_path = './models'
exp_name = 'PDBM_single'
args = {
    'snapshot': '50000',
    'scale': 416,
    'input_folder': 'images',
    'label_folder': 'labels'
}

img_transform = transforms.Compose([
    transforms.Resize(args['scale']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

root = DAVIS_validation_root[0]

to_pil = transforms.ToPILImage()


def main():
    net = PDBM_single().cuda()

    if len(args['snapshot']) > 0:
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
        mae_record = AvgMeter()
        video_list = os.listdir(os.path.join(root, args['input_folder']))
        for video in video_list:
            img_list = [img_name for img_name in os.listdir(os.path.join(root, args['input_folder'], video)) if
                        img_name.endswith('.jpg')]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(root, args['input_folder'], video, img_name)).convert('RGB')
                gt = Image.open(os.path.join(root, args['label_folder'], video, os.path.splitext(img_name)[0]+'.png')).convert('L')
                gt = np.array(gt)
                w, h = img.size
                img_var = img_transform(img).unsqueeze(0).cuda()
                res = net(img_var)
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                # prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                # calculate metric
                precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)

                check_mkdir(os.path.join(ckpt_path, exp_name, "predict", video))
                print(os.path.join(ckpt_path, exp_name, "predict", video, img_name))
                Image.fromarray(prediction).save(
                    os.path.join(ckpt_path, exp_name, "predict", video, img_name))
        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])
        print('MAE:{}, F-beta:{}'.format(mae_record.avg, fmeasure))


if __name__ == '__main__':
    main()