import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms

from config import DAVIS_validation_root
from misc import check_mkdir
from networks.PDBM import PDBM

torch.cuda.set_device(0)

ckpt_path = './models'
exp_name = 'PDBM'
args = {
    'snapshot': '50000',
    'scale': 416,
    'window_size': 3,
    'dilation_list': [2, 3],
    'input_folder': 'images'
}

img_transform = transforms.Compose([
    transforms.Resize(args['scale']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

root = DAVIS_validation_root[0]

to_pil = transforms.ToPILImage()


def main():
    net = PDBM().cuda()

    if len(args['snapshot']) > 0:
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()
    with torch.no_grad():
        video_list = os.listdir(os.path.join(root, args['input_folder']))
        for video in video_list:
            img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, args['input_folder'], video)) if
                        f.endswith('.jpg')]
            img_list = sortImg(img_list)
            for dilation in args['dilation_list']:
                clip_list = img2clip(img_list, dilation)
                for idx, clip in enumerate(clip_list):
                    seq = []
                    for img_name in clip:
                        img = Image.open(os.path.join(root, args['input_folder'], video, img_name+'.jpg')).convert('RGB')
                        w, h = img.size  # assume that same video has same image size
                        img_tensor = img_transform(img).unsqueeze(0).cuda()  # (B=1, C, H, W)
                        seq.append(img_tensor)
                    seq = torch.stack(seq, dim=1)  # (B=1, T, C, H, W)
                    predicts = net(seq)  # T length (B=1, C=1, H, W)
                    for jdx, img_name in enumerate(clip):
                        prediction = np.array(transforms.Resize((h, w))(to_pil(torch.sigmoid(predicts[jdx]).squeeze(0).cpu())))
                        # prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                        check_mkdir(os.path.join(ckpt_path, exp_name, "predict", video))
                        # save form as 001_clip2_dilation1.png
                        save_name = '{}_clip{}_dilation{}.png'.format(img_name, jdx+1, dilation)
                        print(os.path.join(ckpt_path, exp_name, "predict", video, save_name))
                        Image.fromarray(prediction).save(
                            os.path.join(ckpt_path, exp_name, "predict", video, save_name))

def img2clip(img_list, dilation):
    idx = 0
    clip_list = []
    while idx + args['window_size'] * dilation < len(img_list):
        clip = []
        for j in range(0, args['window_size'] * dilation, dilation):
            clip.append(img_list[idx + j])
        clip_list.append(clip)
        idx += 1
    return clip_list



def sortImg(img_list):
    img_int_list = [int(f) for f in img_list]
    sort_index = [i for i, v in sorted(enumerate(img_int_list), key=lambda x: x[1])]  # sort img to 001,002,003...
    return [img_list[i] for i in sort_index]


if __name__ == '__main__':
    main()
