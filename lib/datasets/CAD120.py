import os

import json_tricks as json
import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
from utils.transforms import fliplr_joints

class CAD120(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=16,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(CAD120, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.num_joints = 15
        self.image_size = np.array([256,256])
        self.heatmap_size = np.array([64,64])
        self.sigma = 2
        #self.class_weights = torch.FloatTensor([1]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.flip_pairs = [[3, 5], [4, 6], [7, 9], [8, 10], [11, 13], [12, 14]]
        self.center_crop_test = center_crop_test
        
        self.img_list = [line.strip() for line in open(root+list_path)]

        self.annot_t, self.annot_v = self.load_annotations()
        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
    
    def load_annotations(self):

        with open('data/CAD120/CAD120_keypoints_train_coco_style.json', 'r') as f:
            annot_t = json.load(f)

        with open('data/CAD120/CAD120_keypoints_val_coco_style.json', 'r') as f:
            annot_v = json.load(f)

        return annot_t, annot_v

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for img_id in self.img_list:
                num = int(img_id.split('_')[1])
                if 'train' in img_id:
                    files.append({
                        "img_rgb": self.annot_t["images"][num]["file_name"],
                        "img_depth": self.annot_t["images"][num]["depth_name"],
                        "name": img_id,
                        "weight": 1
                    })
                else:
                    files.append({
                        "img_rgb": self.annot_v["images"][num]["file_name"],
                        "img_depth": self.annot_v["images"][num]["depth_name"],
                        "name": img_id,
                        "weight": 1
                    })
        return files

        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        image_id = item["name"]
        num = int(image_id.split('_')[1])
        image_rgb = cv2.imread(item["img_rgb"],cv2.IMREAD_COLOR)
        label = cv2.imread(item["img_depth"],cv2.IMREAD_GRAYSCALE)
        size = image_rgb.shape

        if 'train' in image_id:
            bbox = list(map(int,self.annot_t["annotations"][num]["bbox"]))
            keypoints = self.annot_t["annotations"][num]["keypoints"]
        else:
            bbox = list(map(int,self.annot_v["annotations"][num]["bbox"]))
            keypoints = self.annot_v["annotations"][num]["keypoints"]

        #crop
        image_rgb = image_rgb[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        label = label[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]

        #resize
        image_rgb = cv2.resize(image_rgb, (256,256), interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, (256,256), interpolation = cv2.INTER_LINEAR)

        #one line
        image_depth = np.zeros((256,256))
        image_depth[:,128] = label[:,128]

        label = cv2.resize(label, (64,64), interpolation = cv2.INTER_LINEAR)

        if 'test' in self.list_path:
            image_rgb = self.input_transform(image_rgb)
            image_depth = self.label_transform(image_depth)
            image_rgb = image_rgb.transpose((2, 0, 1))

            return image_rgb.copy(), image_depth.copy(), np.array(size), image_id

        image_rgb, image_depth, label = self.gen_sample(image_rgb, image_depth, label)

        joints = np.array([[keypoints[3*i],keypoints[3*i+1],0] for i in range(15)])
        joints_vis = np.array([[keypoints[3*i+2],keypoints[3*i+2],0] for i in range(15)])

        for i in range(15):
            joints[i,0] = (joints[i,0] - bbox[0])*256/bbox[2]
            joints[i,1] = (joints[i,1] - bbox[1])*256/bbox[3]

        image_depth = np.expand_dims(image_depth,axis=0)
        image = np.concatenate((image_rgb, image_depth), axis=0)

        label = np.expand_dims(label,axis=0)

        if self.flip:
            flip = np.random.choice(2)*2-1
            image = image[:,:,::flip]
            label = label[:,:,::flip]
            joints, joints_vis = fliplr_joints(joints, joints_vis, 256, self.flip_pairs)

        target, target_weight = self.generate_target(joints, joints_vis)
        target = np.concatenate((target,label),axis=0)

        return image.copy(), target.copy(), target_weight.copy(), np.array(size), image_id, joints.copy(), joints_vis.copy()

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        #if self.use_different_joints_weight:
        #    target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
