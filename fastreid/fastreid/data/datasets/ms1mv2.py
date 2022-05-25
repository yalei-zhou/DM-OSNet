# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
#大规模人脸数据集

@DATASET_REGISTRY.register()
class MS1MV2_V1(ImageDataset):#从图片中读取数据
    dataset_dir = "MS_Celeb_1M_V1"
    dataset_name = "ms1mv2_V1"

    def __init__(self, root="/home/zyl/fast-reid/datasets", **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        train = self.process_dirs("train")[:10000]
        query = self.process_dirs("query")[10000:11000:3]
        gallary1 = self.process_dirs("gallay")[10000:11000]
        gallary = []
        for ga in gallary1:
            ss = ga[:2]
            ss.append(0)
            if ss in query:
                continue
            else:
               gallary.append(ga)
                # train = self.process_dirs()

        super().__init__(train, query, gallary, **kwargs)

    def process_dirs(self,flag):
        train_list = []

        fid_list = os.listdir(self.dataset_dir)

        for fid in fid_list:
            all_imgs = glob.glob(os.path.join(self.dataset_dir, fid, "*.jpg"))
            if flag=="train":
                for img_path in all_imgs:
                    train_list.append([img_path, self.dataset_name + '_' + fid, '0'])
            elif flag == "query":
                for img_path in all_imgs:
                    train_list.append([img_path, int(fid), 0])
            elif flag == "gallay":
                for img_path in all_imgs:
                    train_list.append([img_path, int(fid), 1])

        return train_list


@DATASET_REGISTRY.register()
class MS1MV2_V2(ImageDataset):#从rec中读取数据
    dataset_dir = "MS_Celeb_1M_V2"
    dataset_name = "ms1mv2_V2"

    def __init__(self, root="/home/zyl/fast-reid/datasets", **kwargs):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        train = self.process_dirs("train")[:-3000]
        query = self.process_dirs("query")[-3000::3]
        gallary1 = self.process_dirs("gallay")[-3000:]
        gallary = []
        for ga in gallary1:
            ss = ga[:2]
            ss.append(0)
            if ss in query:
                continue
            else:
               gallary.append(ga)
                # train = self.process_dirs()

        super().__init__(train, query, gallary, **kwargs)

    def process_dirs(self,flag):
        train_list = []

        fid_list = os.listdir(self.dataset_dir)

        for fid in fid_list:
            all_imgs = glob.glob(os.path.join(self.dataset_dir, fid, "*.jpg"))
            if flag=="train":
                for img_path in all_imgs:
                    train_list.append([img_path, self.dataset_name + '_' + fid, '0'])
            elif flag == "query":
                for img_path in all_imgs:
                    train_list.append([img_path, int(fid), 0])
            elif flag == "gallay":
                for img_path in all_imgs:
                    train_list.append([img_path, int(fid), 1])

        return train_list
