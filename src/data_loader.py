from PIL import Image
import os
import copy
import time
import threading
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

"""
"""
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

# 图片加载器
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

"""
数据加载器
- root_path
- transforms
- batch_size
- shuffle
- num_workers
"""
class DataLoader(object):
    def __init__(self, root_path, transforms=None, batch_size=1, shuffle=True, num_workers=1):
        
        # 
        self.work_path = root_path
        # 
        self.transforms = None
        
        if not transforms is None:
            self.transforms = transforms

        # 是否打散数据集
        self.shuffle = shuffle
        # 数据集每次投放大小
        self.batch_size = batch_size
        # num_worker 在加载数据，加载数据快，缺点是内存开销大
        self.num_workers = num_workers
        
        # 定义数据集
        self.database = {}
        self.lable_package = {}
        
        # 初始化图片参数
        self.init_img_parms()
        
        # 图片加载器
        self.loader = pil_loader
        # 创建数据集
        self.create_triplet_db()

        # 数据和标签队列
        self.data_queue = []
        self.lable_queue = []
        

        # 定义数据队列的锁
        self.data_queue_lock = threading.Lock()


        self.data_load_thread = threading.Thread(target=self.data_load)
        self.data_load_thread.start()
        
        self.start = 0
        
        self.end = math.ceil(len(self.triplet_db) / (self.batch_size * 1.0)) - 1
        self.remainder = len(self.triplet_db) % self.batch_size

    # 初始化图像参数
    def init_img_parms(self):

        self.lables = os.listdir(self.work_path)
        self.lables.sort()
        self.lables_map = dict(zip(self.lables, range(len(self.lables))))
        
        for lable in self.lables:
            lable_path = os.path.join(self.work_path, lable)
            if not os.path.isdir(lable_path):
                continue
            self.lable_package[self.lables_map[lable]] = {}
            lable_imgs = os.listdir(lable_path)
            lable_imgs.sort()
            for lable_img in lable_imgs:
                if is_image_file(lable_img):
                    sample = [os.path.join(lable_path, lable_img), self.lables_map[lable]]
                    self.database["%s_%s" % (lable, lable_img)] = sample
                    self.lable_package[self.lables_map[lable]][lable_img] = sample
        self.targets = [d[1] for d in self.database.values()]

    def __getitem__(self, index):
        path, target = self.database[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, target
    
    # 创建三元组数据
    def create_triplet_db(self):
        
        self.triplet_db = []
        
        anchor_db = copy.deepcopy(self.lable_package)
        positive_db = copy.deepcopy(self.lable_package)
        negative_db = copy.deepcopy(self.database)

        anchor_db_keys = list(anchor_db.keys())

        if self.shuffle:
            random.shuffle(anchor_db_keys)

        for lable in anchor_db_keys:
            lable_samples_anchor = anchor_db[lable]
            lable_samples_positive = positive_db[lable]
            if len(lable_samples_anchor) == 0:
                continue
    def data_load(self):
        big_batch_pool_szie = self.batch_size * 3 if self.batch_size * 3 > 10 else 10
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        while True:
            if len(self.data_queue) > big_batch_pool_szie:
                time.sleep(0.4)
            else:
