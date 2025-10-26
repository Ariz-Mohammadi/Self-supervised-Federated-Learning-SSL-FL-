# --------------------------------------------------------
# Based on BEiT and MAE code bases
# Pretrain and Finetune datasets for FL SSL.
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------

import numpy as np
import pandas as pd

import os
from .datasets import DataAugmentationForPretrain, build_transform
import torch
from PIL import Image
from skimage.transform import resize
import cv2
import torch.utils.data as data
from torch.utils.data import Dataset

class DatasetFLPretrain(data.Dataset):
    """ data loader for pre-training """
    def __init__(self, args):    
                
        if args.split_type == 'central':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients', 
                                            args.split_type, args.single_client)

        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
        
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                        open(os.path.join(args.data_path, 'labels.csv'))}
    
        self.transform = DataAugmentationForPretrain(args)
        self.args = args
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)
        
        path = os.path.join(self.args.data_path, 'train', self.img_paths[index])
        name = self.img_paths[index]

        target = self.labels[name]
        target = np.asarray(target).astype('int64')
        
        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))
        
        if img.ndim < 3:
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)
            
        return sample, target

    def __len__(self):
        return len(self.img_paths)



class DatasetFLFinetune(Dataset):
    def __init__(self, args, phase='train', transform=None, client_file=None):
        self.args = args
        self.phase = phase
        self.transform = transform
        self.client_file = client_file
        self.img_paths = []
        self.labels = {}
        self.data_cache = {}

        # Define the directory containing client files
        client_dir = os.path.join(args.data_path, f'{args.n_clients}_clients/{args.split_type}')
        if not os.path.exists(client_dir):
            raise FileNotFoundError(f"Client directory {client_dir} not found")

        if phase == 'train':
            # Get all client CSV files for training
            client_files = [f for f in os.listdir(client_dir) if f.endswith('.csv') and f.startswith('client_')]
            if not client_files:
                raise ValueError(f"No client CSV files found in {client_dir}")
            for client_file in client_files:
                client_path = os.path.join(client_dir, client_file)
                with open(client_path, encoding='utf-8-sig') as f:
                    self.img_paths.extend((client_file, line.strip().split(',')[0]) for line in f if line.strip())
        elif phase == 'test':
            # Load test.csv for validation
            test_path = os.path.join(client_dir, 'test.csv')
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file {test_path} not found")
            with open(test_path, encoding='utf-8-sig') as f:
                self.img_paths.extend(('', line.strip().split(',')[0]) for line in f if line.strip())

        # Load labels from labels.csv
        labels_path = os.path.join(args.data_path, 'labels.csv')
        if os.path.exists(labels_path):
            with open(labels_path, encoding='utf-8-sig') as f:
                self.labels = {line.strip().split(',')[0]: int(round(float(line.strip().split(',')[1]))) for line in f if line.strip()}
        else:
            raise FileNotFoundError(f"Labels file {labels_path} not found")

        # Preload and resize data into memory
        print(f"Preloading and resizing {len(self.img_paths)} images for {phase} to 224x224...")
        for client_file, name in self.img_paths:
            if phase == 'train' and client_file != self.client_file and self.client_file is not None:
                continue  # Skip other clients' data during training if client_file is specified
            path = os.path.join(args.data_path, phase, name + '.npy' if not name.endswith('.npy') else name)
            if os.path.exists(path):
                data = np.load(path)
                if data.ndim == 3 and data.shape[2] == 3:  # HxWxC
                    data = cv2.resize(data, (224, 224), interpolation=cv2.INTER_AREA)
                elif data.ndim == 3 and data.shape[0] == 3:  # CxHxW
                    data = np.transpose(data, (1, 2, 0))
                    data = cv2.resize(data, (224, 224), interpolation=cv2.INTER_AREA)
                else:
                    raise ValueError(f"Unexpected shape for {path}: {data.shape}")
                self.data_cache[name] = data
            else:
                print(f"Warning: File {path} not found, skipping...")
        client_str = f" {self.client_file}" if phase == 'train' and self.client_file is not None else ''
        print(f"Preloaded {len(self.data_cache)} images for {phase}{client_str}, total size approximately {sum(x.nbytes for x in self.data_cache.values()) / (1024 ** 3):.2f} GB")

    def __getitem__(self, index):
        index = index % len(self.img_paths)
        client_file, name = self.img_paths[index]
        if name not in self.data_cache:
            raise KeyError(f"Data for {name} not preloaded")
        data = self.data_cache[name]
        label = self.labels.get(name, 0)
        data = torch.from_numpy(data).permute(2, 0, 1).float()
        data = data / 255.0
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return sum(1 for client_file, name in self.img_paths if name in self.data_cache and (self.phase == 'test' or (self.phase == 'train' and client_file == self.client_file)))
        
####################because we had problem with dataloader, we modified the next function
def create_dataset_and_evalmetrix(args, mode='finetune'):
    ## get the joined clients
    if args.split_type == 'central':
        args.dis_cvs_files = [f for f in os.listdir(os.path.join(args.data_path, args.split_type)) 
                              if f.endswith('.csv') and f not in ['test.csv', 'val.csv']]
    else:
        args.dis_cvs_files = [f for f in os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type)) 
                              if f.endswith('.csv') and f not in ['test.csv', 'val.csv']]
    
    args.clients_with_len = {}
    
    for single_client in args.dis_cvs_files:
        if args.split_type == 'central':
            img_paths = list({line.strip().split(',')[0] for line in
                            open(os.path.join(args.data_path, args.split_type, single_client))})
        else:
            img_paths = list({line.strip().split(',')[0] for line in
                              open(os.path.join(args.data_path, f'{args.n_clients}_clients',
                                                args.split_type, single_client))})
        args.clients_with_len[single_client] = len(img_paths)
    
    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False
    args.best_eval_loss = {}
    
    for single_client in args.dis_cvs_files:
        if mode == 'pretrain':
            args.best_mlm_acc[single_client] = 0 
            args.current_mlm_acc[single_client] = []
        if mode == 'finetune':
            args.best_acc[single_client] = 0 if args.nb_classes > 1 else 999
            args.current_acc[single_client] = 0
            args.current_test_acc[single_client] = []
            args.best_eval_loss[single_client] = 9999


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_w:offset_w + size, offset_h:offset_h + size]

def process_covidx_image(img, size=224, top_percent=0.08, crop=False):
    img = crop_top(img, percent=top_percent)
    if crop:
        img = central_crop(img)
    img = resize(img, (size, size))
    img = img * 255
    return img

def process_covidx_image_v2(img, size=224):
    img = cv2.resize(img, (size, size))
    img = img.astype('float64')
    img -= img.mean()
    img /= img.std()
    return img
    
def random_ratio_resize(img, prob=0.3, delta=0.1):
    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 224 or size[1] > 224:
        print(img.shape, size, ratio)
    
    img = cv2.resize(img, size)
    
    padding = (left, top, right, bot)
    new_im = ImageOps.expand(img, padding)
    
    return img
