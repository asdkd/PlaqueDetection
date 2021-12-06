import numpy as np
import nibabel
import os
import torch
import torch.nn.functional as F

class Dataset():

    def __init__(self, list_IDs, label_IDs, mask_IDs, train_folder_path, imgSize=(128,128), n_channels=7, n_classes=2, transform=None):
        'Initialization'
        self.imgSize = imgSize
        # self.labels = labels
        self.list_IDs = list_IDs
        self.label_IDs = label_IDs
        self.mask_IDs = mask_IDs
        self.train_folder_path = train_folder_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.transform = transform
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) ))

    def __getitem__(self, index):
        # print('+++index+++: '+str(index))
        img = nibabel.load(os.path.join(self.train_folder_path,self.list_IDs[index])).get_fdata() + 1024
        img = (img-img.min())/(img.max()-img.min())
        target = nibabel.load(os.path.join(self.train_folder_path,self.label_IDs[index])).get_fdata()#+1e-10
        # target_tissue = np.ones((tar))
        mask = nibabel.load(os.path.join(self.train_folder_path,self.mask_IDs[index])).get_fdata()

        img = np.squeeze(img,axis=0)
        target = np.squeeze(target)
        mask = np.squeeze(mask,axis=0)
        img = img.transpose(2,0,1) # 7*128*128
        target = target.transpose(2,0,1) # 2*128*128
        mask = mask.transpose(2,0,1) # 2*128*128
        target = target*mask
        img = torch.from_numpy(img).type(torch.float) # 7*128*128
        target = torch.from_numpy(target).type(torch.float) # 2*128*128
        mask = torch.from_numpy(mask).type(torch.int) # 2*128*128

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
            mask = self.transform(mask)

        return img, target, mask

    # def __getitem__(self, index):
    #     # print('+++index+++: '+str(index))
    #     img = nibabel.load(os.path.join(self.train_folder_path,self.list_IDs[index])).get_fdata() + 1024
    #     img = (img-img.min())/(img.max()-img.min())
    #     target = nibabel.load(os.path.join(self.train_folder_path,self.label_IDs[index])).get_fdata()#+1e-10
    #     # target_tissue = np.ones((tar))
    #     mask = nibabel.load(os.path.join(self.train_folder_path,self.mask_IDs[index])).get_fdata() 

    #     img = np.squeeze(img,axis=0)
    #     target = np.squeeze(target)
    #     mask = np.squeeze(mask,axis=0)
    #     img = img.transpose(2,0,1) # 7*128*128
    #     # target = target.transpose(2,0,1) # 128*128
    #     mask = mask.transpose(2,0,1) # 4*128*128
    #     # target = target*mask
    #     img = torch.from_numpy(img).type(torch.float) # 7*128*128
    #     target = torch.from_numpy(target).type(torch.float) # 128*128
    #     target = F.one_hot(target.long(),num_classes=self.n_classes) # 128*128*4
    #     target = target.transpose(2,0).type(torch.float) # 4*128*128
    #     target = target*mask
    #     mask = torch.from_numpy(mask).type(torch.int) # 4*128*128

    #     if self.transform is not None:
    #         img = self.transform(img)
    #         target = self.transform(target)
    #         mask = self.transform(mask)

    #     return img, target, mask



