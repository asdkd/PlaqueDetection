import numpy as np
# import tensorflow as tf
import nibabel
# from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import os
import re

class img_generator(object):
    def __init__(self,test_results_path):
        super().__init__()
        self.test_results_path = test_results_path

    def load_imgs2(self,exclude_str=None,exclude_str2=None,idx_range=None):
        test_results_paths_list = os.listdir(self.test_results_path)
        test_results_paths_list = sorted(test_results_paths_list)
        for ID in test_results_paths_list:
            if exclude_str is not None:
                if exclude_str in ID or exclude_str2 in ID:
                    continue
            if idx_range is not None:
                figure = int(re.findall(r"\d+",ID)[1])
                if figure not in idx_range:
                    continue
            img = nibabel.load(os.path.join(self.test_results_path,ID)).get_fdata()
            try:
                img = np.squeeze(img,axis=(0))
            except:
                pass
            yield(img)

    def load_imgs(self,exclude_str=None,idx_range=None):
        test_results_paths_list = os.listdir(self.test_results_path)
        test_results_paths_list = sorted(test_results_paths_list)
        for ID in test_results_paths_list:
            if exclude_str is not None:
                if exclude_str in ID:
                    continue
            if idx_range is not None:
                figure = int(re.findall(r"\d+",ID)[1])
                if figure not in idx_range:
                    continue
            img = nibabel.load(os.path.join(self.test_results_path,ID)).get_fdata()
            try:
                img = np.squeeze(img,axis=(0))
            except:
                pass
            yield(img)

if __name__ == '__main__':
    # img = nibabel.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\Less_25DBlock\Blockmask2CT060.nii.gz').get_fdata()
    # plt.imshow(img[0,:,:,3],'gray')
    # plt.show()
    # img = nibabel.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\LessPiecesofPie\pie2CT128.nii.gz').get_fdata()
    # data = to_categorical(img,4)
    # fig, axs = plt.subplots(1,5)
    # axs[0].imshow(data[:,:,0],'gray')
    # axs[1].imshow(data[:,:,1],'gray')
    # axs[2].imshow(data[:,:,2],'gray')
    # axs[3].imshow(data[:,:,3],'gray')
    # axs[4].imshow(img,'gray')
    # axs[0].set_title('non-plaque')
    # axs[1].set_title('pure lipid')
    # axs[2].set_title('pure calcium')
    # axs[3].set_title('mixed')
    # axs[4].set_title('integrated label')
    # fig.suptitle('Multi Class Task (Categorical Cross-Entropy Loss)')
    # # plt.show()
    # fig, axs = plt.subplots(1,4)
    # axs[0].imshow(data[:,:,0],'gray')
    # axs[1].imshow(data[:,:,1]+data[:,:,3],'gray')
    # axs[2].imshow(data[:,:,2]+data[:,:,3],'gray')
    # # axs[3].imshow(data[:,:,3],'gray')
    # axs[3].imshow(img,'gray')
    # axs[0].set_title('non-plaque')
    # axs[1].set_title('lipid rich')
    # axs[2].set_title('calcium')
    # # axs[3].set_title('mixed')
    # axs[3].set_title('integrated label')
    # fig.suptitle('Multi Label Task (Binary Cross-Entropy Loss)')
    # plt.show()


    # test_results_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\results_gpu1_batchSize2'
    # test_results_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\results_gpu1_batchSize16'
    # test_results_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\results_gpu1_batchSize8_step20PerEpoch'
    # test_results_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\tempResults_1000epoch_test160_180'
    test_results_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\tempResults'
    shower = img_generator(test_results_path)
    generator_test = shower.load_imgs()

    # label_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\25DBlock'
    # label_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\Less_25DBlock'
    label_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\2Channel_25DBlock'
    # label_path_2 = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\FirstPieceofPie'
    labelimg = img_generator(label_path)

    # index of unconsecutive slice
    # all_path = os.listdir(path=r'\\vf-lkeb\lkeb$\Scratch\Alexander\PiecesofPie\LessPiecesofPie') 
    all_path = os.listdir(path=r'\\vf-lkeb\lkeb$\Scratch\Alexander\PiecesofPie\2ChannelPieces') 
    all_path = sorted(all_path) # ordered
    figure = []
    for ID in all_path:
        temp = int(re.findall(r"\d+",ID)[1])
        if temp not in figure:
            figure.append(temp) 
    # test_idx_temp = np.arange(120,141)
    # test_idx_temp = np.arange(160,181)
    # test_idx_temp = np.arange(10,30)
    test_idx_temp = np.arange(21,41)
    # test_idx_temp = np.arange(70,91)
    test_idx = []
    for ID in test_idx_temp:
        if ID in figure:
            test_idx.append(ID)
    test_idx = np.array(test_idx)

    generator_label = labelimg.load_imgs2('CTplaque3mm','mask',test_idx)
    # generator_label = labelimg.load_imgs('CTplaque3mm',np.arange(150,160))
    # labelimg2 = img_generator(label_path_2)
    # generator_label_2 = labelimg.load_imgs('CTplaque3mm',np.arange(35,62))

    # CTimg_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\FirstPieceofPie'
    # CTimg_path = r'\\vf-lkeb\lkeb$\Scratch\Alexander\PiecesofPie\LessPiecesofPie'
    CTimg_path = r'\\vf-lkeb\lkeb$\Scratch\Alexander\PiecesofPie\2ChannelPieces'
    CTimg = img_generator(CTimg_path)
    # generator_CTimg = CTimg.load_imgs('pie2CT',np.arange(35,62))
    generator_CTimg = CTimg.load_imgs('pie2CT',test_idx)

    cnt = 0

    for data,label,ct in zip(generator_test,generator_label,generator_CTimg):
        mask = (ct+1024)!=0
        mask = mask+0
        fig, axs = plt.subplots(2,3)
        axs[0,0].imshow(data[0,:,:],'gray')
        axs[0,1].imshow(data[0,:,:],'gray')
        label = np.squeeze(label)
        axs[1,0].imshow(label[:,:,0]*mask,'gray')
        axs[1,1].imshow(label[:,:,1]*mask,'gray')
        axs[0,2].imshow(ct,'gray')
        axs[1,2].imshow(ct,'gray')
        axs[0,0].set_title('lipid rich')
        axs[0,1].set_title('calcium')
        axs[0,2].set_title('CT')
        axs[0,0].set_ylabel('predict')
        axs[1,0].set_ylabel('label')
        # fig.suptitle(str(120+cnt).zfill(3)+'.nii.gz')
        fig.suptitle(str(test_idx[cnt]).zfill(3)+'.nii.gz')
        cnt = cnt+1

    # for data,label,ct in zip(generator_test,generator_label,generator_CTimg):
    #     mask = (ct+1024)!=0
    #     mask = mask+0
    #     fig, axs = plt.subplots(2,5)
    #     axs[0,0].imshow(data[:,:,0],'gray')
    #     axs[0,1].imshow(data[:,:,1],'gray')
    #     axs[0,2].imshow(data[:,:,2],'gray')
    #     axs[0,3].imshow(data[:,:,3],'gray')
    #     classification_label = to_categorical(label,4)
    #     axs[1,0].imshow(classification_label[:,:,0]*mask,'gray')
    #     axs[1,1].imshow(classification_label[:,:,1]*mask,'gray')
    #     axs[1,2].imshow(classification_label[:,:,2]*mask,'gray')
    #     axs[1,3].imshow(classification_label[:,:,3]*mask,'gray')
    #     axs[0,4].imshow(ct,'gray')
    #     axs[1,4].imshow(ct,'gray')
    #     axs[0,0].set_title('non-plaque')
    #     axs[0,1].set_title('lipid rich')
    #     axs[0,2].set_title('calcium')
    #     axs[0,3].set_title('mixed')
    #     axs[0,4].set_title('CT')
    #     axs[1,0].set_title('non-plaque label')
    #     axs[1,1].set_title('lipid rich label')
    #     axs[1,2].set_title('calcium label')
    #     axs[1,3].set_title('mixed label')
    #     axs[1,4].set_title('CT')
    #     # fig.suptitle(str(120+cnt).zfill(3)+'.nii.gz')
    #     fig.suptitle(str(test_idx[cnt]).zfill(3)+'.nii.gz')
    #     cnt = cnt+1

    plt.show()

    print(1.0)
