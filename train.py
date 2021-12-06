import os
# import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import re
import random
import numpy as np
import time
from tqdm import tqdm
import torch.nn as nn
import argparse
import nibabel
from torch.utils.tensorboard import SummaryWriter
from scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from Dataset import Dataset
# from DenseUNet4 import Dense_Unets
from DenseUNet import DenseUNet
# from unet import U_Net
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        # self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")

def simple_accuracy(preds, labels):
    preds[preds>0.5] = 1
    preds[preds<0.5] = 0
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_model(args, model, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.npz" % int(global_step/11))
    # if args.fp16:
    #     checkpoint = {
    #         'model': model_to_save.state_dict(),
    #         # 'amp': amp.state_dict()
    #     }
    # else:
    #     checkpoint = {
    #         'model': model_to_save.state_dict(),
    #     }
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    # logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_nii_gz(savepath,img):
    # img.tofile(savepath)
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,savepath)

def load_data(patch_path):
    data_patch_path = []
    label_patch_path = []
    mask_patch_path = []
    pathSet = patch_path.copy()
    for p in pathSet:
        if 'BlockCTplaque3mm' in p: # data
            data_patch_path.append(p)
        elif 'pie' in p:  # label
            label_patch_path.append(p)
        elif 'mask' in p:
            mask_patch_path.append(p)
    return data_patch_path, label_patch_path, mask_patch_path

def test(args, model, test_loader, test_idx):
    cnt = 0
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        data, label, mask = batch
        pred = model(data,mask)
        feature = pred.detach().cpu().numpy()
        save_nii_gz(savepath=os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\tempResults',str(test_idx[cnt]).zfill(3)+'.nii.gz'),img=np.squeeze(feature,axis=(0)))
        cnt = cnt+1

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    # logger.info("***** Running Validation *****")
    # logger.info("  Num steps = %d", len(test_loader))
    # logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label, all_mask = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)

    lossfunc = nn.BCELoss()
    # lossfunc = nn.BCEWithLogitsLoss().cuda()
    lossfunc = nn.MSELoss()
    # lossfunc = nn.CrossEntropyLoss().cuda()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        data, label, mask = batch
        # data = (data-data.min())/(data.max()-data.min())

        with torch.no_grad():
            pred = model(data,mask)
            eval_loss = lossfunc(pred,label)
            eval_losses.update(eval_loss.item())

        if len(all_preds) == 0:
            all_preds.append(pred.detach().cpu().numpy())
            all_label.append(label.detach().cpu().numpy())
            all_mask.append(mask.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], pred.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], label.detach().cpu().numpy(), axis=0
            )
            all_mask[0] = np.append(
                all_mask[0], mask.detach().cpu().numpy(), axis=0
            )

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    val_accuracy = accuracy.detach().cpu().numpy()
    writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=int(global_step/len(epoch_iterator)))
    writer.add_scalar("test/loss", scalar_value=eval_losses.val, global_step=int(global_step/len(epoch_iterator)))

    return val_accuracy

def train(args, model):
    # Prepare dataset
    idx = np.arange(args.blockNum)
    test_idx = np.arange(20,41) 
    # test_idx = np.arange(70,91)
    valid_idx = np.arange(120,141)
    all_patch_path = os.listdir(path=args.data_root) 
    all_patch_path = sorted(all_patch_path) # ordered
    figure = []
    for ID in all_patch_path:
        temp = int(re.findall(r"\d+",ID)[1])
        if temp not in figure:
            figure.append(temp) 
    test_idx_temp = test_idx.copy()
    for ID in test_idx_temp:
        if ID not in figure:
            test_idx = np.delete(test_idx,np.argwhere(test_idx==ID))
    valid_idx_temp = valid_idx.copy()
    for ID in valid_idx_temp:
        if ID not in figure:
            valid_idx = np.delete(valid_idx,np.argwhere(valid_idx==ID))
    # real后缀代表真实的索引（因为索引和层数不是一一对应）
    test_idx_real = []
    for i in test_idx:
        test_idx_real.append(all_patch_path.index('BlockCTplaque3mm'+str(i).zfill(3)+'.nii.gz' ))

    test_idx_real = np.array(test_idx_real)
    
    valid_idx_real = []
    for i in valid_idx:
        valid_idx_real.append(all_patch_path.index('BlockCTplaque3mm'+str(i).zfill(3)+'.nii.gz' ))

    valid_idx_real = np.array(valid_idx_real)
    remain_idx_real = np.delete(idx,test_idx_real)
    remain_idx_real = np.delete(remain_idx_real,np.argwhere(remain_idx_real==valid_idx_real))
    train_idx_real = remain_idx_real.copy()

    # train_patch_path = list(np.array(all_patch_path)[train_idx_real]) # train data path
    # train_patch_path.extend(list(np.array(all_patch_path)[train_idx_real+args.blockNum])) # mask path
    # train_patch_path.extend(list(np.array(all_patch_path)[train_idx_real+2*args.blockNum])) # train label path
    valid_patch_path = list(np.array(all_patch_path)[valid_idx_real]) # valid data path
    valid_patch_path.extend(list(np.array(all_patch_path)[valid_idx_real+args.blockNum])) # mask path
    valid_patch_path.extend(list(np.array(all_patch_path)[valid_idx_real+2*args.blockNum])) # valid label path
    test_patch_path = list(np.array(all_patch_path)[test_idx_real])
    test_patch_path.extend(list(np.array(all_patch_path)[test_idx_real+args.blockNum]))
    test_patch_path.extend(list(np.array(all_patch_path)[test_idx_real+2*args.blockNum]))
    train_patch_path = []
    for p in all_patch_path:
        if p not in valid_patch_path and p not in test_patch_path:
            train_patch_path.append(p)

    # train_transform=transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor()])
    # valid_transform=transforms.Compose([transforms.ToTensor()])

    data_patch_path, label_patch_path, mask_patch_path = load_data(train_patch_path)
    data_patch_path_v, label_patch_path_v, mask_patch_path_v = load_data(valid_patch_path)

    trainset = Dataset(data_patch_path, label_patch_path, mask_patch_path, train_folder_path=args.data_root, transform=None)
    validset = Dataset(data_patch_path_v, label_patch_path_v, mask_patch_path_v, args.data_root, transform = None)

    if args.mode=='test':
        data_patch_path_t, label_patch_path_t, mask_patch_path_t = load_data(test_patch_path)
        testset = Dataset(data_patch_path_t, label_patch_path_t, mask_patch_path_t, args.data_root, transform = None)
        test_loader = DataLoader(testset,
                                #  sampler=test_sampler,
                                batch_size=1,
                                # num_workers=0,
                                drop_last=False,
                                # pin_memory=True,
                                shuffle=False) 
        test(args,model,test_loader,test_idx)
        

    elif args.mode=='train':
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        train_loader = DataLoader(trainset,
                                #   sampler=train_sampler, # sampler is not compatible with iterable-style datasets
                                batch_size=args.train_batch_size,
                                # num_workers=0,
                                drop_last=True,
                                # pin_memory=True,
                                shuffle=True)
        test_loader = DataLoader(validset,
                                #  sampler=test_sampler,
                                batch_size=args.eval_batch_size,
                                # num_workers=0,
                                drop_last=True,
                                # pin_memory=True,
                                shuffle=False) 

        # Prepare optimizer and scheduler
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=args.learning_rate,
        #                             momentum=0.9,
        #                             weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)

        t_total = args.num_steps
        if args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        else:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        global_step, best_acc = 0, 0
        start_time = time.time()
        # model.zero_grad()
        lossfunc = nn.BCELoss()#.cuda()
        # lossfunc = nn.BCEWithLogitsLoss(reduction='none').cuda()
        # lossfunc = nn.MSELoss()
        # lossfunc = nn.CrossEntropyLoss().cuda()
        losses = AverageMeter()
        model.train()
        w = model.dense2_c1.weight.data.clone()
        while True:
            
            
            epoch_iterator = tqdm(train_loader,
                                # desc="Training (X / X Steps) (loss=X.X)",
                                desc="Training (X / X Epoches) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True,
                                disable=False)
            all_preds, all_label, all_mask = [], [], []
            
            for step, batch in enumerate(epoch_iterator):
            # for step, batch in enumerate(train_loader):
                # batch_ = tuple(t.to(args.device) for t in batch)
                # data, label, mask = batch_
                data = batch[0].to(args.device)
                label = batch[1].to(args.device)
                mask = batch[2].to(args.device)
                # data = (data-data.min())/(data.max()-data.min())
                
                # model.train()
                # np.save(os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2','data'+str(global_step)+'.npy'),
                #         data.detach().cpu().numpy())
                # np.save(os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2','mask'+str(global_step)+'.npy'),
                #         mask.detach().cpu().numpy())
                # np.save(os.path.join(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2','label'+str(global_step)+'.npy'),
                #         label.detach().cpu().numpy())  
                pred = model(data,mask)
                # from matplotlib import pyplot as plt
                # plt.imshow(pred[0,0,:,:].detach().cpu().numpy(),'gray')
                # plt.figure()
                # plt.imshow(pred[0,1,:,:].detach().cpu().numpy(),'gray')
                # plt.figure()
                # plt.imshow(label[0,0,:,:].detach().cpu().numpy(),'gray')
                # plt.figure()
                # plt.imshow(label[0,1,:,:].detach().cpu().numpy(),'gray')
                # plt.figure()
                # plt.imshow(data[0,3,:,:].detach().cpu().numpy(),'gray')
                # plt.show()
                # weight = torch.empty(label.shape).cuda()
                # for i in range(args.train_batch_size):
                #     if label[i:].sum()==0: # non-plaque
                #         weight[i:] = 0.2
                #     elif label[i,0,:,:].sum()==0 and label[i,1,:,:].sum()!=0: # calcium
                #         weight[i:] = 0.5
                #     else:
                #         weight[i:] = 1.0
                # lossfunc = nn.BCELoss(weight=weight)
                loss = lossfunc(pred,label)
                # reg_loss = Regularization(model,weight_decay=1e-4,p=2).to(args.device)
                # l2_regularization = torch.tensor([0],dtype=torch.float32).cuda()
                # for name, param in model.named_parameters():
                #     # if  'weight' in name and 'bn' in name:
                #         print(name)
                #         print(param.grad)
                #         print(param)
                #         l2_regularization += torch.norm(param, 2)
                # loss = loss+reg_loss(model)
                losses.update(loss.item())
                # loss = loss.mean()
                # if len(all_preds) == 0:
                #     all_preds.append(pred.detach().cpu().numpy())
                #     all_label.append(label.detach().cpu().numpy())
                #     all_mask.append(mask.detach().cpu().numpy())
                # else:s
                #     all_preds[0] = np.append(
                #         all_preds[0], pred.detach().cpu().numpy(), axis=0
                #     )
                #     all_label[0] = np.append(
                #         all_label[0], label.detach().cpu().numpy(), axis=0
                #     )
                #     all_mask[0] = np.append(
                #         all_mask[0], mask.detach().cpu().numpy(), axis=0
                #     )

                optimizer.zero_grad()
                loss.backward()
                # for name, param in model.named_parameters():
                #     print('-->name：',name,param.size())
                #     print('-->grad_requires:',param.requires_grad)
                #     print('-->grad_value:',param.grad)
                optimizer.step()
                # print((w - model.dense2_c1.weight.data).sum())
                print(global_step, optimizer.param_groups[0]['lr'])
                # scheduler.step()
                global_step += 1
                epoch_iterator.set_description(
                    # "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                    "Training (%d / %d Epoches) (loss=%2.5f)" % (int(global_step/len(epoch_iterator)), int(t_total/len(epoch_iterator)), losses.val)
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                # writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model, global_step)
                        best_acc = accuracy
                    # logger.info("best accuracy so far: %f" % best_acc)
                    # model.train()
                if global_step % t_total == 0:
                    break
            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=int(global_step/len(epoch_iterator)))
            # all_preds, all_label = all_preds[0], all_label[0]
            # accuracy = simple_accuracy(all_preds, all_label) 
            # accuracy = torch.tensor(accuracy).to(args.device)
            # train_accuracy = accuracy.detach().cpu().numpy()
            # logger.info("train accuracy so far: %f" % train_accuracy)
            losses.reset()
            if global_step % t_total == 0:
                save_model(args, model, global_step)
                break

        writer.close()
        # logger.info("End Training!")
        end_time = time.time()
        # logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=False,
                        default="train",
                        help="option: train or test")
    parser.add_argument("--name", required=False,
                        default="Lipid_Detection",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument('--data_root', type=str, 
                        default=r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\2Channel_25DBlock')
                        # default=r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\Less_25DBlock')
    parser.add_argument("--output_dir", default=r"\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\logs", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=128, type=int,
                        help="Resolution size")
    parser.add_argument("--blockNum", default=208, type=int,
                        help="The number of input blocks")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=11*5, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=11000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.device = device
    # model = U_Net(7,2)
    model = DenseUNet(subSlices=7, categories=2)
    # model = Dense_Unet(subSlices=7, categories=2)
    # model = DenseNet(num_classes=2)
    
    # args.mode = 'test'

    if args.mode == 'train':
        model.to(args.device)
        train(args, model)
    elif args.mode == 'test':
        model_path = r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\logs\100_checkpoint.npz'
        model.load_state_dict(torch.load(model_path)['model'])
        model.eval()
        model.to(args.device)
        train(args, model)
