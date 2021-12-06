import numpy as np
from matplotlib import pyplot as plt

data = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2\data20.npy')
fig, axs = plt.subplots(4,4)
for i in range(data.shape[0]):
    axs[int(i/4),int(i%4)].imshow(data[int(i/4)*4+int(i%4),3,:,:],'gray')
    axs[int(i/4),int(i%4)].set_title('batch:'+str(i+1))
# plt.show()

mask = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2\mask20.npy')
fig, axs = plt.subplots(4,4)
for i in range(mask.shape[0]):
    axs[int(i/4),int(i%4)].imshow(mask[int(i/4)*4+int(i%4),0,:,:],'gray')
    axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

label = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2\label20.npy')
fig, axs = plt.subplots(4,4)
for i in range(label.shape[0]):
    axs[int(i/4),int(i%4)].imshow(label[int(i/4)*4+int(i%4),0,:,:],'gray')
    axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

label = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp2\label20.npy')
fig, axs = plt.subplots(4,4)
for i in range(label.shape[0]):
    axs[int(i/4),int(i%4)].imshow(label[int(i/4)*4+int(i%4),1,:,:],'gray')
    axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

plt.show()
print(1.0)

# data = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp\data26.npy')
# fig, axs = plt.subplots(4,4)
# for i in range(data.shape[0]):
#     axs[int(i/4),int(i%4)].imshow(data[int(i/4)*4+int(i%4),3,:,:],'gray')
#     axs[int(i/4),int(i%4)].set_title('batch:'+str(i+1))
# # plt.show()

# mask = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp\mask26.npy')
# fig, axs = plt.subplots(4,4)
# for i in range(mask.shape[0]):
#     axs[int(i/4),int(i%4)].imshow(mask[int(i/4)*4+int(i%4),0,:,:],'gray')
#     axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

# label = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp\label26.npy')
# fig, axs = plt.subplots(4,4)
# for i in range(label.shape[0]):
#     axs[int(i/4),int(i%4)].imshow(label[int(i/4)*4+int(i%4),0,:,:],'gray')
#     axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

# label = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp\label26.npy')
# fig, axs = plt.subplots(4,4)
# for i in range(label.shape[0]):
#     axs[int(i/4),int(i%4)].imshow(label[int(i/4)*4+int(i%4),1,:,:],'gray')
#     axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

# label = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp\label26.npy')
# fig, axs = plt.subplots(4,4)
# for i in range(label.shape[0]):
#     axs[int(i/4),int(i%4)].imshow(label[int(i/4)*4+int(i%4),2,:,:],'gray')
#     axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))

# label = np.load(r'\\vf-lkeb\lkeb$\Scratch\Xiaotong\lipid_plaque_detection\UNet_PyTorch\temp\label26.npy')
# fig, axs = plt.subplots(4,4)
# for i in range(label.shape[0]):
#     axs[int(i/4),int(i%4)].imshow(label[int(i/4)*4+int(i%4),3,:,:],'gray')
#     axs[int(i/4),int(i%4)].set_title ('batch:'+str(i+1))
# plt.show()
# print(1.0)
