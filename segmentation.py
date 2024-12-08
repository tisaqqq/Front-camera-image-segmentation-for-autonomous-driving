#cite: https://www.kaggle.com/code/vikram12301/multiclass-semantic-segmentation-pytorch
#代码来自于上方开源模型并进行过调整。

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from torchsummary import summary

#image（原图） stored in TotalImage, Label（真值） stored in TotalLabel
#如果要扩增训练数据，请在/home/liumuyuan/mountHome/qll/tinyData下的TotalImage和TotalLabel中同步增加
root = '/home/liumuyuan/mountHome/qll/tinyData'
data_dir = [root]

#label marking for 3 categories, car(class 3), lane_marks(class 2), others (class 0)
#注意：不同的图片有不同数量的channel，如果后续添加新的training/testing图片数据，需要更新这里的mapping
#注意：如果更改channel数量（不再是3个），需要将下方unet_model中的out-channels数值对应更改
lb_mapping = {
    0: 0, 
    200: 1, 
    201: 1, 
    202: 1, 
    205: 1, 
    210: 1, 
    212: 1,
    214: 1, 
    217: 1, 
    220: 1, 
    221: 1, 
    222: 1, 
    227: 1, 
    250: 1, 
    255: 2
}

#transforms
#后续调参可更改transforms
t1 = A.Compose([
    A.Resize(160,240),
    #A.HorizontalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.2),
    #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.4),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

#############################--Dataset setup--#############################
class LyftUdacityMap(Dataset):
    def __init__(self, img_dir, transform=None, label_mapping=None):
        self.transforms = transform
        self.label_mapping = label_mapping
        image_paths = [i+'/TotalImage' for i in img_dir]#path for debugging purpose, not really used for the model
        seg_paths = [i+'/TotalLabel' for i in img_dir]#path for debugging purpose, not really used for the model
        self.images, self.masks = [], []
        self.image_paths, self.mask_paths = [], []

        for i in image_paths:
            imgs = sorted(os.listdir(i))  # Ensure sorted order
            self.images.extend([os.path.join(i, img) for img in imgs])
            self.image_paths.extend([os.path.join(i, img) for img in imgs])
            
        for i in seg_paths:
            masks = sorted(os.listdir(i))  # Ensure sorted order
            self.masks.extend([os.path.join(i, mask) for mask in masks])
            self.mask_paths.extend([os.path.join(i, mask) for mask in masks])

        # Check if the number of images and masks are the same
        assert len(self.images) == len(self.masks), "Number of images and masks do not match"

        # Check if image and mask pairs are correctly ordered by comparing filenames
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            assert img_path.split('/')[-1].replace('.jpg', '') == mask_path.split('/')[-1].replace('_bin.png', ''), \
                f"Image {img_path} and mask {mask_path} are not correctly paired"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        if self.transforms is not None:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        # Apply label mapping if provided
        if self.label_mapping is not None:
            mask = self.remap_targets(mask, self.label_mapping)

        return img, mask, img_path, mask_path  # Return paths as well

    def remap_targets(self, targets, mapping):
        targets_mapped = targets.clone()
        for original_value, new_value in mapping.items():
            targets_mapped[targets == original_value] = new_value
        return targets_mapped

def get_images_map(image_dir,transform = None,batch_size=1,shuffle=True,pin_memory=True):
    #注意：label_mapping = lb_mapping/None, change mapping if neccessary
    #如果想看到label_mapping之前的所有channel,将下方label_mapping改为None
    data = LyftUdacityMap(image_dir,transform = t1,label_mapping = lb_mapping)
    train_size = int(0.8 * data.__len__())
    print("train_size", train_size)
    test_size = data.__len__() - train_size
    print("test_size", test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
    return train_batch,test_batch

#get train batch and test batch
train_batchMap,test_batchMap = get_images_map(data_dir,transform =t1,batch_size=8)

'''
#check path of batches of whether they match each other
#检查每组batch原图和真值的地址是否相互对应
for img, mask, img_path, mask_path in train_batchMap:
    print(f"Image path: {img_path}")
    print(f"Mask path: {mask_path}")
    # Optionally, you can print or display the image and mask
    # img1 = np.transpose(img[0, :, :, :], (1, 2, 0))
    # mask1 = np.array(mask[0, :, :])
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(mask1, cmap='gray')
    # plt.show()
    break  # Break after the first batch for demonstration purposes
'''

#输出multi_channel_plot.png：展示模型输出的各通道图像，以便了解不同输出通道对应的图像区域
#如果想要plot出在lb_mapping之前的所有channel，可以先把上方get_images_map里的lb_mapping改成None
def visualize_channels(img, mask, save_path="channel_plot.png"):
    unique_classes = torch.unique(mask)
    print(f"Unique classes in the mask: {unique_classes.tolist()}")
    
    fig, axs = plt.subplots(1, len(unique_classes) + 1, figsize=(18, 6))
    
    # Show the original image
    img_np = np.transpose(img, (1, 2, 0))  # Convert CHW to HWC
    axs[0].imshow(img_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    for i, cls in enumerate(unique_classes):
        # Create a binary mask for each class
        binary_mask = (mask == cls).float()
        
        # Plot the binary mask
        axs[i + 1].imshow(binary_mask, cmap='gray')  # Plot the 2D binary mask
        axs[i + 1].set_title(f'Class {cls.item()}')
        axs[i + 1].axis('off')

    plt.savefig(save_path)
    print(f"Channel-wise visualization saved as {save_path}")

for img, mask, pth1, pth2 in train_batchMap:
    print("Image plot saved as multi_channel_plot.png")
    visualize_channels(img[0], mask[0], save_path="multi_channel_plot.png")
    break  # Remove this if you want to process more images

#check image input
#输出multi_plot.png, 展示几组原图与真值图像，用于训练前了解使用的图片数据。
for img,mask,img_path,mask_path in train_batchMap:
    img1 = np.transpose(img[0,:,:,:],(1,2,0))
    mask1 = np.array(mask[0,:,:])
    img2 = np.transpose(img[1,:,:,:],(1,2,0))
    mask2 = np.array(mask[1,:,:])
    img3 = np.transpose(img[2,:,:,:],(1,2,0))
    mask3 = np.array(mask[2,:,:])
    fig , ax =  plt.subplots(3, 2, figsize=(18, 18))
    ax[0][0].imshow(img1)
    ax[0][1].imshow(mask1)
    ax[1][0].imshow(img2)
    ax[1][1].imshow(mask2)
    ax[2][0].imshow(img3)
    ax[2][1].imshow(mask3)
    plt.savefig("multi_plot.png")
    print("Image plot saved as multi_plot.png")
    break #如果想要plot更多组图片可删除/更改


#############################--U-net Model--#############################
class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)    

class unet_model(nn.Module):
    def __init__(self,out_channels=3,features=[64, 128, 256, 512]):#remember to change out_channels if needed, 这里暂时设为3个（对应class 0,1,2）
        super(unet_model,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(3,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = unet_model().to(DEVICE)

##########freeze layers if needed
for param in model.conv1.parameters():
    param.requires_grad = False

for param in model.conv2.parameters():
    param.requires_grad = False

summary(model, (3, 256, 256))

LEARNING_RATE = 1e-3

num_epochs = 15

#########--3个loss functions--##########
#在下方训练中使用的是loss_fn2，目前看下来效果最好
loss_fn1 = nn.CrossEntropyLoss() #pure cross loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.softmax(inputs, dim=1)  # Apply softmax to get probabilities
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Convert to shape [batch_size, num_classes, height, width]
        
        inputs = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        intersection = (inputs * targets_one_hot).sum(dim=2)
        dice = (2. * intersection + smooth) / (inputs.sum(dim=2) + targets_one_hot.sum(dim=2) + smooth)
        return 1 - dice.mean()
def loss_fn2(predictions, targets): #dice loss + cross loss
    ce_loss = nn.CrossEntropyLoss()(predictions, targets)
    dice = DiceLoss()(predictions, targets)
    return ce_loss + dice

class TverskyLoss(nn.Module):#
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply softmax to the inputs
        inputs = torch.softmax(inputs, dim=1)
        # Ensure the targets are on the same device as inputs
        targets = targets.to(inputs.device)
        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        one_hot = torch.eye(num_classes, device=inputs.device)
        targets_one_hot = one_hot[targets].permute(0, 3, 1, 2)
        # Calculate True Positives, False Positives, and False Negatives
        TP = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        FP = ((1 - targets_one_hot) * inputs).sum(dim=(0, 2, 3))
        FN = (targets_one_hot * (1 - inputs)).sum(dim=(0, 2, 3))
        # Calculate Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        # Return Tversky loss
        return 1 - tversky.mean()
        # Apply softmax to the inputs
        inputs = torch.softmax(inputs, dim=1)
        # Convert targets to one-hot encoding
        targets_one_hot = torch.eye(inputs.shape[1])[targets].permute(0, 3, 1, 2).to(inputs.device)
        # Calculate True Positives, False Positives, and False Negatives
        TP = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        FP = ((1 - targets_one_hot) * inputs).sum(dim=(0, 2, 3))
        FN = (targets_one_hot * (1 - inputs)).sum(dim=(0, 2, 3))
        # Calculate Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        # Return Tversky loss
        return 1 - tversky.mean()
def loss_fn3(predictions, targets): #Tversky loss + cross loss
    ce_loss = nn.CrossEntropyLoss()(predictions, targets)
    tk = TverskyLoss()(predictions, targets)
    return ce_loss + tk

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.amp.GradScaler(device='cuda')



#############################--trainning--#############################
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_batchMap),total=len(train_batchMap))
    for batch_idx, (data, targets, path1, path2) in loop:#path1 path2 not used. only for debug purpose
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn2(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y, path1, path2 in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

check_accuracy(train_batchMap, model)

check_accuracy(test_batchMap, model)

#############################--output result--#############################
#模型输出multi_result.png，展示模型输出结果，包括几组原图、预测图与真值图的对比。
for x,y,path1,path2 in test_batchMap:
    x = x.to(DEVICE)
    fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
    softmax = nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
    img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
    preds1 = np.array(preds[0,:,:])
    mask1 = np.array(y[0,:,:])
    img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
    preds2 = np.array(preds[1,:,:])
    mask2 = np.array(y[1,:,:])
    img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
    preds3 = np.array(preds[2,:,:])
    mask3 = np.array(y[2,:,:])
    ax[0,0].set_title('Image')
    ax[0,1].set_title('Prediction')
    ax[0,2].set_title('Mask')
    ax[1,0].set_title('Image')
    ax[1,1].set_title('Prediction')
    ax[1,2].set_title('Mask')
    ax[2,0].set_title('Image')
    ax[2,1].set_title('Prediction')
    ax[2,2].set_title('Mask')
    ax[0][0].axis("off")
    ax[1][0].axis("off")
    ax[2][0].axis("off")
    ax[0][1].axis("off")
    ax[1][1].axis("off")
    ax[2][1].axis("off")
    ax[0][2].axis("off")
    ax[1][2].axis("off")
    ax[2][2].axis("off")
    ax[0][0].imshow(img1)
    ax[0][1].imshow(preds1)
    ax[0][2].imshow(mask1)
    ax[1][0].imshow(img2)
    ax[1][1].imshow(preds2)
    ax[1][2].imshow(mask2)
    ax[2][0].imshow(img3)
    ax[2][1].imshow(preds3)
    ax[2][2].imshow(mask3)  
    plt.savefig("multi_result.png")
    print("Image plot saved as multi_result.png")
    break #if you want to plot more images change the break








