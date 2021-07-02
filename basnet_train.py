import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from data_loader import SalObjDataset
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab


import os
from model import BASNet
import pytorch_ssim
import pytorch_iou
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)


	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(preds, labels_v):
    loss0 = bce_ssim_loss(preds[0], labels_v)
    loss=0
    for pred in preds:
        loss = loss+bce_ssim_loss(pred,labels_v)
    return loss0, loss
if __name__ == '__main__':


    epoch_num = 500
    batch_size_train = 8
    batch_size_val = 1
    train_num = 0
    val_num = 0

    root="./datasets/NEU Surface Defect Dataset/"
    salobj_dataset = SalObjDataset(image_root=os.path.join(root,"Source Images/"),
                                   gt_root=os.path.join(root,"Ground truth/"),
                                   transform=transforms.Compose([
                                       RescaleT(256),
                                       RandomCrop(224),
                                       ToTensorLab(flag=0)])
                                   )
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
    train_num =len(salobj_dataset)
    # ------- 3. define model --------

    # define the net
    net = BASNet(3, 1)
    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, label0= data
            inputs = inputs.type(torch.FloatTensor)
            label0 = label0.type(torch.FloatTensor)
            # label1 = label1.type(torch.FloatTensor)

            inputs_v, labels0_v = Variable(inputs.cuda(), requires_grad=False), Variable(label0.cuda(),requires_grad=False)
            # y zero the parameter gradients
            optimizer.zero_grad()


            # forward + backward + optimize
            outputs = net(inputs_v)

            loss0, loss = muti_bce_loss_fusion(outputs,labels0_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss0.item()

            # del temporary outputs and loss
            del outputs, loss0, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % 200 == 0:  # save model every 200 iterations

                torch.save(net.state_dict(), "neu_model.pth" )
                running_loss = 0.0
                running_tar_loss = 0.0

                net.train()  # resume train
                ite_num4val = 0

    print('-------------Congratulations! Training Done!!!-------------')
