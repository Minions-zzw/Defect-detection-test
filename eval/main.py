import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval
import torch.optim as optim
from data_loader import SalObjDataset
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from model import Net
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root="../datasets/NEU Surface Defect Dataset/"
    salobj_dataset = SalObjDataset(image_root=os.path.join(root, "Source Images/"),
                                  gt_root=os.path.join(root, "Ground truth/"),
                                  transform=transforms.Compose([
                                      RescaleT(256),

                                      ToTensorLab(flag=0)])
                                  )
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=True, num_workers=1)
    net = Net(3, 1)
    net = net.to(device)
    net.load_state_dict(torch.load("../neu_net_model.pth"))
    net.eval()
    thread = Eval(salobj_dataloader,net, True)
    print(thread.run())