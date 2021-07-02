import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim



from PIL import Image
import numpy as np
from model import BASNet



def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)

	return dn


# --------- 4. inference for each image ---------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imageDir="./datasets/NEU Surface Defect Dataset/Source Images/"
saveDir="./pred/"
transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
net = BASNet(3, 1)
net=net.to(device)
net.load_state_dict(torch.load("neu_model.pth"))
net.eval()
for file in os.listdir(imageDir):
    file_path=os.path.join(imageDir,file)
    image=Image.open(file_path).convert('RGB')
    W,H=image.size
    img=transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
             outputs = net(img)
    # normalization
    pred = outputs[-1][:, 0, :, :]
    # pred = normPRED(pred)
    pred = pred.squeeze(0).cpu().data.numpy()
    alpha = Image.fromarray(pred*255).convert('RGB')
    alpha=alpha.resize((W,H))
    savePath=os.path.join(saveDir,file)

    alpha.save(savePath)


