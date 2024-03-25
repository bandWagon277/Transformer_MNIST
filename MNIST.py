import ENCODER_1
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def fileRead(batch_size):
    data_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
    train_dataset = Subset(train_dataset,list(range(1, 20001)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader     

def modelTrain(model,batch_size,epoches,lr,device):
    train_loader,test_loader = fileRead(batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    fake_image_l = []
    real_image_l = []

    for epochs in range(epoches):
        print("The {} th epoch is:".format(epochs + 1))
        for i, (x, y) in enumerate(train_loader):
            img = x.to(device)
            img = img.squeeze()
            out_img,u = model(img)
            loss = criterion(out_img, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 250 == 0:
                print("epochs:[{}],iteration:[{}]/[{}],loss:{:.3f}".format(epochs, i, len(train_loader), loss.float()))

            # save the last group img
        fake_image = out_img.cpu().data
        real_image = img.cpu().data
        fake_image_l.append(fake_image)
        real_image_l.append(real_image)
    return fake_image_l,real_image_l,u


def visualize(fake_image_l,real_image_l,epoch):
    fig = plt.figure()

    for i in range(4):
        plt.subplot(2,4,2*i+1)
        plt.tight_layout()
        plt.imshow(fake_image_l[epoch - 1][i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,4,2*i + 2)
        plt.tight_layout()
        plt.imshow(real_image_l[epoch - 1][i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

    plt.show()

def PCA_visualize(u):
    data = u.detach().numpy()
    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(data[0])

    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Visualization of Data after PCA')
    plt.show()



def main(d=28,k=12,h=8,p=100,L=6,pos = False,contact=False,m=7,col=True,overlap=False,stride=3,
         batch_size=20,epoches = 5,lr = 0.001,device = torch.device("cpu"),u_visualize = False):
    #redefine parameters for encoder transform
    ENCODER_1.k = k
    ENCODER_1.h = h
    ENCODER_1.d = d
    ENCODER_1.p = p
    ENCODER_1.L = L
    ENCODER_1.pos = pos
    
    if contact:
        #redefine parameters for contact encoder transform
        ENCODER_1.contact = True
        block_num = int((28 - m) / stride + 1) if overlap else int(28/m)
        ENCODER_1.d = m if col else (block_num)**2 * m
        ENCODER_1.m = m
        ENCODER_1.col = col
        ENCODER_1.overlap = overlap
        ENCODER_1.stride = stride

    model = ENCODER_1.ContactEncoder().to(device)
    F_img,R_img,u = modelTrain(model,batch_size,epoches,lr,device)
    PCA_visualize(u) if u_visualize else visualize(F_img,R_img,epoches)


'''
def pos_encode(d=28,k=12,h=8,p=100,L=6, batch_size=20,epoches = 2,lr = 0.001,device = torch.device("cpu")):
    ENCODER_1.k = k
    ENCODER_1.h = h
    ENCODER_1.d = d
    ENCODER_1.p = p
    ENCODER_1.L = L

    model = ENCODER_pos.Encoder().to(device)
    F_img,R_img = modelTrain_2(model,batch_size,epoches,lr,device)
    visualize(F_img,R_img,epoches)

    

def pca_visual_u(d=28,k=12,h=8,p=100,L=6,contact=False,m=7,col=True,overlap=False,stride=3,
         batch_size=20,epoches = 5,lr = 0.001,device = torch.device("cpu")):
    #redefine parameters for encoder transform
    ENCODER_1.k = k
    ENCODER_1.h = h
    ENCODER_1.d = d
    ENCODER_1.p = p
    ENCODER_1.L = L
    
    if contact:
        #redefine parameters for contact encoder transform
        ENCODER_1.contact = contact
        block_num = int((28 - m) / stride + 1) if overlap else int(28/m)
        ENCODER_1.d = m if col else (block_num)**2 * m
        ENCODER_1.m = m
        ENCODER_1.col = col
        ENCODER_1.overlap = overlap
        ENCODER_1.stride = stride

    model = ENCODER_1.ContactEncoder().to(device)
    F_img,R_img,u = modelTrain(model,batch_size,epoches,lr,device)
    PCA_visualize(u)
'''
#main(u_visualize = True)

#main(contact=True,col=False,overlap=True,epoches=1)

#pos_encode()
#main(d=98,contact = False,u_visualize = True)
