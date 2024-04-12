from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy

##1. Visualize original img and transform img
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

##2. Visualize u by PCA
def PCA_visualize(u,jump=False):
    data = u.detach().numpy()
    if jump: 
        data = data[:, ::2]
    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(data[0])
    
    sizes = np.arange(len(data[0])) + 10
    #print(sizes)

    plt.scatter(pca_data[:, 0], pca_data[:, 1], s=sizes,alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Visualization of Data after PCA')
    plt.show()

##3. add noise
def noise_gaussian(X,tol = 1):
    i=1
    kl_div = 99
    while kl_div > tol and i < 5:      #for reconstruction, i < 10; for remove noise,i<5
        alpha = np.exp(-i/5)
        rand = np.random.rand(X.size(0), X.size(1),X.size(2))
        X = np.sqrt(alpha) * X + np.sqrt(1-alpha) * rand
        i+=1
        #print(i)
        kl_div = entropy(rand.flatten(),X.flatten())

    return torch.tensor(X).float()

##4.recover noise
def noiseRecover(test_loader,model,device):   #从modeltrain里return model
    for i, (x, y) in enumerate(test_loader):
        img = x.to(device)
        img = img.squeeze()
        noise = noise_gaussian(img)
        out_img,u = model(img)
        noise_out,u = model(noise)
    noise_image = noise.cpu().data
    real_image = img.cpu().data
    out_image = out_img.cpu().data
    noise_out = noise_out.cpu().data

    fig = plt.figure()

    for i in range(4):
        plt.subplot(4,4,4*i+1)
        plt.tight_layout()
        plt.imshow(real_image[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,4*i + 2)
        plt.tight_layout()
        plt.imshow(noise_image[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,4*i + 3)
        plt.tight_layout()
        plt.imshow(out_image[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,4*i + 4)
        plt.tight_layout()
        plt.imshow(noise_out[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

    plt.show()

##5. real noise by model reconstruct
def noiseRecover_1(test_loader,model,device):   #从modeltrain里return model
    for i, (x, y) in enumerate(test_loader):
        img = x.to(device)
        img = img.squeeze()
        X = np.random.rand(img.size(0), img.size(1),img.size(2))
        noise = torch.tensor(X).float()
        out_img,u = model(img)
        noise_out,u = model(noise)
    noise_image = noise.cpu().data
    real_image = img.cpu().data
    out_image = out_img.cpu().data
    noise_out = noise_out.cpu().data

    fig = plt.figure()

    for i in range(4):
        plt.subplot(4,4,4*i+1)
        plt.tight_layout()
        plt.imshow(real_image[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,4*i + 2)
        plt.tight_layout()
        plt.imshow(noise_image[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,4*i + 3)
        plt.tight_layout()
        plt.imshow(out_image[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4,4,4*i + 4)
        plt.tight_layout()
        plt.imshow(noise_out[i], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])

    plt.show()
