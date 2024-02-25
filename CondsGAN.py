import sys
import numpy as np
import random
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import pickle

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('torch version:',torch.__version__)
print('device:', device)

with open('conds_rand5.pkl', 'rb') as file:
    train_data = pickle.load(file)
train_tensor = torch.Tensor(train_data.values)
with open('conds_rand5.pkl', 'rb') as file:
    test_data = pickle.load(file)
train_size = train_tensor.size()[0]
test_tensor = torch.Tensor(test_data.values)

points = 5
conds = 6
noise_size = points * 2

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(points*2 + conds, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        nn.Linear(128, 1),
        nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator().to(device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(points*2+conds, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, points*2),
            )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator().to(device)

# define hyperparameters

lr = 1E-4
num_epochs = 5000
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

counter = 0
loss_list = torch.Tensor([[counter, 0.5, 0.5]])

"""
# Load discriminator and generator models

discriminator.load_state_dict(torch.load('D_CGAN1.pth'))
generator.load_state_dict(torch.load('G_CGAN1.pth'))
"""

t=time.time()
for epoch in range(num_epochs):
    
    train_conds = train_tensor[:, :conds]
    noise_vector = torch.randn(train_size, noise_size).to(device)
    latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
    #print(latent_space_samples.size())
    
    
    #generate samples
    generated_samples = generator(latent_space_samples).to(device)
    generated_data = torch.cat([train_conds, generated_samples], dim = 1)
    #print(generated_data.size())
    
    generated_samples_labels = torch.zeros((train_size,1))
    real_samples_labels = torch.ones((train_size,1)).to(device)
    

    all_samples = torch.cat((train_tensor, generated_data)).to(device)
    all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(device)


    # Training the discriminator
    discriminator.zero_grad()
    output_discriminator = discriminator(all_samples).to(device)
    
    #print(output_discriminator.max())
    D_real = torch.mean(output_discriminator[:15])
    D_fake = torch.mean(output_discriminator[:-15])
    #calc loss and optimise
    

    loss_discriminator = loss_function(output_discriminator, all_samples_labels).to(device)
    loss_discriminator.backward()
    optimizer_discriminator.step()

    # Data for and training of the generator

    generator.zero_grad()
    noise_vector = torch.randn(train_size, noise_size).to(device)
    latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
    #generate samples    
    generated_samples = generator(latent_space_samples).to(device)
    generated_data = torch.cat([train_conds, generated_samples], dim = 1)
    #output_discriminator_generated = discriminator(torch.cat((generated_samples, the_r),1))
    output_discriminator_generated = discriminator(generated_data).to(device)

    #calc loss and optimise
    loss_generator = loss_function(output_discriminator_generated, real_samples_labels).to(device)
    loss_generator.backward()
    optimizer_generator.step()
    
    
    # Create a list of tensors with counter, loss_g, and loss_d
    loss_g = loss_generator.detach()
    loss_d = loss_discriminator.detach()

    loss_entry = torch.tensor([[counter, loss_g, loss_d]])

    # Append the new entry to the list of losses
    loss_list = torch.cat((loss_list,loss_entry))
    #print(loss_entry)

    counter += 1
    print("epoch: {} lossD: {} lossG: {}".format(epoch, loss_d, loss_g))
    # Show loss

    if epoch % 5 == 1 and epoch !=1:
        e = epoch
        print(f"Epoch: {e} loss D: {loss_discriminator}")
        print(f"Epoch: {e} loss G: {loss_generator}")
        t_e = 100*(time.time()-t)/epoch
        print("Epoch: {} time: {}".format(epoch,t_e))

        
        
        print(loss_list)
        loss_list = loss_list.to(cpu).numpy()
        plt.figure(figsize=(16,16))
        plt.subplot(2,2,1)
        #plt.title.set_text("Discriminator loss")
        plt.plot(loss_list[:,0], loss_list[:,1])
        plt.subplot(2,2,2)
        #plt.title.set_text("Generator loss")
        plt.plot(loss_list[:,0], loss_list[:,2])
        
        loss_list = torch.tensor(loss_list).to(device)
        
        noise_vector = torch.randn(train_size, noise_size).to(device)
        latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
        #generate samples    
        generated_points = generator(latent_space_samples).to(device)
        
        
        
        points = generated_points.detach().to(cpu)
        points1 = np.array(points[510])
        xs1 = points1[:5]
        ys1 = points1[5:]
        points2 = np.array(train_points[510])
        xs2 = points2[:5]
        ys2 = points2[5:]
        
        print(xs1)
        print(ys1)
        plt.subplot(2,2,3)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        
        
        plt.plot(xs1, ys1, ".")
        
        
        plt.subplot(2,2,4)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        
        plt.plot(xs2, ys2, ".")
        plt.show()
        #plt.savefig(f'testset_{e}.png')
        plt.close('all')
