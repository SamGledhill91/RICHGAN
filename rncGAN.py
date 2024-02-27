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
cpu = 'cpu'

print('torch version:',torch.__version__)
print('device:', device)

with open('rnc_data.pkl', 'rb') as file:
    train_data = pickle.load(file)
train_tensor = torch.Tensor(train_data.values)
print(train_data)

train_size = train_tensor.size()[0]


targets = 3
conds = 6
noise_size = targets

train_conds = train_tensor[:,:conds]
train_targets = train_tensor[:,conds+1:]

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(targets + conds, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 128),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Dropout(0.3),
        # nn.Linear(128, 128),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Dropout(0.3),
        # nn.Linear(128, 128),
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
            nn.Linear(targets+conds, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, targets),
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
loss_list = torch.Tensor([[counter, 0.7, 0.7]])

"""
# Load discriminator and generator models

discriminator.load_state_dict(torch.load('D_CGAN1.pth'))
generator.load_state_dict(torch.load('G_CGAN1.pth'))
"""

t=time.time()
for epoch in range(num_epochs):
    
        
    noise_vector = torch.randn(train_size, noise_size).to(device)
    latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
    #print(latent_space_samples.size())
    
    
    #generate samples
    generated_samples = generator(latent_space_samples).to(device)
    generated_data = torch.cat([train_conds, generated_samples], dim = 1)
    #print(generated_data.size())
    
    
    generated_samples_labels = torch.zeros((train_size,1))
    real_samples_labels = torch.ones((train_size,1)).to(device)
    
    train_tensor = torch.cat([train_conds, train_targets], axis =1)
    #print(train_tensor.size())
    #print(generated_data.size())
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
    #print("epoch: {} lossD: {} lossG: {}".format(epoch, loss_d, loss_g))
    # Show loss

    if epoch % 5 == 1 and epoch !=1:
        e = epoch
        print("Epoch: {} loss D: {}".format(e, loss_discriminator))
        print("Epoch: {} loss G: {}".format(e, loss_generator))
        t_e = 100*(time.time()-t)/epoch
        print("Epoch: {} time: {}".format(epoch,t_e))
        
        noise_vector = torch.randn(train_size, noise_size).to(device)
        latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
        generated_points = generator(latent_space_samples).to(device)
        
        # Assuming train_targets and generated_points are both tensors with 3 rows and N columns
        train_targets_first_column = train_targets[:, 0]
        generated_points_first_column = generated_points[:, 0]
        
        # Calculate the difference
        difference =  generated_points_first_column - train_targets_first_column
        
        # Create a figure and two subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        
        # Plot the first histogram (Radius Difference)
        axes[0,0].hist(difference.cpu().detach().numpy(), bins=50)
        axes[0,0].set_xlabel('Difference')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Radius Difference')
        
        # Assuming train_targets and generated_points are both tensors with 3 rows and N columns
        train_targets_second_row = train_targets[:,1]
        generated_points_second_row = generated_points[:,1]
        train_targets_third_row = train_targets[:,2]
        generated_points_third_row = generated_points[:,2]
        
        # Calculate the differences
        difference_second_row = train_targets_second_row - generated_points_second_row
        difference_third_row = train_targets_third_row - generated_points_third_row
        
        # Add the differences in quadrature
        squared_sum = difference_second_row**2 + difference_third_row**2
        rooted_sum = torch.sqrt(squared_sum)
        
        # Plot the second histogram (Centres Difference)
        axes[0,1].hist(rooted_sum.cpu().detach().numpy(), bins=50)
        axes[0,1].set_xlabel('Rooted Sum of Differences')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Centres Difference')
        
        
        
        
        #print(loss_list)
        loss_plot = loss_list.to(cpu).numpy()
        # plt.figure(figsize=(16,16))
        # plt.subplot(2,2,1)
        # plt.title.set_text("Discriminator loss")
        # plt.plot(loss_list[:,0], loss_plot[:,1])
        # plt.subplot(2,2,2)
        # plt.title.set_text("Generator loss")
        # plt.plot(loss_list[:,0], loss_plot[:,2])
        
        axes[1,0].plot(loss_list[:,0], loss_plot[:,1])
        axes[1,0].set_title('Discriminator loss')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].set_xlabel('Epoch')
        
        axes[1,1].plot(loss_list[:,0], loss_plot[:,2])
        axes[1,1].set_title('Generator loss')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].set_xlabel('Epoch')
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        # loss_list = torch.tensor(loss_list).to(device)
        
        # noise_vector = torch.randn(train_size, noise_size).to(device)
        # latent_space_samples = torch.cat([train_conds, noise_vector], dim = 1)
        # #generate samples    
        # generated_points = generator(latent_space_samples).to(device)
        
        
        
        # points = generated_points.detach().to(cpu)
        # points1 = np.array(points[510])
        # xs1 = points1[:5]
        # ys1 = points1[5:]
        # points2 = np.array(train_targets[510])
        # xs2 = points2[:5]
        # ys2 = points2[5:]
        
        # #print(xs1)
        # #print(ys1)
        # plt.subplot(2,2,3)
        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        
        
        # plt.plot(xs1, ys1, ".")
        
        
        # plt.subplot(2,2,4)
        # plt.xlim(-1,1)
        # plt.ylim(-1,1)
        
        # plt.plot(xs2, ys2, ".")
        # plt.show()
        # #plt.savefig(f'testset_{e}.png')
        # plt.close('all')
