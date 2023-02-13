import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np

from network.sade_network import SADE
from preprocess.preprocess_for_sade import MyDataset


class SadeTrainer(object):
    def __init__(self, in_features, out_features, encoder_hidden, decoder_hidden, 
                 data_pca_file="./result/matrix_pca.txt",
                 batch_size=32,
                 lr=0.0001,
                 epoch=10,
                 sade_output_file="./result/SADE_out.txt",
                 ):
        # initialize parameters
        self.in_features = in_features
        self.out_features = out_features
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.data_pca_file = data_pca_file
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.sade_output_file = sade_output_file
        
        # initialize model 
        self.network()
        
        # generate dataloader
        self.generate_dataloader()
        
    def network(self):
        self.model = SADE(in_features=self.in_features, out_features=self.out_features, 
                          encoder_hidden=self.encoder_hidden, decoder_hidden=self.decoder_hidden)
    
    def generate_dataloader(self):
        dataset = MyDataset(data_pca_file=self.data_pca_file)
        self.loader = DataLoader(dataset, batch_size=self.batch_size)
    
    
    def start_train(self):
        optimizer = Adam(lr=self.lr, params=self.model.parameters())
        for epoch in range(self.epoch):
            epoch_mse = []
            self.output = []
            for batch, data in enumerate(self.loader):
                data = data.to(torch.float32)
                out = self.model(data)
                loss = nn.functional.mse_loss(data, out)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_mse.append(loss.detach().numpy())
                self.output.append(out.detach().numpy())
            print("Epoch {:4d} : average mse = {:5f}".format(epoch, np.mean(epoch_mse)))

    def save_data(self):
        # print(np.concatenate(self.output).shape)
        np.savetxt(self.sade_output_file, np.concatenate(self.output))
    
    
if __name__ =="__main__":
    trainer = SadeTrainer(in_features=100, out_features=100, encoder_hidden=[64, 32], decoder_hidden=[32, 64], epoch=1)
    trainer.start_train()
    trainer.save_data()