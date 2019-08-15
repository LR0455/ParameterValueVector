import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# LSTM model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ParameterValueVector():
    def __init__(self, window_size, input_size, hidden_size,
                       num_layers, num_classes, num_epochs, batch_size):   
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.window_size = window_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
    
        # path
        self.model_dir = 'model'
        self.log = 'Adam_batch_size=' + str(self.batch_size) + ';epoch=' + str(self.num_epochs)
    
        # train variable
        self.data = None
        self.pvv = None 
        self.write = None

        # model variable
        self.model = None
        self.seq_dataset = None
        self.dataloader = None
        self.criterion = None
        self.optimizer = None
        self.train_loss = None
        self.model_path = None

        # test variable
        self.test = None
        self.predicted = None
        self.threshold = 4

    def generate_tensor_dataset(self, pvv):
        inputs = []
        outputs = []   
        window_size = self.window_size

        for i in range(len(pvv) - window_size):
            inputs.append(pvv[i : i + window_size])
            outputs.append(pvv[i + window_size])

        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
        return dataset
    
    def model_preprocess(self, key, data):
        self.num_classes = len(data[key][0])
        self.input_size = len(data[key][0])
       
        self.model = Model(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)
        self.seq_dataset = self.generate_tensor_dataset(data[key])
        self.dataloader = DataLoader(self.seq_dataset, batch_size = self.batch_size, pin_memory = True)

        # Loss and optimizer
        self.criterion = nn.MSELoss() 
        self.optimizer = optim.Adam(self.model.parameters())
    
    def forward_and_backward(self, seq, label):       
        # Forward pass
        seq = seq.clone().detach().view(-1, self.window_size, self.input_size).to(self.device)
        output = self.model(seq)
        loss = self.criterion(output.float(), label.float().to(self.device))
    
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.train_loss += loss.item()
        self.optimizer.step()

    def save_model(self, key):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), self.model_dir + '/' + str(key) + '.pt')
    
    def get_model(self, key):
        self.model_path = 'model/' + str(key) + '.pt'  
        self.model = Model(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
     
    def key_merge(self, raw_data):
        merge_data = {}
        for data in raw_data:
            key = data['key'] 
            pvv = data['pvv']
            if key not in merge_data:
                merge_data[key] = pvv
            else:
                merge_data[key] += pvv

        return merge_data    

    def pvv_model_predict(self, raw_data):
    
        data = self.key_merge(raw_data)

        for key in data:           
            self.num_classes = len(data[key][0])
            self.input_size = len(data[key][0])
    
            print("log key : ", key, " predicting ...")
            
            self.get_model(key)
            
            self.test = data[key]

            self.criterion = nn.MSELoss()
        
            # Test the model
            with torch.no_grad():
                for i in range(len(self.test) - self.window_size):
                    seq = self.test[i : i + self.window_size]
                    label = self.test[i + self.window_size]
                    
                    seq = torch.tensor(seq, dtype = torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    predicted = self.model(seq).view(-1)

                    loss = self.criterion(predicted.float(), label.float().to(self.device))
                    if loss.item() >= self.threshold:
                        print("predicted :", predicted)
                        print("label :", label)          
                        print("loss :", loss.item())
                        print("log key :", key, "->", label, "is anomaly")
     
            print("log key :", key, "predict finish")
    
        print("Finished Predicting")
    
    def pvv_model_train(self, raw_data):

        data = self.key_merge(raw_data)
        self.data = data.copy()

        for key in data:
            self.model_preprocess(key, data)

            self.writer = SummaryWriter(logdir='log/' + self.log)

            print(key, " training ...")

            for epoch in range(self.num_epochs):  # Loop over the dataset multiple times
                self.train_loss = 0
                for (seq, label) in self.dataloader:
                    self.forward_and_backward(seq, label)
                   
                print('Epoch [{}/{}], Train_loss: {:.4f}'.format(epoch + 1, self.num_epochs, self.train_loss / len(self.dataloader.dataset)))
                self.writer.add_scalar('train_loss', self.train_loss / len(self.dataloader.dataset), epoch + 1)
            
            self.save_model(key)
            self.writer.close()
            
            print(key, ' Finished Training')    
        