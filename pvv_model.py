import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

from sklearn.decomposition import PCA

# LSTM model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ParameterValueVector():

    # initial 
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

        # Mapping table
        self.table_fti = {}
        self.table_itf = {}
        self.table_ftv = {}
        self.table_kind = {}
        
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
        self.threshold = 10

    def generate_tensor_dataset(self, pvv):
    
        inputs = []
        outputs = []   
        window_size = self.window_size

        for i in range(len(pvv) - window_size):
            inputs.append(pvv[i : i + window_size])
            outputs.append(pvv[i + window_size])
    
        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
        return dataset

    def clear_unnecessary_dimension(self, data):
        data2 = []
        for i in range(len(data)):
            data2.append(data[i][0])
        return data2

    def reduce_data_dimension(self, key, data, test_flag):
        estimator = PCA(n_components = 1)
        transform_data = self.clear_unnecessary_dimension(estimator.fit_transform(data))
        
        if test_flag == False:
            self.table_ftv[key] = {}

            for i in range(len(transform_data)):
                transform_data[i] = round(transform_data[i])
                self.table_ftv[key][transform_data[i]] = data[i]
        else:
            for i in range(len(transform_data)):
                transform_data[i] = round(transform_data[i])
   
        return transform_data
    
    def mapping_float_to_int(self, key, transform_data, test_flag):
        if test_flag == False:
            self.table_kind[key] = 0
            self.table_fti[key] = {}
            self.table_itf[key] = {}

            for i in range(len(transform_data)):
                if transform_data[i] in self.table_fti[key]:
                    transform_data[i] = self.table_fti[key][transform_data[i]]
                else:
                    self.table_itf[key][self.table_kind[key]] = transform_data[i]
                    self.table_fti[key][transform_data[i]] = self.table_kind[key]
                    transform_data[i] = self.table_kind[key]
                    self.table_kind[key] += 1
        else:
            for i in range(len(transform_data)):
                if transform_data[i] in self.table_fti[key]:
                    transform_data[i] = self.table_fti[key][transform_data[i]]
        
        return transform_data
    
    def preprocess(self, key, data, test_flag):
        transform_data = self.reduce_data_dimension(key, data, test_flag)
        transform_data = self.mapping_float_to_int(key, transform_data, test_flag)
        return transform_data
    
    def model_preprocess(self, key, data):
        
        self.num_classes = self.table_kind[key]

        self.model = Model(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)

        self.seq_dataset = self.generate_tensor_dataset(data[key])

        self.dataloader = DataLoader(self.seq_dataset, batch_size = self.batch_size, shuffle = True, pin_memory = True)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
    
    def forward_and_backward(self, seq, label):
         # Forward pass
         seq = seq.clone().detach().view(-1, self.window_size, self.input_size).to(self.device)
         output = self.model(seq)
         loss = self.criterion(output, label.to(self.device))

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
            
            self.num_classes = self.table_kind[key]
    
            print("log key : ", key, " predicting ...")
            
            self.get_model(key)
            
            print("data : ", data[key])

            self.test = self.preprocess(key, data[key], True)
            
            print("test : ", self.test)

            # Test the model
            with torch.no_grad():
                for i in range(len(self.test) - self.window_size):
                    
                    seq = self.test[i:i + self.window_size]
                    label = self.test[i + self.window_size]
                    
                    if label not in self.table_ftv[key]:
                        continue
                    
                    seq = torch.tensor(seq, dtype = torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    
                    output = self.model(seq)
                    predicted = torch.argsort(output, 1)[0][-1:]
                    
                    pdt = self.table_itf[key][int(predicted)]
                    lbl = self.table_itf[key][int(label)]
                    
                    if (pdt - lbl)*(pdt - lbl) >= self.threshold:
                        print("log key : ", key, " ", self.table_ftv[key][lbl], " is anomaly")
     
            print("log key : ", key, " predict finish")
    
        print("Finished Predicting")
    
    def pvv_model_train(self, raw_data):
        
        data = self.key_merge(raw_data)

        self.data = data.copy()

        for key in data:

            data[key] = self.preprocess(key, data[key], False)

            self.model_preprocess(key, data)
            
            self.writer = SummaryWriter(logdir='log/' + self.log)

            print(str(key) + " training ...")
    
            for epoch in range(self.num_epochs):  # Loop over the dataset multiple times
                self.train_loss = 0
                for (seq, label) in self.dataloader:
                    self.forward_and_backward(seq, label)
                   
                print('Epoch [{}/{}], Train_loss: {:.4f}'.format(epoch + 1, self.num_epochs, self.train_loss / len(self.dataloader.dataset)))
                self.writer.add_scalar('train_loss', self.train_loss / len(self.dataloader.dataset), epoch + 1)
            
            self.save_model(key)
            self.writer.close()
            print('Finished Training')    
        