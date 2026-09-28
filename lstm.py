import torch
import torch.nn as nn            
import torch.nn.functional as F             
import torchvision  
import torch.utils.data as data             
import torch.optim as optim     
from torchvision import transforms      
from torchvision.utils import make_grid
from dataclasses import dataclass 
from torch.utils.data import DataLoader



@dataclass 
class LSTMConfig:
    device = torch.device("gpu") if torch.cuda.is_available()else torch.device("cpu")
    no_of_neurons = 128
    batch_size = 32
    block_size = 32
    vocab_size = 100
    max_lr = 1e-5
    epochs = 200 
    drop_out = 0.5
    embedding_dim :int = 784


torch.set_default_device("cpu")

def set_seed(seed):
    torch.manual_seed(seed)
set_seed(42)



num_samples = 10000
seq_length = LSTMConfig.block_size 
device = LSTMConfig.device

t = torch.linspace(0, 100, num_samples + seq_length, device = device)
total = torch.sin(t)
x_tensor = torch.stack([total[i:i + seq_length] for i in range(num_samples)])
y_tensor = total[seq_length:]

num_train = int(0.8 * num_samples)
x_train, y_train = x_tensor[:num_train], y_tensor[:num_train]
x_val, y_val = x_tensor[num_train:], y_tensor[num_train:]



class Lstm_Dataset(data.Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target         


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]

        return input, target        




train_dataset = Lstm_Dataset(x_train, y_train)
val_dataset = Lstm_Dataset(x_val, y_val)


train_loader = DataLoader(train_dataset, batch_size = LSTMConfig.batch_size, shuffle = True, pin_memory_device = True)
val_loader = DataLoader(val_dataset, batch_size = LSTMConfig.batch_size, shuffle = False, pin_memory_device = True)



class input_gate(nn.Module):
    def __init__(self, device, no_of_neurons):
        super().__init__()

        self.it = nn.Linear(in_features = LSTMConfig.no_of_neurons + LSTMConfig.embedding_dim, out_features = no_of_neurons, device = device)
        self.ct = nn.Linear(in_features = LSTMConfig.no_of_neurons + LSTMConfig.embedding_dim, out_features= no_of_neurons, device = device)


    def forward(self, x, ht_1):
        r = torch.cat([x, ht_1], dim = 1)
        combined = torch.nn.functional.sigmoid(self.it(r))
        totalls = torch.nn.functional.tanh(self.ct(r))
        return combined, totalls 




class output_gate(nn.Module):
    def __init__(self,device, no_of_neurons):
        super().__init__()

        self.it = nn.Linear(in_features = LSTMConfig.no_of_neurons + LSTMConfig.embedding_dim, out_features=no_of_neurons, device = device)
        

    def forward(self, x, ht_1):
        
        h = torch.cat([x, ht_1], dim = 1)
        total = torch.nn.functional.sigmoid(self.it(h))
        return total


class forget_gate(nn.Module):
    def __init__(self, device, no_of_neurons):
        super().__init__()

        self.linear = nn.Linear(in_features=LSTMConfig.no_of_neurons + LSTMConfig.embedding_dim, out_features = no_of_neurons, device = device)


    def forward(self, x, ht_1):
        v = torch.cat([x, ht_1], dim = 1)
        t = torch.nn.functional.sigmoid(self.linear(v))
        return t         




class LSTM_Block(nn.Module):
    def __init__(self, device, no_of_neurons):
        super().__init__()


        self.input = input_gate(device = device, no_of_neurons=no_of_neurons)
        self.output = output_gate(device = device, no_of_neurons = no_of_neurons)
        self.forget = forget_gate(device = device, no_of_neurons = no_of_neurons)
        self.neurons = no_of_neurons
        self.device = device 
        self.linears = nn.Linear(in_features = LSTMConfig.no_of_neurons, out_features=LSTMConfig.embedding_dim,device = device, dtype = torch.float32)

    def forward(self, x, ht_prev = None, outputs = None, embeds = None):

        if(ht_prev is None):
            ht_prev = torch.randn( (x.shape[0], self.neurons), device = self.device, requires_grad=True, dtype = torch.float32)

        ct_prev = torch.randn((x.shape[0], self.neurons), device = self.device, requires_grad=True, dtype = torch.float32)
        seq_len = x.shape[1]

        if(outputs == None):

            outputs = []
            

            for t in range(seq_len):

                xt = x[:, t, :]
                ft = self.forget(xt, ht_prev) 
                it, ct_bar = self.input(xt, ht_prev)
                cs = ft * ct_prev + it * ct_bar

                ht = self.output(xt, ht_prev) * torch.nn.functional.tanh(cs)
                outputs.append(ht)

            return ht, cs, torch.stack(outputs, dim = 1)
        elif (outputs is not None):
            

            new_output = []
            for t in range(seq_len):
                xt = outputs[:, t, :]
                xt = self.linears(xt)
                ft = self.forget(xt, ht_prev)
                it, ct_bar = self.input(xt, ht_prev)
                cs = ft * ct_prev + it * ct_bar
                ht = self.output(xt, ht_prev) * torch.nn.functional.tanh(cs)
                new_output.append(ht)

            return ht, cs, torch.stack(new_output, dim = 1)



class LSTM(nn.Module):
    def __init__(self,  vocab_size, embedding_dim, device, no_of_neurons, out_features):
        super().__init__()
        self.block = LSTM_Block(device = device, no_of_neurons = no_of_neurons)

        self.dropout = nn.Dropout(p = LSTMConfig.drop_out)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.output = nn.Linear(in_features = LSTMConfig.no_of_neurons, out_features=out_features, device = device, dtype = torch.float32)


    def forward(self, x):
        x = self.embedding(x.long())
        ht, ct, outputs = self.block(x)
        return self.output(self.dropout(ht))
    


Model = LSTM(device = LSTMConfig.device, no_of_neurons= LSTMConfig.no_of_neurons, out_features = 1, embedding_dim = LSTMConfig.embedding_dim, vocab_size = LSTMConfig.vocab_size)
model = Model.to(LSTMConfig.device)
model


from torchinfo import summary

x = torch.randint(0, 100, (LSTMConfig.batch_size, LSTMConfig.block_size))
x = x.to(LSTMConfig.device)
summary(
    model = model,
    input_data = x, 
    col_names = ["input_size", "output_size", "num_params", "trainable"],
        col_width = 20,
        row_settings = ["var_names"]
)







criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr = LSTMConfig.max_lr)
model.train()
train_loss = torch.zeros(len(train_loader))
val_loss = torch.zeros(len(val_loader))

for epoch in range(LSTMConfig.epochs):
    count = 0
for x, y in train_loader:
    pred = model(x)
    loss = criterion(pred, y)
    train_loss[count] = loss.item()


optimizer.zero_grad()
loss.backward()
optimizer.step()
count += 1

print("epoch", epoch, "|", "step", count, "|", loss.item())


model.eval()
count = 0



for x, y in val_loader:
    pred = model(x)
    loss = criterion(pred, y)
    val_loss[count] = loss.item()



    count += 1 

print("epoch", epoch, "|","step", count, "|", loss.item())
model.train()

print("epoch", epoch, "|", "Train Loss:", train_loss.mean(), "|", "val_loss:", val_loss.mean())