import torch.nn as nn  
import torch     
import torch.nn.functional as F      
import torchvision     
from torchvision import transforms  
import torch.utils.data as data            
from torch.utils.data import Dataset, DataLoader   
from collections import defaultdict   
from dataclasses import dataclass

@dataclass 
class GRUConfig:
    device = torch.device("cuda") if torch.cuda.is_available()else torch.device("cpu")
    no_of_neurons = 16
    block_size = 16
    batch_size = 16
    drop_out = 0.1
    embedding_dim :int = 768
    max_lr = 1e-6
    epochs = 100




num_samples = 1000
seq_length = GRUConfig.block_size
device = GRUConfig.device 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

torch.backends.cudnn.benchmark = True    
torch.backends.cudnn.deterministic = True   



t = torch.linspace(0, 100, num_samples + seq_length, device = GRUConfig.device)
total = torch.sin(t)

x_tensor = torch.stack([total[i:i + seq_length] for i in range (num_samples)])
y_tensor = total[seq_length:]

train_size = int(0.8 *num_samples)
x_train, y_train = x_tensor[:train_size], y_tensor[:train_size]
x_val, y_val = x_tensor[train_size:], y_tensor[train_size:]



class Dataset(data.Dataset):
    def __init__(self, input, target):
        super().__init__()

        self.input = input 
        self.target = target


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):

        input = self.input[idx]
        target = self.target[idx]

        return input, target         


num_train = Dataset(x_train,y_train)
num_val = Dataset(x_val, y_val)


train_loader = data.DataLoader(num_train, batch_size = GRUConfig.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
val_loader = data.DataLoader(num_val, batch_size = GRUConfig.batch_size, shuffle = False, pin_memory_device = True, num_workers = 0)
print(len(num_val), len(val_loader))



class ResetGate(nn.Module):
    def __init__(self, no_of_neurons, device):
        super().__init__()

        self.linear = nn.Linear(in_features = GRUConfig.no_of_neurons + 1, out_features = no_of_neurons, device = device)




    def forward(self, x, ht_1):
        t_value = torch.cat([x, ht_1], dim = -1)
        t = torch.nn.functional.sigmoid(self.linear(t_value))
        return t              




class Update(nn.Module):
    def __init__(self, no_of_neurons, device):
        super().__init__()

        self.linear = nn.Linear(in_features = GRUConfig.no_of_neurons + 1, out_features = no_of_neurons, device = device)

    def forward(self, x, ht_1):
        s = torch.cat([x, ht_1], dim = -1)
        t_value = torch.nn.functional.sigmoid(self.linear(s))
        return t_value     



class GRUBLOCK(nn.Module):
    def __init__(self, device, no_of_neurons):
        super().__init__()


        self.update = Update(device = device, no_of_neurons = no_of_neurons)
        self.resetgate = ResetGate(device = device, no_of_neurons = no_of_neurons)
        self.candidate = nn.Linear(in_features = GRUConfig.no_of_neurons + 1, out_features = no_of_neurons, device = GRUConfig.device)
        self.linear_layer = nn.Linear(in_features = GRUConfig.no_of_neurons, out_features = 1, device = GRUConfig.device)
        self.no_of_neurons = no_of_neurons  


    def forward(self, x, output = None):
        ht_1 = nn.Parameter(torch.zeros(x.shape[0], self.no_of_neurons, device = device, requires_grad=True,dtype = torch.float32))

        seq_len = x.shape[1]


        if(output is None):
            output = []
            for t in range(seq_len):
                xt = x[:, t].unsqueeze(-1)
                


                reset = self.resetgate(xt, ht_1)  

                ht_2 = nn.functional.tanh(self.candidate(torch.cat([reset * ht_1, xt], dim = -1)))
                update_out = self.update(xt, ht_1)
                ht = ((1 - update_out) * ht_1) + (update_out * ht_2)
                ht_1 = ht 
                output.append(ht_1)
            return ht_1, torch.stack(output, dim = 1)


        elif(output is not None and len(output) != 0):
            new_output = []
            for t in range(seq_len):

                xt = output[:, t]
                xt = self.linear_layer(xt)


                resets = self.resetgate(xt, ht_1)

                ht_2 = nn.functional.tanh(self.candidate(torch.cat([resets * ht_1, xt],dim = -1)))
                update_out = self.update(xt, ht_1)
                ht = ((1- update_out) * ht_1) + (update_out * ht_2)
                ht_1 = ht 
                new_output.append(ht_1)
            return ht_1, torch.stack(new_output, dim = 1)



class GRU(nn.Module):
    def __init__(self, no_of_neurons, device, out_features):
        super().__init__()

        self.block1 = GRUBLOCK(device = device, no_of_neurons = no_of_neurons)


        self.output = nn.Linear(in_features = GRUConfig.no_of_neurons, out_features = out_features, device = GRUConfig.device)
        self. dropout = nn.Dropout(p = GRUConfig.dropout)


    def forward(self, x):

        ht, hidden_states = self.block1(x)

        ht = self.dropout(ht)
        out = self.output(ht)
        return out     




class DeepGRU(nn.Module):
    def __init__(self, device, no_of_neurons, out_features):
        super().__init__()
        self.block1 = GRUBLOCK(device = device, no_of_neurons = no_of_neurons)
        self.block2 = GRUBLOCK(device = device, no_of_neurons = no_of_neurons)
        self.block3 = GRUBLOCK(device = device, no_of_neurons = no_of_neurons)
        self.linear = nn.Linear(in_features = GRUConfig.no_of_neurons, out_features = out_features,device = GRUConfig.device)
        self.dropout = nn.Dropout(p = GRUConfig.drop_out)

    def forward(self, x):
        ht, hidden_states = self.block1(x)

        ht, hidden_states = self.block2(x, hidden_states)
        ht, hidden_states = self.block3(x, hidden_states)
        ht = self.dropout(ht)
        out = self.linear(ht)
        return out     




model = DeepGRU(no_of_neurons = GRUConfig.no_of_neurons, device = GRUConfig.device, out_features = 1)
model = model.to(GRUConfig.device)
model


import torchinfo   
from torchinfo import summary 

x = torch.randint(0, 100, (GRUConfig.batch_size, GRUConfig.block_size))

summary(
    model = model,
    input_data = x,
    col_names = ["input_size", "output_size", "num_params","trainable"],
    col_width = 40,
    row_settings = ["var_names"]
)

criterion = nn.MSELoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr = GRUConfig.max_lr)



model.train()
train_losses = torch.zeros(len(train_loader))
val_losses = torch.zeros(len(val_loader))


for epoch in range(GRUConfig.epochs):
    count = 0 
    for x, y in train_loader:
      pred = model(x)
      loss = criterion(pred, y)
      train_losses[count] = loss.item()
    

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      count += 1
    


print("epoch", epoch, "|", "step", count,  "|", "train_loss", loss.item())

model.eval()
count = 0

for x, y in val_loader:
    pred = model(x)
    loss = criterion(pred, y)
    val_losses[count] = loss.item()
    count += 1

print("epoch", epoch, "|", "step", count,  "|", "val_loss:", loss.item())

model.train()



print("epoch", epoch, "|", "step", count, "|",  "train_losses:", train_losses.mean(), "val_loss:", val_losses.mean())





