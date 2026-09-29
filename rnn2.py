import torch  
import torch.nn.functional as F       
import torchvision     
import torch.utils.data as  data             
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset  
import torch.nn as nn     



@dataclass 
class RNNConfig:
    device = torch.device("cuda") if torch.cuda.is_available()else torch.device("cpu")
    no_of_neurons = 16
    block_size = 16
    batch_size = 16
    max_lr = 1e-5
    epoch = 200 
    dropout = 0.1
    embedding_dim:int = 784 



torch.set_default_device(RNNConfig.device)
def set_seed(seed):
    torch.manual_seed(seed)
    

set_seed(42)

num_samples = 10000
seq_length = RNNConfig.block_size 
device = RNNConfig.device  

t = torch.linspace(0, 100, num_samples + seq_length, device = device)
total = torch.sin(t)


x_tensor = torch.stack([total[i:i+ seq_length] for i in range(num_samples)])
y_tensor = total[seq_length:]

train_size = int(0.8 * num_samples)
x_train, y_train = x_tensor[:train_size], y_tensor[:train_size]
x_val, y_val = x_tensor[train_size:], y_tensor[train_size:]



class datasets(Dataset):
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




train_dataset = datasets(x_train,y_train)
val_dataset = datasets(x_val, y_val)

generator = torch.Generator(device = device)

train_loader = DataLoader(
    train_dataset, batch_size = RNNConfig.batch_size, shuffle = True, pin_memory = True, generator = generator)

val_loader = DataLoader(
    val_dataset, batch_size = RNNConfig.batch_size, shuffle = False, pin_memory = True, generator = generator

)


class RNNCell(nn.Module):
    def __init__(self, device, no_of_neurons):
        super().__init__()
        self.linear_layer = nn.Linear(in_features = RNNConfig.block_size+1, out_features = no_of_neurons, device  = device)

    def forward(self, x, ht_1):
       x = self.linear_layer(torch.cat([x, ht_1], dim = -1))
       ht = torch.nn.functional.sigmoid(x)
       return ht





class RNNLayer(nn.Module):
    def __init__(self, device, no_of_neurons):
        super().__init__()

        self.rnn = RNNCell(device = device, no_of_neurons=no_of_neurons)
        self.linear_layer = nn.Linear(in_features = RNNConfig.block_size, out_features=no_of_neurons, device = RNNConfig.device)


    def forward(self, x):
        ht_1 = torch.zeros((RNNConfig.block_size, RNNConfig.no_of_neurons), device = RNNConfig.device, requires_grad = True)
        seq_len = x.shape[1]


        for t in range(seq_len):
            xt = x[:, t]
            xt = xt.unsqueeze(-1)
            ht = self.rnn(xt, ht_1)
            ht_1 = ht 
        return ht_1 





class RNN(nn.Module):
    def __init__(self, device, no_of_neurons, out_features):
        super().__init__()
        self.rnn_layer = RNNLayer(device = RNNConfig.device, no_of_neurons = no_of_neurons)
        self.output = nn.Linear(in_features = RNNConfig.no_of_neurons, out_features = out_features, device = device)
        self.dropout = nn.Dropout(p = RNNConfig.dropout)


    def forward(self, x):
        ht = self.rnn_layer(x)
        out = self.output(ht)
        out = self.dropout(out)
        return out         

    








model = RNN(device = RNNConfig.device, no_of_neurons = RNNConfig.no_of_neurons, out_features = 1)
model = model.to(RNNConfig.device)


from torchinfo import summary  

x = torch.randint(0, 100, (RNNConfig.batch_size, RNNConfig.block_size))
x = x.to(RNNConfig.device)
summary(
    model = model,
    input_data = x, 
    col_names = ["input_size", "output_size", "num_params", "trainable"],
    col_width = 20,
    row_settings = ["var_names"])

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(),weight_decay=0.8, lr = RNNConfig.max_lr)
model.train()
train_losses = torch.zeros(len(train_loader))
val_loss = torch.zeros(len(val_loader))

for epoch in range(RNNConfig.epoch):
    count = 0
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y)
        train_losses[count] = loss.item()





        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1


print("epoch", epoch, "|", "step", count, "|", "Train Loss:", loss.item())
model.eval()
count = 0


for x, y in val_loader:
    y_pred = model(x)
    loss = criterion(y_pred, y)
    val_loss[count] = loss.item()


    count += 1 



print("epoch", epoch, "|", "step:", count, "|", "val_loss:", loss.item())
model.train()

print("epoch", epoch, "|","step", count, "|", "Train Loss:", train_losses.mean(), "|", "val_loss:", val_loss.mean())