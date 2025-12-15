from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import random
from model import LatentReasoningModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class StructuredTokenDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = []

        for _ in range(num_samples):
            start = random.randint(0, vocab_size-1)
            # create a deterministic sequence: next token = (prev + 1) % vocab_size
            seq = [(start + i) % vocab_size for i in range(seq_len)]
            self.data.append(torch.tensor(seq))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = seq[:-1]  # input
        y = seq[1:]   # target: next token
        return x, y
    

seq_len = 20
vocab_size = 52
batch_size = 32
num_samples = 2000

dataset = StructuredTokenDataset(num_samples, seq_len, vocab_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 5
n_layers = 2
latent_dim = 64
attn_dim = 64
input_length = seq_len-1
n_heads = 4
dropout = 0.1
lr = 0.001



model = LatentReasoningModel(
    n_layers=n_layers,
    latent_dim=latent_dim,
    attn_dim=attn_dim,
    input_length=input_length,
    n_heads=n_heads,
    vocab_size=vocab_size,
    dropout=dropout
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    losses = []
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, vocab_size = logits.shape
        logits = logits.view(B*T, vocab_size)
        y = y.view(B*T)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"{i} / {len(dataloader)}")
            with torch.no_grad():
                logits = model(x)
                logits = logits[0]
                preds = logits.argmax(dim=-1)
                y_first = y.view(B, T)[0]
                print(f"Prediction: {preds}\nReal: {y_first}")
    
    print(f"Epoch {epoch} Loss: {sum(losses)/len(losses)}")



data_iter = iter(dataloader)

# grab the first batch
x_test, y_test = next(data_iter)

# move to device
x_test, y_test = x_test.to(device), y_test.to(device)

with torch.no_grad():
    logits_seq = logits.view(B, T, vocab_size)  # reshape back to (B, T, vocab_size)
    preds_seq = logits_seq.argmax(dim=-1)

    # take first sequence in batch
    preds_first = preds_seq[0]
    y_first = y.view(B, T)[0]

    print(f"Prediction: {preds_first}")
    print(f"Real:       {y_first}")