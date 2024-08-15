import torch
from torch.utils.data import DataLoader, Dataset
from transformers import EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer
from tqdm import tqdm

# Example dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Autoencoder: input is the same as output

# Example training function
def train(model, tokenizer, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_texts, _ = batch
        input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    return total_loss / len(train_loader)

# Define parameters
train_texts = ["This is an example sentence.", "Another example sentence.", "Yet another example."]
batch_size = 2
num_epochs = 3
learning_rate = 1e-4
model_name = "Helsinki-NLP/opus-mt-en-fr"  # Pre-trained model to initialize encoder and decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = EncoderDecoderConfig.from_encoder_decoder_pretrained(model_name, model_name)
model = EncoderDecoderModel(config).to(device)

# Prepare dataset and data loader
dataset = MyDataset(train_texts)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train(model, tokenizer, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

# Save the trained model
output_model_path = "autoencoder_model"
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)