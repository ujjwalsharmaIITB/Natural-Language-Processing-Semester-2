import torch
from torch.utils.data import DataLoader, Dataset
from transformers import EncoderDecoderModel, EncoderDecoderConfig, AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F

# Example dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Autoencoder: input is the same as output

# Siamese Autoencoder model
class SiameseAutoencoder(torch.nn.Module):
    def __init__(self, encoder_decoder_model):
        super(SiameseAutoencoder, self).__init__()
        self.encoder_decoder_model = encoder_decoder_model

    def forward(self, input_ids1, input_ids2):
        outputs1 = self.encoder_decoder_model(input_ids1)
        outputs2 = self.encoder_decoder_model(input_ids2)
        return outputs1, outputs2

# Contrastive loss function
def contrastive_loss(outputs1, outputs2, margin=1):
    # Calculate Euclidean distance between the encoded representations
    distance = F.pairwise_distance(outputs1, outputs2)
    # Calculate contrastive loss
    loss_contrastive = torch.mean((1 - distance) ** 2)  # Squared hinge loss
    return loss_contrastive

# Example training function
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_texts, _ = batch
        input_ids1 = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        input_ids2 = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        
        optimizer.zero_grad()
        
        outputs1, outputs2 = model(input_ids1, input_ids2)
        loss = contrastive_loss(outputs1.last_hidden_state.mean(dim=1), outputs2.last_hidden_state.mean(dim=1))
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
encoder_decoder_model = EncoderDecoderModel(config).to(device)

# Create Siamese Autoencoder model
model = SiameseAutoencoder(encoder_decoder_model)

# Prepare dataset and data loader
dataset = MyDataset(train_texts)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

# Save the trained model
output_model_path = "siamese_autoencoder_model"
encoder_decoder_model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)