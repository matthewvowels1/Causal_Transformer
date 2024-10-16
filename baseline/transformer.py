import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, n_layers=2, n_heads=2, device='cuda'):
        super(TransformerRegressor, self).__init__()
        # Define an encoder layer with the hidden_dim (embedding dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        # Stack the encoder layers into a TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Project input_dim to hidden_dim for input embeddings
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Fully connected layer for final regression output (projecting hidden_dim to 1)
        self.fc = nn.Linear((input_dim) * hidden_dim, 1)  # Output a scalar per input sample
        self.device = device

    def forward(self, x):
        # Project input_dim to hidden_dim
        x = self.embedding(x)  # (batch_size, num_variables, hidden_dim)
        # Pass input through the transformer encoder
        out = self.transformer_encoder(x)  # (batch_size, num_variables, hidden_dim)
        # Flatten the sequence dimension and pass through fully connected layer
        out = out.view(out.size(0), -1)  # Flatten to (batch_size, num_variables * hidden_dim)
        # Fully connected layer to output a prediction
        out = self.fc(out)  # Adjust self.fc to match the flattened size
        return out.squeeze(-1)  # (N,)
