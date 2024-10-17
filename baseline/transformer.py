import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    def __init__(self, num_variables, input_dim, hidden_dim=16, n_layers=2, n_heads=2, device='cuda'):
        super(TransformerRegressor, self).__init__()
        # Define an encoder layer with the hidden_dim (embedding dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim,
                                                   batch_first=True)
        # Stack the encoder layers into a TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Project input_dim to hidden_dim for input embeddings
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Fully connected layer for final regression output (projecting hidden_dim to 1)
        self.fc = nn.Linear(num_variables * hidden_dim, 1)  # Output a scalar per input sample
        self.device = device
        self.criterion = nn.MSELoss()
        self.to(device)

    def forward(self, X: torch.tensor, targets=None):
        if X.dim() == 2:
            X = X.unsqueeze(-1)

        # Project input_dim to hidden_dim
        X = self.embedding(X)  # (batch_size, num_variables, hidden_dim)
        # Pass input through the transformer encoder
        out = self.transformer_encoder(X)  # (batch_size, num_variables, hidden_dim)
        # Flatten the sequence dimension and pass through fully connected layer
        out = out.view(out.size(0), -1)  # Flatten to (batch_size, num_variables * hidden_dim)
        # Fully connected layer to output a prediction
        out = self.fc(out)  # Adjust self.fc to match the flattened size
        out = out.squeeze(-1)  # (N,)
        if targets is None:
            return out
        else:
            loss = self.criterion(out, targets)
            loss_tracking = {}
            return out, loss, loss_tracking
