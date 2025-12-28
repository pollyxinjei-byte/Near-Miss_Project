# src/model.py
import torch
import torch.nn as nn

class VesselLSTM(nn.Module):
    """
    Module 1: AI Prediction Layer (LSTM)
    Predicts the relative displacement (Delta) of the vessel.
    This corresponds to the architecture described in Chapter 4.2.1.
    """
    def __init__(self, input_size=4, hidden_size=128, output_size=2, dropout=0.2):
        super(VesselLSTM, self).__init__()
        
        # LSTM with 2 layers and dropout (Table 4.2)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True, 
            num_layers=2,
            dropout=dropout  # Added: Regularization between layers
        )
        
        # Fully connected layer to output Delta Latitude and Delta Longitude
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        out, _ = self.lstm(x)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out