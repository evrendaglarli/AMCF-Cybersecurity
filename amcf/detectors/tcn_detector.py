"""
Temporal Convolutional Network detector
Part of ML ensemble baseline (Section 5.2)
"""

import numpy as np
import torch
import torch.nn as nn
from .base_detector import BaseDetector

class TemporalConvNet(nn.Module):
    """TCN architecture for sequence modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.temporal_conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.temporal_conv(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        return self.sigmoid(x).squeeze()


class TCNDetector(BaseDetector):
    """Temporal Convolutional Network detector"""
    
    def __init__(self, detector_id: str = 'tcn', input_dim: int = 50):
        super().__init__(detector_id)
        self.model = TemporalConvNet(input_dim=input_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50):
        """Train TCN model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw scores"""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            scores = self.model(X_t).cpu().numpy()
        
        return scores