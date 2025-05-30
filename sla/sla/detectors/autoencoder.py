import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Autoencoder-based anomaly detector module.

This module provides implementation of an autoencoder-based approach
for anomaly detection in time series data.
"""

import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """
    Encoder neural network for anomaly detection.
    
    This class implements an encoder with configurable architecture
    for encoding input features to a latent space representation.

    Args:
            - n_features (int): Dimension of input features

            - seq_len (int): Length of input sequence

            - latent_dim (int): Dimension of the latent space representation
    """
    def __init__(self, n_features, seq_len, latent_dim=20):

        super(Encoder, self).__init__()
        
        self.seq_len, self.n_features = seq_len, n_features
        self.latent_dim = latent_dim
        
        self.rnn = nn.LSTM(
        input_size=n_features,
        hidden_size=latent_dim,
        num_layers=1,
        batch_first=True
        )
        
    def forward(self, x):
        """
        Forward pass of the encoder model.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:

            torch.Tensor: Latent space representation
        """
        x = x.reshape((1, self.seq_len, self.n_features))
        _, (hidden_n, _) = self.rnn(x)
        
        return hidden_n.reshape((self.n_features, self.latent_dim))
    

class Decoder(nn.Module):
    """
    Decoder neural network for anomaly detection.
    
    This class implements a decoder with configurable architecture
    for decoding latent space representation to output features.

    Args:
            - n_features (int): Dimension of output features

            - seq_len (int): Length of output sequence

            - input_dim (int): Dimension of the latent space representation
    """
    def __init__(self, n_features, seq_len, input_dim=20):

        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        """
        Forward pass of the decoder model.
        
        Args:

            x (torch.Tensor): Latent space representation
        
        Returns:

            torch.Tensor: Output features
        """
        x = x.repeat(self.seq_len, self.n_features)
        
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)

class Autoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection.
    
    This class implements an autoencoder with configurable architecture
    for encoding and decoding input features.

    Args:
            - n_features (int): Dimension of input features

            - seq_len (int): Length of input sequence

            - latent_dim (int): Dimension of the latent space representation
    """
    def __init__(self, n_features, seq_len, latent_dim=20):
    
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(n_features, seq_len, latent_dim)
        self.decoder = Decoder(n_features, seq_len, latent_dim)
    
    def forward(self, x):
        """
        Forward pass of the autoencoder model.
        
        Args:

            x (torch.Tensor): Input features
        
        Returns:

            torch.Tensor: Decoded features
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.
    
    This class implements an anomaly detector using an autoencoder
    for detecting anomalies in time series data.

    Args:
            - n_features (int): Dimension of input features

            - seq_len (int): Length of input sequence

            - latent_dim (int): Dimension of the latent space representation

            - learning_rate (float): Learning rate for training the model

            - batch_size (int): Number of samples per batch

            - epochs (int): Number of epochs for training the model

            - threshold_multiplier (float): Anomaly threshold multiplier
    """
    def __init__(self, n_features, seq_len, latent_dim=20, 
                 learning_rate=0.001, batch_size=32, epochs=20, 
                 threshold_multiplier=3.0):

        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold_multiplier = threshold_multiplier
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(n_features, seq_len, latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='sum')
        self.scaler = StandardScaler()
        self.threshold = None
        self.history = {'train': [], 'val': []}
        
    def fit(self, X, X_test=None):
        """
        Fit the anomaly detector to the input data.
        
        Args:

            X (np.ndarray): Input data for training the model
        """

        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32) if X_test is not None else None
        
        self.model = self.model.to(self.device)
        
        best_loss = np.inf
        
        for epoch in range(self.epochs):
            self.model = self.model.train()

            train_losses = []
            for seq_true in X_tensor:
                seq_true = seq_true.to(self.device)
                seq_pred = self.model(seq_true)
                seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                loss = self.criterion(seq_pred, seq_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            if X_test is not None:
                self.model = self.model.eval()
                with torch.no_grad():
                    for seq_true in X_test_tensor:

                        seq_true = seq_true.to(self.device)
                        seq_pred = self.model(seq_true)
                        seq_pred = seq_pred.reshape((seq_pred.shape[0],))
                        loss = self.criterion(seq_pred, seq_true)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                self.history['val'].append(val_loss)
                print(f'Epoch {epoch}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}')
            
            train_loss = np.mean(train_losses)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
            self.history['train'].append(train_loss)
            
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
        
        return self.history, best_model_wts
        
    
    def _calculate_threshold(self, X_scaled):
        """
        Calculate the anomaly threshold.
        
        Args:

            X_scaled (np.ndarray): Scaled input data
            
        Returns:

            float: Anomaly threshold
        """
        self.model.eval()
        if not isinstance(X_scaled, torch.Tensor):
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        else:
            X_tensor = X_scaled.to(self.device)
            
        with torch.no_grad():
            X_pred = self.model(X_tensor).cpu().detach().numpy()
        
        X_numpy = X_scaled.cpu().numpy() if isinstance(X_scaled, torch.Tensor) else X_scaled
        mse = np.mean(np.square(X_numpy - X_pred), axis=1)
        self.threshold = np.mean(mse) + self.threshold_multiplier * np.std(mse)
        
        return self.threshold
        
    def predict(self, X):
        """
        Predict anomalies in the input data.
        
        Args:

            X (np.ndarray): Input data for anomaly detection
            
        Returns:

            np.ndarray: Binary array where 1 indicates anomaly, 0 indicates normal
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            X_pred = self.model(X_tensor).cpu().detach().numpy()
        
        mse = np.mean(np.square(X - X_pred), axis=1)
        
        if self.threshold is None:
            self._calculate_threshold(X)
        
        anomalies = (mse > self.threshold).astype(int)
        
        return anomalies
    
    def anomaly_score(self, X):
        """
        Calculate anomaly scores for the input data.
        
        Args:

            X (np.ndarray): Input data for anomaly detection
        
        Returns:

            np.ndarray: Anomaly scores
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            X_pred = self.model(X_tensor).cpu().detach().numpy()
        
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)
        
        return mse
    
    def feature_importance(self, X):
        """
        Calculate feature importance based on reconstruction error contribution.
        
        Args:
            X (np.ndarray): Input data for feature importance analysis
            
        Returns:
            pd.DataFrame: Feature importance scores for each input feature
        """
        if len(X.shape) == 2:
            sequences = []
            for i in range(X.shape[0]):
                sequence = X[i].reshape(self.seq_len, self.n_features)
                sequences.append(sequence)
        else:
            sequences = X
        
        self.model.eval()
        
        all_errors = []
        
        for sequence in sequences:
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            with torch.no_grad():
                pred = self.model(sequence_tensor).cpu().detach().numpy()
            
            squared_error = np.square(sequence - pred)
            all_errors.append(squared_error)
        
        avg_squared_errors = np.mean(all_errors, axis=0)
        
        feature_mse = np.mean(avg_squared_errors, axis=0)
        
        total_mse = np.sum(feature_mse)
        if total_mse > 0:
            importance_scores = feature_mse / total_mse
        else:
            importance_scores = np.ones_like(feature_mse) / len(feature_mse)
        
        feature_names = [f"feature_{i}" for i in range(self.n_features)]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def plot_feature_importance(self, X, feature_names=None, top_n=None):
        """
        Plot feature importance scores.
        
        Args:
            X (np.ndarray): Input data for feature importance analysis
            feature_names (list, optional): List of feature names
            top_n (int, optional): Number of top features to display
            
        Returns:
            matplotlib.figure.Figure: The generated plot figure
        """        
        importance_df = self.feature_importance(X)
        
        if feature_names is not None:
            if len(feature_names) == self.n_features:
                importance_df['feature'] = feature_names
            else:
                print(f"Warning: {len(feature_names)} feature names provided but model has {self.n_features} features.")
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance Based on Reconstruction Error')
        plt.tight_layout()
        
        return fig
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:

            path (str): File path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        
    def load_model(self, path):
        """
        Load the model from disk.
        
        Args:
        
            path (str): File path to load the model
        """
        self.model.load_state_dict(torch.load(path))