import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class Tier1LightGBM:
    """
    Tier 1 Fast Screening Model.
    Runs fast on CPU, handles tabular features well.
    Implements Quantile Regression bounds to provide prediction uncertainty.
    """
    def __init__(self, lower_alpha=0.1, upper_alpha=0.9):
        # We simulate uncertainty by training three models
        
        params_base = {
            'objective': 'quantile',
            'metric': 'quantile',
            'learning_rate': 0.05,
            'max_depth': 4,
            'verbose': -1
        }
        
        self.model_lower = lgb.LGBMRegressor(**params_base, alpha=lower_alpha)
        self.model_upper = lgb.LGBMRegressor(**params_base, alpha=upper_alpha)
        
        # Primary classifier for point estimates
        self.model_median = lgb.LGBMClassifier(learning_rate=0.05, max_depth=4, verbose=-1, is_unbalance=True)
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        print("Training Tier 1 LightGBM models...")
        self.model_lower.fit(X, y)
        self.model_upper.fit(X, y)
        self.model_median.fit(X, y)
        
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """Returns standard probability plus the bound spread as uncertainty metric."""
        lower_bound = self.model_lower.predict(X)
        upper_bound = self.model_upper.predict(X)
        prob = self.model_median.predict_proba(X)[:, 1]
        
        # The spread between P90 and P10 gives us an uncertainty proxy
        uncertainty = np.abs(upper_bound - lower_bound)
        return prob, uncertainty

class TransformerDeepModel(nn.Module):
    """
    Tier 2 Deep Model Stub (Transformer for time-series features).
    Implements Monte Carlo Dropout for Bayesian uncertainty estimation.
    """
    def __init__(self, input_dim: int, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # We explicitly keep this dropout modules active during inference for MC Dropout
        self.mc_dropout = nn.Dropout(p=dropout) 
        self.fc_out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Expected x shape: (batch_size, seq_len, input_dim)
        emb = self.embedding(x)
        out = self.transformer(emb)
        
        # Pooling over time step (take the last timestep representations)
        last_out = out[:, -1, :]
        
        # Drop out happens here, EVEN IN eval() mode if driven explicitly
        dropped_out = self.mc_dropout(last_out)
        
        logits = self.fc_out(dropped_out)
        return self.sigmoid(logits)

def predict_mc_dropout(model: TransformerDeepModel, x: torch.Tensor, num_passes: int = 15) -> tuple:
    """
    Runs inference multiple times with dropout enabled to build a distribution.
    Returns the mean prediction and the standard deviation (uncertainty).
    """
    model.train() # Force dropout to remain active
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            preds = model(x)
            predictions.append(preds.cpu().numpy())
            
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0).flatten()
    std_pred = predictions.std(axis=0).flatten()
    
    return mean_pred, std_pred

def ensemble_inference(tier1_prob: np.ndarray, tier2_prob: np.ndarray, weight_t1: float = 0.6) -> np.ndarray:
    """
    Simple weighted ensemble for blending the two models. 
    Can be dynamically weighted based on uncertainty.
    """
    return (tier1_prob * weight_t1) + (tier2_prob * (1.0 - weight_t1))
