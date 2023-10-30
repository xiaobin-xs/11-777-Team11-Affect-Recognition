import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.5):
        super(SimpleMLP, self).__init__()
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = torch.relu(x)
        
        # Hidden layer with dropout
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return x


class ComplexMLP(nn.Module):
    def __init__(self, eeg_embed_dim, face_embed_dim, output_dim, dropout_prob=0.5):
        super(ComplexMLP, self).__init__()
        self.eeg_embed_dim = eeg_embed_dim
        self.face_embed_dim = face_embed_dim
        self.combine_embed_dim = eeg_embed_dim + face_embed_dim//8
        ### transform facial embedding into lower dimension
        self.MLP1 = SimpleMLP(face_embed_dim, face_embed_dim//2, face_embed_dim//8)
        ### transform EEG embedding to facial embedding
        self.MLP2 = SimpleMLP(eeg_embed_dim, eeg_embed_dim//2, face_embed_dim//8)
        ### predictor
        self.predictor = SimpleMLP(self.combine_embed_dim, self.combine_embed_dim//2, 2)

        
    def forward(self, x):
        eeg_embed  = x[:, :self.eeg_embed_dim]
        face_embed = x[:, self.eeg_embed_dim:]
        face_embed_low = self.MLP1(face_embed)
        face_embed_pred = self.MLP2(eeg_embed)
        combine_embed = torch.concat([eeg_embed, face_embed_pred], axis=1)
        out = self.predictor(combine_embed)
        return out, face_embed_pred, face_embed_low