
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,
                 num_layers, num_classes, dropout,
                 padding_idx=0, pretrained_embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded           = self.dropout(self.embedding(x))
        output, (hidden,_) = self.lstm(embedded)
        hidden_fwd         = hidden[-2]
        hidden_bwd         = hidden[-1]
        combined           = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        return self.classifier(self.dropout(combined))
