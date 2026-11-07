"""BiLSTM Model with Masked Mean Pooling for EE6405 PromptShield (E2).

Architecture specifications from syllabus / project plan:
- 100d embedding (trainable for both random and pretrained initialization)
- Bidirectional LSTM with 128 hidden units per direction (total 256 output dimensions)
- Masked mean pooling across non-padded tokens
- Dropout rate 0.3
- Linear classification head mapping 256 -> 1 (raw logits for sigmoid / BCE)
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        dropout: float = 0.3,
        pretrained_weight: torch.Tensor = None,
        trainable_embedding: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
        else:
            # Initialize random embeddings from normal distribution, keep padding vector zero
            nn.init.normal_(self.embedding.weight.data, mean=0.0, std=0.05)
            self.embedding.weight.data[padding_idx].fill_(0.0)

        self.embedding.weight.requires_grad = trainable_embedding

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(p=dropout)
        # BiLSTM concatenates forward (128) + backward (128) -> 256 dims
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            lengths: Tensor of shape (batch_size,) containing true sequence lengths

        Returns:
            logits: Tensor of shape (batch_size,)
        """
        # (batch_size, seq_len, embedding_dim)
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)

        # Pack sequence for LSTM efficiency and accurate padding handling
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeds,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )  # (batch_size, seq_len, hidden_size * 2)

        # Masked Mean Pooling: exclude pad tokens (padding_idx = 0)
        # mask shape: (batch_size, seq_len, 1)
        mask = (input_ids != 0).unsqueeze(-1).float()
        sum_masked = torch.sum(lstm_out * mask, dim=1)  # (batch_size, hidden_size * 2)
        actual_lens = mask.sum(dim=1).clamp(min=1.0)    # (batch_size, 1)
        pooled = sum_masked / actual_lens

        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(-1)  # (batch_size,)
        return logits
