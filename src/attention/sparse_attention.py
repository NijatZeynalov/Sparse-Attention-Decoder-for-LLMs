import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from einops import rearrange, reduce
from ..config.config import AttentionConfig
from ..utils.logger import get_logger
from .position_persistence import PositionPersistenceManager
from .token_selector import TopKTokenSelector

logger = get_logger(__name__)


class SparseAttention(nn.Module):
    """
    Sparse Attention implementation for efficient long-sequence processing.
    """

    def __init__(
            self,
            config: AttentionConfig,
            hidden_size: int,
            num_attention_heads: int,
            max_position_embeddings: int,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        # Initialize components
        self.token_selector = TopKTokenSelector(config)
        self.position_persistence = PositionPersistenceManager(config)

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Initialize cache for position persistence
        self.persistent_tokens = None
        self.layer_index = 0

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of sparse attention.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_length]
            position_ids: Position IDs of shape [batch_size, seq_length]
            past_key_value: Cached key and value tensors
            output_attentions: Whether to return attention weights
            use_cache: Whether to use KV cache

        Returns:
            tuple: (output, attention_weights, past_key_value)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = rearrange(query_states, 'b n (h d) -> b h n d', h=self.num_attention_heads)
        key_states = rearrange(key_states, 'b n (h d) -> b h n d', h=self.num_attention_heads)
        value_states = rearrange(value_states, 'b n (h d) -> b h n d', h=self.num_attention_heads)

        # Handle KV cache if provided
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        kv_seq_length = key_states.shape[-2]

        # Apply sparse attention selection
        if kv_seq_length > self.config.attention_window:
            # Get persistent tokens from previous layer if available
            persistent_indices = self.position_persistence.get_persistent_indices(
                self.layer_index,
                batch_size,
                kv_seq_length,
                device=hidden_states.device
            )

            # Select top-k tokens based on attention scores
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
            attention_scores = attention_scores / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32)
            )

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # Combine persistent and newly selected tokens
            selected_indices = self.token_selector(
                attention_scores,
                persistent_indices,
                self.config.top_k
            )

            # Update persistent tokens for next layer
            self.position_persistence.update_persistent_indices(
                self.layer_index,
                selected_indices,
                attention_scores
            )

            # Apply sparse attention
            key_states = torch.gather(
                key_states,
                2,
                selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            )
            value_states = torch.gather(
                value_states,
                2,
                selected_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            )

            if attention_mask is not None:
                attention_mask = torch.gather(
                    attention_mask,
                    2,
                    selected_indices.unsqueeze(1).expand(-1, attention_mask.size(1), -1)
                )

        # Compute attention weights
        attention_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_weights = attention_weights / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, value_states)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')

        # Project output
        output = self.o_proj(attention_output)

        # Prepare cache for next layer
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # Update layer index
        self.layer_index = (self.layer_index + 1) % self.max_position_embeddings

        if output_attentions:
            return output, attention_weights, past_key_value
        return output, None, past_key_value

    def _reset_persistence(self):
        """Reset position persistence state."""
        self.persistent_tokens = None
        self.layer_index = 0
        self.position_persistence.reset()