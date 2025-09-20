import torch
import torch.nn as nn


# Hierarchy/Flow for VisionEncoder:
# Input image → PatchEmbedding → PositionalEmbedding → [Stack of TransformerEncoder] → LayerNorm → Mean Pooling → Linear Head (classification)


####### Adds fixed sinusoidal positional information to input embeddings (sequence or patch embeddings).
class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, emb_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term) # even-index columns
        pe[:, 1::2] = torch.cos(position * div_term) # odd-index columns
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # Adjust to match the input size
        
        return x


####### Implements a single attention head for the transformer, including query, key, value projections and attention computation.
class AttentionHead(nn.Module):
  def __init__(self, emb_dim, head_size, causal=False):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(emb_dim, head_size, bias=False)
    self.key = nn.Linear(emb_dim, head_size, bias=False)
    self.value = nn.Linear(emb_dim, head_size, bias=False)

    self.causal = causal

  def forward(self, x, kv=None, k_mask=None):
    B, T, C = x.shape
    Q = self.query(x)                            # Shape: (B, q_len, head_size)
    K = self.key(kv if kv is not None else x)    # Shape: (B, seq_len, head_size) where seq_len = k_len if cross attention, else q_len
    V = self.value(kv if kv is not None else x)  # Shape: (B, seq_len, head_size) where seq_len = k_len if cross attention, else q_len

    # Dot Product of Queries and Keys
    attention = Q @ K.transpose(-2,-1) # Shape: (B, q_len, k_len)

    # Scaling to prevent softmax saturation
    attention = attention / (self.head_size ** 0.5)
    
    # Applying k padding mask if provided (column-wise mask)
    if k_mask is not None:
      k_mask = k_mask.unsqueeze(1) # Shape: (B, 1, k_len) can be broadcast with attention
      attention = attention.masked_fill(k_mask == 0, float("-inf"))

    # Applying causal mask for decoder's masked MHA
    if self.causal:
      c_mask = torch.tril(torch.ones(T, T, device=x.device)) # is broadcastable with attention (B, T, T)
      attention = attention.masked_fill(c_mask == 0, float("-inf"))

    attention = torch.softmax(attention, dim=-1)

    # Weighted sum of values
    output = attention @ V # Shape: (B, seq_len, head_size)  where seq_len = k_len if cross attention, else q_len

    return output
  

####### Combines multiple AttentionHead instances to form multi-head attention, then projects the concatenated output.
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, causal=False):
        super().__init__()
        
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        
        self.head_size = emb_dim // n_heads

        self.W_o = nn.Linear(emb_dim, emb_dim, bias=False)

        self.causal = causal

        self.heads = nn.ModuleList([AttentionHead(emb_dim, self.head_size, causal=self.causal) for _ in range(n_heads)])

    def forward(self, x, kv=None, k_mask=None):
        # Combine attention heads
        out = torch.cat([head(x, kv, k_mask=k_mask) for head in self.heads], dim=-1)
        
        out = self.W_o(out)

        return out
    


####### Implements a transformer encoder block with layer normalization, multi-head attention, and a feed-forward MLP, plus residual connections.
class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, n_heads, r_mlp=4, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(emb_dim)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(emb_dim, n_heads)

        # Dropout after MHA
        self.dropout1 = nn.Dropout(dropout)  

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(emb_dim)

        # Multilayer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * r_mlp),
            nn.GELU(),
            nn.Linear(self.emb_dim * r_mlp, self.emb_dim)
        )

        # Dropout after MLP
        self.dropout2 = nn.Dropout(dropout)  

    def forward(self, x, src_mask=None):
        # Residual Connection After Sub-Layer 1 (MHA)
        x = x + self.dropout1(self.mha(self.ln1(x), k_mask=src_mask))

        # Residual Connection After Sub-Layer 2 (MLP)
        x = x + self.dropout2(self.mlp(self.ln2(x)))

        return x


####### Splits an image into non-overlapping patches and projects each patch to an embedding vector using a convolution.
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, emb_dim):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.emb_dim = emb_dim # width
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, emb_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, emb_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, emb_dim)
        return x


####### The encoder adapted for object detection with an encoder-decoder architecture. CLS token removed.
class VisionEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, emb_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, emb_dim)
        self.n_patches = self.patch_embed.n_patches
        self.positional_embedding = PositionalEmbedding(emb_dim, self.n_patches)  # No CLS token
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoder(emb_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, n_patches, emb_dim)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        x = self.norm(x)
        return x  # (B, n_patches, emb_dim)

# TransformerDecoderLayer: cross-attention + self-attention + MLP
class TransformerDecoderLayer(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_dim, n_heads, causal=True) # Self attention is causal for sequence prediction
        self.cross_attn = MultiHeadAttention(emb_dim, n_heads, causal=False)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln3 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-attention
        tgt2 = self.self_attn(self.ln1(tgt))
        tgt = tgt + self.dropout(tgt2)
        # Cross-attention
        tgt2 = self.cross_attn(self.ln2(tgt), kv=memory)
        tgt = tgt + self.dropout(tgt2)
        # MLP
        tgt2 = self.mlp(self.ln3(tgt))
        tgt = tgt + self.dropout(tgt2)
        return tgt
    
# TransformerDecoder 
class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim, n_heads, n_layers, n_queries, dropout=0.1):
        super().__init__()
        self.n_queries = n_queries
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, emb_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(emb_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, memory):
        # memory: (B, seq_len, emb_dim) from encoder
        B = memory.size(0)
        query_embed = self.query_embed.expand(B, -1, -1)  # (B, n_queries, emb_dim)
        
        # initialize target sequence with the learnable query embeddings
        tgt = query_embed

        for layer in self.layers:
            tgt = layer(tgt, memory)
        tgt = self.norm(tgt)
        return tgt  # (B, n_queries, emb_dim)


# Action generation head: predicts a sequence of (x,y) coordinates
class ActionGenerationHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Predicts a sequence of 2D coordinates
        self.action_head = nn.Linear(emb_dim, 2)  # for (x, y)

    def forward(self, x):
        # x: (B, n_queries, emb_dim)
        # n_queries is the length of the action sequence
        actions = self.action_head(x).tanh()  # (B, n_queries, 2), normalized to [-1, 1] with tanh
        return actions


# Full encoder-decoder model for action generation
class VisionActionModel(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, emb_dim, n_heads, n_enc_layers, n_dec_layers, n_queries, dropout=0.1):
        super().__init__()
        self.encoder = VisionEncoder(img_size, patch_size, in_channels, emb_dim, n_heads, n_enc_layers, dropout=dropout)
        self.decoder = TransformerDecoder(emb_dim, n_heads, n_dec_layers, n_queries, dropout=dropout)
        self.action_head = ActionGenerationHead(emb_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        memory = self.encoder(x)  # (B, seq_len, emb_dim)
        hs = self.decoder(memory)  # (B, n_queries, emb_dim)
        actions = self.action_head(hs) # (B, n_queries, 2)
        return actions
