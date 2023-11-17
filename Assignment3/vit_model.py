import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbeddings(nn.Module):
    """TODO: (0.5 out of 8) Calculates patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """
    def __init__(
        self, 
        image_size: int,
        patch_size: int,
        hidden_size: int,
        num_channels: int = 3,      # 3 for RGB, 1 for Grayscale
        ):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        '''
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        '''
        #num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        num_patches = (image_size // patch_size) ** 2
        # #########################

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

    def forward(
        self, 
        x: torch.Tensor,
        ) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # #########################
        # Finish Your Code HERE
        # #########################

        # Calculate Patch Embeddings, then flatten into
        # batched 1D sequence (batch_size, seq_length, hidden_size)

        embeddings = self.projection(x).flatten(2).transpose(1, 2)
        return embeddings

        # #########################
        

class PositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches,
        hidden_size,
        ):
        """TODO: (0.5 out of 8) Given patch embeddings, 
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################
        
        # Specify [CLS] and positional embedding as learnable parameters
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.position_embeddings =  nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        #self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        # #########################
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        embeddings: torch.Tensor
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################

        # Concatenate [CLS] token with embedded patch tokens
        
        # Then add positional encoding to each token

        batch_size, seq_length, _ = embeddings.shape
       # embeddings = self.patch_embeddings(embeddings, interpolate_pos_encoding=interpolate_pos_encoding)

        
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
       
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

      


        # #########################
        return embeddings

class GELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """TODO: (0.25 out of 8) Residual Attention Block.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # #########################
        # Finish Your Code HERE
        # #########################

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=0.1   
        )    # Refer to nn.MultiheadAttention
        mlp_hidden_dim = 4 * d_model  # This can be adjusted
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),   # Activation function used in GPT and ViT
            nn.Linear(mlp_hidden_dim, d_model)
        )
        #self.ln_1 = None
        #self.mlp = None     # A trick is to use nn.Sequential to specify multiple layers at once
        #self.ln_2 = None
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        # #########################

    def forward(self, x: torch.Tensor):

        # #########################
        # Finish Your Code HERE
        # #########################

        # LayerNorm -> Multi-head Attention
        # Residual connection against x

        # LayerNorm -> MLP Block
        # Residual connection against x

        # #########################
        #attn_output, _ = self.attn(x, x, x)
        #x = x + attn_output
        #x = self.ln_1(x)

        # MLP part
        #x = x + self.mlp(x)
        #x = self.ln_2(x)
        # LayerNorm -> Multi-head Attention
        attn_output, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        # Residual connection against x
        x = x + attn_output

        # LayerNorm -> MLP Block
        mlp_output = self.mlp(self.ln_2(x))
        # Residual connection against x
        x = x + mlp_output


        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2)  # (batch_size, seqlen, dim) -> (seqlen, batch_size, dim)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # (seqlen, batch_size, dim) -> (batch_size, seqlen, dim)
        return x

class ViT(nn.Module):
    """TODO: (0.5 out of 8) Vision Transformer.
    """
    def __init__(
        self, 
        image_size: int, 
        patch_size: int, 
        num_channels: int,
        hidden_size: int, 
        layers: int, 
        heads: int,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.patch_embed = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_channels=num_channels
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionEmbedding(
            num_patches=num_patches,
            hidden_size=hidden_size
        )
        self.ln_pre = nn.LayerNorm(hidden_size)
        self.transformer = Transformer(
            width=hidden_size,
            layers=layers,
            heads=heads
        )
       

        

        self.ln_post = nn.LayerNorm(hidden_size)

        # #########################


    def forward(self, x: torch.Tensor):
        # #########################
        # Finish Your Code HERE
        # #########################

        # Convert images to patch embeddings
        x = self.patch_embed(x)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Apply Layer Normalization before Transformer
        x = self.ln_pre(x)

        # Pass through the transformer
        x = self.transformer(x)

        # Apply Layer Normalization after Transformer
        out = self.ln_post(x)
        # #########################

        return out[:, 0]

class ClassificationHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_classes: int = 10,
        ):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        out = self.classifier(feats)
        return out

class LinearEmbeddingHead(nn.Module):
    """TODO: (0.25 out of 8) Given features from ViT, generate linear embedding vectors.
    """
    def __init__(
        self, 
        hidden_size: int,
        embed_size: int = 64,
        ):
        super().__init__()
        self.embed_size = embed_size
        # #########################
        # Finish Your Code HERE
        # #########################
        self.projection = nn.Linear(hidden_size, embed_size)
        # #########################

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        # #########################
        # Finish Your Code HERE
        # #########################
        out = self.projection(feats)
        # #########################
        return out