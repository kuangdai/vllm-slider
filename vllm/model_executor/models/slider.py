import torch
import torch.nn as nn


class SliderModel(nn.Module):
    def __init__(self, n_variables, n_hidden, n_heads_sharing_slider, dropout,
                 n_base_heads, n_token_dim):
        """
        A model that encodes slider variables into attention key-value pairs.

        Args:
            n_variables (int): Number of slider variables.
            n_hidden (int): Hidden layer size in the prefix encoder.
            n_heads_sharing_slider (int): Number of base heads sharing one slider head.
            dropout (float): Dropout rate in the prefix encoder.
            n_base_heads (int): Total number of attention heads in the transformer.
            n_token_dim (int): Embedding dimension per token.
        """
        super().__init__()

        # Store model parameters
        self.n_variables = n_variables
        self.n_hidden = n_hidden
        self.n_heads_sharing_slider = n_heads_sharing_slider
        self.n_base_heads = n_base_heads
        self.n_token_dim = n_token_dim

        # Ensure `n_base_heads` is evenly divisible by `n_heads_sharing_slider`
        assert self.n_base_heads % self.n_heads_sharing_slider == 0, \
            "n_base_heads must be divisible by n_heads_sharing_slider."

        # Compute number of slider-specific attention heads
        self.n_slider_heads = self.n_base_heads // self.n_heads_sharing_slider

        # Define the output size for combined key and value
        self.kv_size = 2 * self.n_token_dim * self.n_slider_heads  # Merged KV size

        # Independent weight matrices for each variable (each maps R^1 -> R^kv_size)
        self.encode_linear = nn.Linear(1, n_variables * self.kv_size)

        # Hidden layer transformation per variable
        self.upscale_linear = nn.Linear(self.kv_size, n_variables * n_hidden)

        # Final output transformation per variable
        self.downscale_linear = nn.Linear(n_hidden, n_variables * self.kv_size)

        # Define attention factor
        self.attention_factor = nn.Linear(1, 1, bias=False)
        self.attention_factor.weight.data.zero_()
        self.attention_factor.SLIDER_DO_NOT_REINITIALIZE = True

        # Activation and dropout layers
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, prefix: torch.Tensor, cast_dtype_device=False):
        """
        Forward pass for generating key-value pairs from slider variables.

        Args:
            prefix (Tensor): Input slider values of shape [batch_size, n_variables].
            cast_dtype_device (bool): Cast dtype and device of prefix.

        Returns:
            Tensor: Key-value pairs of shape [2, batch_size, n_base_heads, seq_len, n_token_dim].
        """

        # Move input to the same device and dtype as the model parameters
        if cast_dtype_device:
            device = self.encode_linear.weight.device
            dtype = self.encode_linear.weight.dtype
            prefix = prefix.to(device=device, dtype=dtype)

        # Reshape input to [batch_size, n_variables, 1] for matrix multiplication
        prefix = prefix.unsqueeze(-1)  # Shape: [batch_size, n_variables, 1]

        # Independent linear transformation for each variable
        encode_w = self.encode_linear.weight.view(self.n_variables, self.kv_size, 1)
        encode_b = self.encode_linear.bias.view(1, self.n_variables, self.kv_size)
        slider_kv = torch.einsum("VKI,BVI->BVK", encode_w, prefix) + encode_b
        slider_kv = self.tanh(slider_kv)
        slider_kv = self.dropout(slider_kv)

        # Hidden layer transformation
        upscale_w = self.upscale_linear.weight.view(self.n_variables, self.n_hidden, self.kv_size)
        upscale_b = self.upscale_linear.bias.view(1, self.n_variables, self.n_hidden)
        slider_kv = torch.einsum("VHK,BVK->BVH", upscale_w, slider_kv) + upscale_b
        slider_kv = self.tanh(slider_kv)
        slider_kv = self.dropout(slider_kv)

        # Final transformation
        downscale_w = self.downscale_linear.weight.view(self.n_variables, self.kv_size, self.n_hidden)
        downscale_b = self.downscale_linear.bias.view(1, self.n_variables, self.kv_size)
        slider_kv = torch.einsum("VKH,BVH->BVK", downscale_w, slider_kv) + downscale_b

        # Reshape to separate keys and values
        # Shape: [batch_size, n_variables, 2, n_slider_heads, n_token_dim]
        slider_kv = slider_kv.view(prefix.shape[0], self.n_variables, 2, self.n_slider_heads, self.n_token_dim)

        # Expand `n_slider_heads` across `n_base_heads`
        b, n, _, h, z = slider_kv.shape
        slider_kv = slider_kv.unsqueeze(3)  # [B, N, 2, 1, H, Z]
        slider_kv = slider_kv.expand(b, n, 2, self.n_heads_sharing_slider, h, z)
        slider_kv = slider_kv.reshape(b, n, 2, -1, z)

        # Permute for attention format: [batch_size, n_base_heads, seq_len, n_token_dim, 2]
        slider_kv = slider_kv.permute(0, 3, 1, 4, 2)  # Move slider head dim before sequence length

        # Split into keys and values along the last dimension
        slider_keys, slider_values = slider_kv[..., 0], slider_kv[..., 1]

        return slider_keys, slider_values, self.attention_factor.weight[0, 0]
