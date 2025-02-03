import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class EvidentialTransformerDenoiseAutoEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout_rate, output_dim=None):
        super(EvidentialTransformerDenoiseAutoEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        if output_dim:
            self.output_dim = output_dim

        # Positional Encoding
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(self.max_seq_length, d_model), requires_grad=False)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Fully connected layers for encoding inputs and decoding outputs
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, output_dim or input_dim * 4)   # Output 4x dimensions in order to separate mu, v, alpha, beta

        # Apply weight initialization
        self.apply(initialize_weights)

    def forward(self, src, padding_mask=None, return_latent=False, noise_factor=0.05):
        # add noise
        noise = torch.randn_like(src) * noise_factor
        noisy_src = src + noise
        noisy_src = self.input_fc(noisy_src)  # Shape: (batch_size, seq_length, d_model)

        # add positional embedding
        noisy_src += self.positional_encoding[:, :noisy_src.size(1), :]
            
        if padding_mask is not None:
            # padding_mask：(batch_size, seq_len, input_dim)
            padding_mask_timestep = padding_mask.any(dim=-1)  # [batch_size, seq_len]
            mask_expanded = padding_mask_timestep.unsqueeze(-1).expand_as(noisy_src).bool()
            # mask
            noisy_src = torch.where(mask_expanded, noisy_src, torch.tensor(0.0, device=noisy_src.device))
        # encode
        encoded_memory = self.transformer_encoder(noisy_src)  
        
        # 准备解码器的目标输入序列
        tgt = self.input_fc(src)  # 使用原始的干净输入序列
        tgt += self.positional_encoding[:, :tgt.size(1), :]
        
        if padding_mask is not None:
            mask_expanded_tgt = padding_mask_timestep.unsqueeze(-1).expand_as(tgt).bool()
            tgt = torch.where(mask_expanded_tgt, tgt, torch.tensor(0.0, device=tgt.device))

        # decode
        decoded_output = self.transformer_decoder(tgt, encoded_memory)
        decoded_output = self.output_fc(decoded_output)  # Shape: (batch_size, seq_length, output_dim * 4)
    
        if padding_mask is not None:
            mask_expanded = padding_mask_timestep.unsqueeze(-1).expand(-1, -1, decoded_output.size(-1)).float()
            #print(f"decoded_output shape: {decoded_output.shape}")
            #print(f"mask_expanded shape: {mask_expanded.shape}")
            decoded_output = decoded_output * mask_expanded

        # Output mu, v, alpha, beta via Evidential Learning
        mu, logv, logalpha, logbeta = torch.chunk(decoded_output, 4, dim=2)
        v = F.softplus(logv) + 1e-6
        alpha = F.softplus(logalpha) + 1.0
        beta = F.softplus(logbeta) + 1e-6

        # Return the encoded representation and decoded uncertainty outputs
        if return_latent:
            return mu, v, alpha, beta, encoded_memory
        else:
            return mu, v, alpha, beta
       


    def _generate_positional_encoding(self, length, d_model):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # add batch dimension




# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
# def nig_reg(gamma, v, alpha, _beta, y):
#     #reg = (y - gamma).abs() * (2 * v + alpha)
#     reg = torch.pow(y - gamma, 2) * (2 * v + alpha)
#     return reg.mean()
    

# def nig_nll(gamma, v, alpha, beta, y):

#     v = torch.clamp(v, min=1e-3, max=10.0)
#     alpha = torch.clamp(alpha, min=1, max=10.0)
#     beta = torch.clamp(beta, min=1e-3, max=10.0)

#     two_beta = 2 * beta
#     t1 = 0.5 * torch.log(torch.pi / v)
#     t2 = alpha * torch.log(two_beta)
#     t3 = (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_beta)
#     t4 = torch.lgamma(alpha)
#     t5 = torch.lgamma(alpha + 0.5)

#     nll = t1 - t2 + t3 + t4 - t5

#     return nll.mean()

# def evidential_regression(dist_params, y, lamb=1.0, offset=2.0):
#     return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y) + offset


def nig_reg(gamma, v, alpha, _beta, y, mask=None):
    reg = torch.pow(y - gamma, 2) * (2 * v + alpha)
    
    if mask is not None:
        mask_expanded = mask.float()
        reg = reg * mask_expanded  # mask
        valid_count = mask_expanded.sum()
        reg_loss = reg.sum() / valid_count
    else:
        reg_loss = reg.mean()

    return reg_loss

def nig_nll(gamma, v, alpha, beta, y, mask=None):
    # v = torch.clamp(v, min=1e-4, max=20.0)
    # alpha = torch.clamp(alpha, min=1.0 + 1e-6, max=20.0)
    # beta = torch.clamp(beta, min=1e-4, max=20.0)
    
    # v = torch.clamp(v, min=1e-3)
    # alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    # beta = torch.clamp(beta, min=1e-3)
    
    v = torch.clamp(v, min=1e-3, max=10.0)
    alpha = torch.clamp(alpha, min=1.0 + 1e-6, max=10.0)
    beta = torch.clamp(beta, min=1e-3, max=10.0)

    two_beta = 2 * beta
    t1 = 0.5 * torch.log(torch.pi / v)
    t2 = alpha * torch.log(two_beta)
    t3 = (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_beta)
    t4 = torch.lgamma(alpha)
    t5 = torch.lgamma(alpha + 0.5)

    nll = t1 - t2 + t3 + t4 - t5

    if mask is not None:
        mask_expanded = mask.float()
        nll = nll * mask_expanded  # mask
        valid_count = mask_expanded.sum()
        nll_loss = nll.sum() / valid_count
    else:
        nll_loss = nll.mean()

    return nll_loss

def evidential_regression(dist_params, y, lamb=1.0, offset=2.0, mask=None, recon_error=None, recon_weight=0.1):
    nll_loss = nig_nll(*dist_params, y, mask)
    reg_loss = lamb * nig_reg(*dist_params, y, mask)
    
    # Optionally add reconstruction error penalty for OOD sensitivity
    if recon_error is not None:
        recon_penalty = recon_weight * recon_error.mean()
    else:
        recon_penalty = 0.0
        
    # Total loss
    total_loss = nll_loss + reg_loss + recon_penalty + offset
    return total_loss

