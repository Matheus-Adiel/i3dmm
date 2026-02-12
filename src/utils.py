import torch

def fourier_features(x, num_freqs=6):
    """
    fourier features
    
    :param x: tensor (batch_size, 3), points of 3d model
    :param freqs: int, frequence for lengh of fourier features 
    :return: tensor (batch_size, 3 + 3*num_freqs*2)
    """

    powers_of_two = 2.0 ** torch.arange(num_freqs, device=x.device)
    base = powers_of_two * torch.pi * x.unsqueeze(-1)
    sin_base = torch.sin(base)
    cos_base = torch.cos(base)
    concat = torch.cat([x.unsqueeze(-1), sin_base, cos_base], dim=-1)
    batch_size = x.shape[0]
    return concat.reshape(batch_size, -1) 