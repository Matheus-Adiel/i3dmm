import torch
import torch.optim as optim
from model import I3DMM
from loss import total_loss
from utils import fourier_features

batch_size = 4
epochs = 2000  
learning_rate = 0.0005
loss_weights = {'Wg': 1.0, 'Wc': 1.0, 'Ws': 1.0, 'Wr': 0.01, 'Wlm': 1.0}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Rodando em: {device}")

i3dmm_model = I3DMM().to(device)
optimizer = optim.Adam(i3dmm_model.parameters(), lr=learning_rate)

print("Gerando dados sintéticos fixos...")

x_fixed = torch.randn(batch_size, 3).to(device)

id_idx_fixed = torch.randint(0, 58, (batch_size,)).to(device)
ex_idx_fixed = torch.randint(0, 10, (batch_size,)).to(device)
geo_h_fixed = torch.randint(0, 4, (batch_size,)).to(device)
col_h_fixed = torch.randint(0, 3, (batch_size,)).to(device)

sdf_gt_fixed = torch.randn(batch_size, 1).to(device) # Distância real
col_gt_fixed = torch.rand(batch_size, 3).to(device)  # Cor real
landmarks_gt_fixed = torch.randn(batch_size, 68, 3).to(device) # 68 landmarks reais

print("Iniciando treinamento...")

for epoch in range(epochs):
    optimizer.zero_grad()

    sdf_pred, col_pred, delta, z_geo, z_col = i3dmm_model(
        x_fixed, id_idx_fixed, ex_idx_fixed, geo_h_fixed, col_h_fixed
    )

    lm_flat = landmarks_gt_fixed.view(-1, 3) 
    
    lm_encoded = fourier_features(lm_flat, 6)
    
    z_geo_expanded = z_geo.unsqueeze(1).expand(-1, 68, -1).reshape(-1, 384)
    
    delta_lm = i3dmm_model.deformNet(lm_encoded, z_geo_expanded)
    
    landmarks_pred_ref = lm_flat + delta_lm
    
    landmarks_pred_ref = landmarks_pred_ref.view(batch_size, 68, 3)

    loss, l_geo, l_def, l_col, l_reg, l_lm = total_loss(
        loss_weights, 
        z_geo, z_col, 
        delta, 
        landmarks_pred_ref, 
        col_pred, col_gt_fixed, 
        sdf_pred, sdf_gt_fixed, 
        clamp_value=0.1
    )
    
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Total Loss: {loss.item():.6f} | Geo: {l_geo.item():.4f} | LM: {l_lm.item():.4f}")