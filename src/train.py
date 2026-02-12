import torch
import torch.optim as optim
from model import I3DMM, DeformNet
from loss import total_loss
from utils import fourier_features

batch_size = 4
# Article: "we use the Adam solver... We train for 1000 epochs with a learning rate of 0.0005" [Sec. 4.]
epochs = 1000
learning_rate = 0.0005
loss_weights = {'Wg': 1, 'Wc': 1, 'Ws': 1, 'Wr': 1, 'Wlm': 1}

i3dmm_model = I3DMM()
optimizer = optim.Adam(i3dmm_model.parameters(), lr=learning_rate)

# random numbers for test training
x = torch.randn(batch_size, 3)
id_idx = torch.randint(0, 58, (batch_size,))
ex_idx = torch.randint(0, 10, (batch_size,))
geo_hair_idx = torch.randint(0, 4, (batch_size,))
col_hair_idx = torch.randint(0, 3, (batch_size,))

sdf_gt = torch.randn(batch_size, 1)
col_gt = torch.randn(batch_size, 3)

landmarks_gt = torch.randn(batch_size, 68, 3)

results = []

for epoch in range(epochs):

    optimizer.zero_grad()

    sdf_pred, col_pred, delta, z_geo, z_col = i3dmm_model(x, id_idx, ex_idx, geo_hair_idx, col_hair_idx) #

    lm_flat = landmarks_gt.view(-1, 3)
    lm_encoded = fourier_features(lm_flat, 6)
    z_geo_expanded = z_geo.unsqueeze(1).expand(-1, 68, -1).reshape(-1, 384)
    delta_lm = i3dmm_model.deformNet(lm_encoded, z_geo_expanded)
    landmarks_pred_ref = lm_flat + delta_lm
    landmarks_pred_ref = landmarks_pred_ref.view(batch_size, 68, 3)

    
    loss, loss_geo, loss_def, loss_col, loss_reg, loss_lm = total_loss(loss_weights, z_geo, z_col, delta, landmarks_pred_ref, 
                                                                       col_pred, col_gt, sdf_pred, sdf_gt, clamp_value=0.1) #
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Total Loss: {loss.item():.4f}")