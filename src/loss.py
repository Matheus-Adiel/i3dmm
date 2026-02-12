import torch

# Article: loss_geo(.) enforces good geometry reconstructions [Sec. 3.3.][Eq. 4]
def loss_geo(sdf_pred, sdf_gt, clamp_val):
    clamped_sdf = torch.clamp(sdf_pred, -clamp_val, clamp_val)
    clamped_sdf_gt = torch.clamp(sdf_gt, -clamp_val, clamp_val)

    loss = torch.abs(clamped_sdf - clamped_sdf_gt)

    return loss.mean()

# Article: loss_col(.) is used to train the ColorNet [Sec. 3.3.][Eq. 5]
def loss_col(col_pred, col_gt):
    loss = torch.abs(col_pred - col_gt)
    return loss.mean()

# Article: loss_landmark(.) is a sparse pairwise landmark supervision loss [Sec. 3.3.][Eq. 6]
def loss_landmark(landmarks_pred_ref):
    bach_size = landmarks_pred_ref.shape[0]

    loss = torch.tensor(0.0, device=landmarks_pred_ref.device) 
    num_pairs = 0

    if (bach_size < 2):
        return loss 

    for i in range(bach_size): 
        for j in range(i + 1, bach_size):
            diff = landmarks_pred_ref[i] - landmarks_pred_ref[j]
            loss += torch.linalg.norm(diff)
            num_pairs += 1

    if num_pairs > 0:
        return loss / num_pairs
    return loss  

def loss_def(delta):
    loss = torch.linalg.norm(delta, dim=1).mean()
    return loss

def loss_reg(z_geo, z_col):
    loss = (torch.linalg.norm(z_geo, dim=1) + torch.linalg.norm(z_col, dim=1)).mean()
    return loss

def total_loss(weights, z_geo, z_col, delta, landmarks_pred_ref, 
               col_pred, col_gt, sdf_pred, sdf_gt, clamp_value=0.1):

    comp_loss_geo = loss_geo(sdf_pred, sdf_gt, clamp_value)
    comp_loss_def = loss_def(delta)
    comp_loss_col = loss_col(col_pred, col_gt)
    comp_loss_reg = loss_reg(z_geo, z_col)
    comp_loss_lm = loss_landmark(landmarks_pred_ref)

    total_loss = 0
    total_loss += weights['Wg'] * comp_loss_geo
    total_loss += weights['Ws'] * comp_loss_def 
    total_loss += weights['Wc'] * comp_loss_col
    total_loss += weights['Wr'] * comp_loss_reg

    total_loss += weights['Wlm'] * comp_loss_lm

    return total_loss, comp_loss_geo, comp_loss_def, comp_loss_col, comp_loss_reg, comp_loss_lm