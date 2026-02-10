# Article: loss_geo(.) enforces good geometry reconstructions [Sec. 3.3.][Eq. 4]
def loss_geo(sdf_predict, sdf_gt, clamp_value):
    clamped_sdf = torch.clamp(sdf_predict, -clamp_value, clamp_value)
    clamped_sdf_gt = torch.clamp(sdf_gt, -clamp_value, clamp_value)

    loss = torch.abs(clamped_sdf - clamped_sdf_gt)

    return loss.mean()

# Article: loss_col(.) is used to train the ColorNet [Sec. 3.3.][Eq. 5]
def loss_col(color_predict, color_gt):
    loss = torch.abs(color_predict - color_gt)
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