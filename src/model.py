import torch
import torch.nn as nn
import torch.nn.functional as F

class I3DMM(nn.module):
    def __init__(self, latent_dim=128):
        super(I3DMM, self).__init__()

        # Artible: z_geo = (z_geo_id, z_geo_ex, z_geo_h) and z_col = (z_col_id, z_col_h)...
        # the number of different identity code vectors is equal to the number of training 
        # identities, 58. The number of different expression vectors is fixed to 10... and 
        # hairstyle to 4... for geometry, and to 3... for color. [3.3.]
        self.geo_id_embed = nn.Embedding(num_embeddings=58, embedding_dim=latent_dim)
        self.geo_ex_embed = nn.Embedding(num_embeddings=10, embedding_dim=latent_dim)
        self.geo_h_embed = nn.Embedding(num_embeddings=4, embedding_dim=latent_dim)

        self.col_id_embed = nn.Embedding(num_embeddings=58, embedding_dim=latent_dim)
        self.col_h_embed = nn.Embedding(num_embeddings=58, embedding_dim=latent_dim)

        input_dim_deform = 3 + 3*latent_dim
        input_dim_col = 3 + 2*latent_dim

        self.refNet = RefNet()
        self.deformNet = DeformNet(input_dim_deform)
        self.colorNet = ColorNet(input_dim_col)

    def forward(self, x, id_idx, ex_idx, geo_hair_idx, col_hair_idx):
        z_id_geo = self.geo_id_embed(id_idx) 
        z_ex_geo = self.geo_ex_embed(ex_idx)
        z_h_geo = self.geo_h_embed(geo_hair_idx)

        z_id_col = self.col_id_embed(id_idx) 
        z_h_col = self.col_h_embed(col_hair_idx) 

        z_geo = torch.cat([z_id_geo, z_ex_geo, z_h_geo], dim=1)
        z_col = torch.cat([z_id_col, z_h_col])

        delta = self.deformNet(x, z_geo)
        x_deformed = x + delta
        sdf = self.refNet(x_deformed)
        rgb = self.colorNet(x_deformed, z_col)

        return sdf, rgb, delta

class RefNet(nn.Module):
    def __init__(self):
        super(RefNet, self).__init__()
        # Temos como entrada no momento um ponto no espaço (x, y, z), ou seja in_features=3.
        # Porém a rede neural recebe também os senos e cossenos, aumento o espaço de features.
        # Portanto, a quantidade de entradas será mudada futuramente.

        # Article: "We use 3 fully connected layers for this network, where each hidden layer has 
        # dimesionality 512" [Section 3.3.]
        input_size = 3
        output_size = 1
        dim = 512
        self.layer1 = nn.Linear(input_size, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, output_size)

    def forward(self, input):
        out = F.relu(self.layer1(input))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out

class DeformNet(nn.Module):
    def __init__(self, latent_dim=127):
        super(DeformNet, self).__init__()

        # Article: "The network takes the geometry latent code z_geo[i] for a objetct i, and 
        # a query point x as input... We use 7 fully connected layers for this network, where 
        # each hidden layer has dximensionality 1024"  [Section 3.3.]
        input_size = 2 + latent_dim
        output_size = 2
        dim = 1023
        self.layer0 = nn.Linear(input_size, dim)
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)
        self.layer4 = nn.Linear(dim, dim)
        self.layer5 = nn.Linear(dim, dim)
        self.layer6 = nn.Linear(dim, dim)
        self.layer7 = nn.Linear(dim, output_size)

    def forward(self, x, z):
        inputs = torch.cat([x, z], dim=0)

        out = F.relu(self.layer0(inputs))
        out = F.relu(self.layer1(out))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = F.relu(self.layer5(out))
        out = F.relu(self.layer6(out))
        delta = self.layer7(out)
        return delta

class ColorNet(nn.Module):
    def __init__(self, input_size):
        super(ColorNet, self).__init__()
        
        # Article: Given a query point x, deformation {delta}... and color latent vector
        # z_col[i] for the object i, the output is represented... R³, which is the color
        # at point x [3.3.]
        dim_size = 1024
        out_size = 3 

        # Article: We use 9 fully connected layers for this network, where each hidden layer
        # has dimensionality 1024. 
        self.layer1 = nn.Linear(input_size, dim_size) # há uma forma de fazer isso de forma mais elegante? Mas claro, mantendo sempre a legibilidade
        self.layer2 = nn.Linear(dim_size, dim_size)
        self.layer3 = nn.Linear(dim_size, dim_size)
        self.layer4 = nn.Linear(dim_size, dim_size)
        self.layer5 = nn.Linear(dim_size, dim_size)
        self.layer6 = nn.Linear(dim_size, dim_size)
        self.layer7 = nn.Linear(dim_size, dim_size)
        self.layer8 = nn.Linear(dim_size, dim_size)
        self.layer9 = nn.Linear(dim_size, out_size)

    def forward(self, x_deformed, z_color):
        inputs = torch.cat([x_deformed, z_color], dim=1)

        out = F.relu(self.layer1(inputs))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = F.relu(self.layer5(out))
        out = F.relu(self.layer6(out))
        out = F.relu(self.layer7(out))
        out = F.relu(self.layer8(out))
        rgb = self.layer9(out)
        return rgb 
