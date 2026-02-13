import torch
import numpy as np
import trimesh
from skimage import measure
from model import I3DMM


def generate_mesh(model, resolution=64, threshold=0.0):
    """
    Reconstroi a malha 3D a partir da rede neural implícita.
    resolution: Resolução da grade (ex: 64x64x64). Quanto maior, mais detalhe, mas mais pesado.
    threshold: O valor de SDF que consideramos a "pele" (geralmente 0).
    """
    device = next(model.parameters()).device
    
    # 1. Definir o "Aquário" (Grid 3D)
    # Criamos pontos de -1 a 1 em cada eixo
    voxel_coords = np.linspace(-1.0, 1.0, resolution)
    # Cria a grade completa
    grid_x, grid_y, grid_z = np.meshgrid(voxel_coords, voxel_coords, voxel_coords, indexing='ij')
    
    # Achata a grade para uma lista de pontos (N, 3)
    points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    points_tensor = torch.from_numpy(points).float().to(device)

    # 2. Definir Códigos Latentes FIXOS para inferência
    # Queremos ver UMA cabeça específica, então fixamos os IDs
    batch_size_inference = points_tensor.shape[0] # Precisamos repetir o ID para cada ponto da grade!
    
    # Vamos gerar a "Pessoa 0" com "Expressão 0"
    # Criamos tensores com o mesmo ID repetido milhares de vezes
    id_idx = torch.zeros((batch_size_inference,), dtype=torch.long, device=device)
    ex_idx = torch.zeros((batch_size_inference,), dtype=torch.long, device=device)
    geo_h_idx = torch.zeros((batch_size_inference,), dtype=torch.long, device=device)
    col_h_idx = torch.zeros((batch_size_inference,), dtype=torch.long, device=device)

    # 3. Processar em Fatias (Chunks)
    # Se passarmos 200.000 pontos de uma vez, a GPU explode. Vamos de 10k em 10k.
    chunk_size = 10000
    sdf_values = []
    rgb_values = []
    
    print(f"Processando {points_tensor.shape[0]} pontos na grade...")
    
    with torch.no_grad(): # Desliga o gradiente para economizar memória
        for i in range(0, points_tensor.shape[0], chunk_size):
            # Pega um pedacinho dos pontos
            p_chunk = points_tensor[i : i + chunk_size]
            id_chunk = id_idx[i : i + chunk_size]
            ex_chunk = ex_idx[i : i + chunk_size]
            gh_chunk = geo_h_idx[i : i + chunk_size]
            ch_chunk = col_h_idx[i : i + chunk_size]
            
            # Roda o Modelo
            sdf, rgb, _, _, _ = model(p_chunk, id_chunk, ex_chunk, gh_chunk, ch_chunk)
            
            # Guarda os resultados na CPU (para liberar GPU)
            sdf_values.append(sdf.cpu().numpy())
            # Não precisamos da cor de TODOS os pontos do espaço, só da superfície.
            # Mas vamos guardar por simplicidade agora.
    
    # Junta tudo num array só
    sdf_grid = np.concatenate(sdf_values).reshape(resolution, resolution, resolution)
    
    # 4. Marching Cubes (A Mágica)
    # O algoritmo encontra a superfície onde sdf = 0
    print("Executando Marching Cubes...")
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=threshold)
    except ValueError:
        print("Aviso: Nenhuma superfície encontrada (SDF nunca cruzou zero).")
        return None

    # O Marching Cubes retorna índices da matriz (0 a 63). Precisamos converter de volta para coordenadas (-1 a 1)
    # Regra de três simples para normalizar
    verts = verts * (2.0 / (resolution - 1)) - 1.0

    # 5. Pintar a Malha (Color Query)
    # Agora que sabemos onde estão os Vértices da superfície, perguntamos a cor deles
    print("Pintando a malha...")
    verts_tensor = torch.from_numpy(verts).float().to(device)
    num_verts = verts_tensor.shape[0]
    
    # IDs para os vértices
    id_verts = torch.zeros((num_verts,), dtype=torch.long, device=device)
    ex_verts = torch.zeros((num_verts,), dtype=torch.long, device=device)
    gh_verts = torch.zeros((num_verts,), dtype=torch.long, device=device)
    ch_verts = torch.zeros((num_verts,), dtype=torch.long, device=device)
    
    rgb_colors = []
    with torch.no_grad():
        for i in range(0, num_verts, chunk_size):
            v_chunk = verts_tensor[i : i + chunk_size]
            # ... índices ...
            chunk_ids = id_verts[i : i + chunk_size]
            chunk_ex = ex_verts[i : i + chunk_size]
            chunk_gh = gh_verts[i : i + chunk_size]
            chunk_ch = ch_verts[i : i + chunk_size]
            
            # Chamamos o modelo de novo, só para os vértices da casca
            _, rgb, _, _, _ = model(v_chunk, chunk_ids, chunk_ex, chunk_gh, chunk_ch)
            rgb_colors.append(rgb.cpu().numpy())

    vertex_colors = np.concatenate(rgb_colors)
    
    # 6. Salvar usando Trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
    return mesh

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instanciar modelo (Mesmos parâmetros do treino!)
    model = I3DMM(latent_dim=128, num_freqs_fourier=6).to(device)
    model.eval() # Modo de avaliação (desliga dropout, etc)
    
    print("Gerando forma aleatória (pesos não treinados)...")
    mesh = generate_mesh(model, resolution=64) # Comece com resolução baixa
    
    if mesh:
        mesh.export('resultado_aleatorio.obj')
        print("Salvo como 'resultado_aleatorio.obj'. Abra no Blender/MeshLab!")
    else:
        print("Falha ao gerar malha.")