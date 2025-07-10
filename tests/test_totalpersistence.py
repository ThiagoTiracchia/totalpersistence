import numpy as np
from totalpersistence import totalpersistence, kercoker_via_cone
from totalpersistence.utils import general_position_distance_matrix, squareform_torch, matrix_size_from_condensed_torch,general_position_distance_matrix_torch


def test_totalpersistence_basic():
    # Create simple test data
    X = np.array([[0, 0], [1, 0], [0, 1]])  # Triangle vertices
    Y = np.array([[0, 0], [2, 0], [0, 2]])  # Scaled triangle vertices
    f = np.array([0, 1, 2])  # Simple function values

    dX = general_position_distance_matrix(X)
    dY = general_position_distance_matrix(Y)

    coker_dgm, ker_dgm, cone_dgm, dgmX, dgmY = kercoker_via_cone(dX, dY, f, maxdim=2, cone_eps=0, tol=1e-11)

    # Call totalpersistence
    coker_bottleneck_distances, ker_bottleneck_distances, coker_matchings, ker_matchings = totalpersistence(
        coker_dgm, ker_dgm
    )

    # TODO: Add assertions for a relevant case

    # Basic assertions
    assert isinstance(coker_bottleneck_distances, list)
    assert isinstance(ker_bottleneck_distances, list)
    assert isinstance(coker_matchings, list)
    assert isinstance(ker_matchings, list)

    # The distance should be non-negative
    assert all(d >= 0 for d in coker_bottleneck_distances)
    assert all(d >= 0 for d in ker_bottleneck_distances)


def test_squareform():
    import torch
    from scipy.spatial.distance import squareform
    test_vector = np.array([1.0, 2.0, 3.0])  # Vector condensado
    test_matrix = np.array([[0, 1, 2], 
                            [1, 0, 3], 
                            [2, 3, 0]])      # Matriz cuadrada
    
    print("=== Test vector → matriz ===")
    scipy_mat = squareform(test_vector)
    torch_mat = squareform_torch(torch.from_numpy(test_vector)).numpy()

    print("SciPy:\n", scipy_mat)
    print("PyTorch:\n", torch_mat)
    print("¿Son iguales?", np.allclose(scipy_mat, torch_mat))

    print("\n=== Test matriz → vector ===")
    scipy_vec = squareform(test_matrix)
    torch_vec = squareform_torch(torch.from_numpy(test_matrix)).numpy()

    print("SciPy:", scipy_vec)
    print("PyTorch:", torch_vec)
    print("¿Son iguales?", np.allclose(scipy_vec, torch_vec))

    if torch.cuda.is_available():
        print("\n=== Test en GPU ===")
        gpu_vec = torch.from_numpy(test_vector).cuda()
        gpu_mat = squareform_torch(gpu_vec)
        print("Resultado GPU:", gpu_mat.cpu().numpy())
        print("¿Igual que CPU?", np.allclose(gpu_mat.cpu().numpy(), scipy_mat))

def test_funcionesTorch():
    import torch

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])

    XTorch = torch.from_numpy(X).to('cuda')
    YTorch = torch.from_numpy(Y).to('cuda')
    dX = general_position_distance_matrix(X)    
    dY = general_position_distance_matrix(Y)
   
    dXTorch = general_position_distance_matrix_torch(XTorch)    
    dYTorch = general_position_distance_matrix_torch(YTorch)  

    print("¿Son iguales general_position_distance_matrix  dXTorch y DX?", np.allclose(dXTorch.cpu(), dX))

    print("¿Son iguales general_position_distance_matrix  dYTorch y DY?", np.allclose(dYTorch.cpu(), dY))
    

    n = matrix_size_from_condensed_torch(dX)
    m = matrix_size_from_condensed_torch(dY)

    nTorch = matrix_size_from_condensed_torch(dXTorch)
    mTorch = matrix_size_from_condensed_torch(dYTorch)

    print("¿Son iguales general_position_distance_matrix  dXTorch y DX?", np.allclose(n, nTorch))

    print("¿Son iguales general_position_distance_matrix  dYTorch y DY?", np.allclose(m, mTorch))




    indices = np.indices((n, m))

    rows = torch.arange(n,device='cuda')
    cols = torch.arange(m,device='cuda')
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    indicesTorch = torch.stack((grid_y, grid_x))

    print("¿Son iguales como se calculan los indices indicesTorch y indices ?", np.allclose(indices, indicesTorch.cpu()))
    f = np.arange(m)

    i = indices[0].flatten()
    j = indices[1].flatten()


    f_i = f[i]
    ijs = [(ii, jj) for ii, jj in zip(i, j) if jj in f_i]
    i, j = zip(*ijs)
    i = np.array(i, dtype=int)
    j = np.array(j, dtype=int)

    ftorch = torch.arange(m,device='cuda')
    iTorch = indicesTorch[0].flatten()
    jTorch = indicesTorch[1].flatten()
    mask = torch.isin(jTorch, ftorch)
    iTorch = iTorch[mask]
    jTorch = jTorch[mask]

    print("¿Son iguales como se calcula la parte de zip i y iTorch ?", np.allclose(i, iTorch.cpu()))
    print("¿Son iguales como se calcula la parte de zip j y jTorch ?", np.allclose(j, jTorch.cpu()))


test_funcionesTorch()        
test_squareform()