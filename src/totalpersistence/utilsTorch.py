import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch


def matrix_size_from_condensed_torch(d:torch.Tensor):
    '''
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)
    '''
    n_elements = d.shape[0]
    # Convertimos a tensor antes de aplicar torch.sqrt
    value = 8.0 * float(n_elements) + 1.0
    sqrt_value = torch.sqrt(torch.tensor(value, device=d.device))
    return int(0.5 * (sqrt_value - 1.0)) + 1
       
def general_position_distance_matrix_torch(X, perturb=1e-7,device='cuda'):
    n = len(X)
    Xperturbation = perturb * torch.rand((n * (n - 1) // 2),device=device)
    dX = torch.pdist(X) + Xperturbation
    return dX

def squareform_torch(X,force='no',checks=True): 
    s = X.shape
    
    # Validación de argumentos
    if force is not None:
        force = force.lower()
        if force == 'tomatrix' and len(s) != 1:
            raise ValueError("Forcing 'tomatrix' but input is not a condensed vector")
        elif force == 'tovector' and len(s) != 2:
            raise ValueError("Forcing 'tovector' but input is not a square matrix")

    # Vector condensado → Matriz cuadrada
    if len(s) == 1 or (force == 'tomatrix' and len(s) == 1):
        if s[0] == 0:
            return torch.zeros((1, 1), dtype=X.dtype, device=X.device)

        # Calcular dimensión de la matriz
        n_elements = torch.tensor(s[0], dtype=torch.float32, device=X.device)
        d = int((torch.sqrt(8 *  n_elements + 1) + 1) // 2)
        if d * (d - 1) // 2 !=  n_elements:
            raise ValueError('El tamaño del vector no corresponde a una matriz condensada')

        M = torch.zeros((d, d), dtype=X.dtype, device=X.device)
        triu_indices = torch.triu_indices(d, d, 1, device=X.device)
        M[triu_indices[0], triu_indices[1]] = X
        M = M + M.T  # Hacer simétrica
        return M

    # Matriz cuadrada → Vector condensado
    elif len(s) == 2 or (force == 'tovector' and len(s) == 2):
        if s[0] != s[1]:
            raise ValueError('La matriz debe ser cuadrada')
        
        if checks:
            if not torch.allclose(X, X.T):
                raise ValueError('La matriz de distancia debe ser simétrica')
            if (X < 0).any():
                raise ValueError('Las distancias no pueden ser negativas')

        d = s[0]
        if d <= 1:
            return torch.tensor([], dtype=X.dtype, device=X.device)

        triu_indices = torch.triu_indices(d, d, 1, device=X.device)
        return X[triu_indices[0], triu_indices[1]].contiguous()
    
    else:
        raise ValueError(f"Input debe ser 1D o 2D, pero tiene dimensión {len(s)}")
    

def lipschitzTorch(dX, dY):
    return torch.max(dY / dX)

def make_conematrix_torch(DX, DY, DY_fy, eps):
    n = len(DX)
    m = len(DY)
    
    D = torch.zeros((n + m + 1, n + m + 1),device=DY.device)
    
    D[0:n, 0:n] = DX
    D[n : n + m, n : n + m] = DY

    D[0:n, n : n + m] = DY_fy
    D[n : n + m, 0:n] = DY_fy.T

    R = torch.inf

    D[n + m, n : n + m] = R
    D[n : n + m, n + m] = R

    D[n + m, :n] = eps
    D[:n, n + m] = eps

    return D

DEBUG = True
def log(*args, **kwargs):
    """
    Log messages if DEBUG is True.
    """
    if DEBUG:
        print(*args, **kwargs)


def conematrix_torch(dX:torch.Tensor, dY:torch.Tensor, cone_eps=0.0):
    
    n = matrix_size_from_condensed_torch(dX)
    m = matrix_size_from_condensed_torch(dY)

    device = dX.device
    f = torch.arange(m,device=device)              

    i, j = torch.triu_indices(n,n, offset=1,device=device)
    f_i, f_j = f[i], f[j]
   
    dY_ff = squareform_torch(dY,'tomatrix')[f_i, f_j]  ###    testear       ##
    rows = torch.arange(n,device=device)
    cols = torch.arange(m,device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    indices = torch.stack((grid_y, grid_x)) 
    

    i = indices[0].flatten() # TODO probar que es igual a gridyflatten
    j = indices[1].flatten() # lo mismo
    DY_fy = torch.full((n, m), float('inf'), dtype=dY.dtype, device=device)

    mask = torch.isin(j, f)
    i = i[mask]
    j = j[mask]
   
    f_i = f[i]
    
    DY_fy[i, j] = squareform_torch(dY,'tomatrix')[f_i, j]
    L = lipschitzTorch(dX, dY_ff)
    log(f"lipschitz constant: {L:.2f}")
    dY = dY / L
    D = make_conematrix_torch(squareform_torch(dX,'tomatrix'), squareform_torch(dY,'tomatrix'), DY_fy, cone_eps)
    return D, L


