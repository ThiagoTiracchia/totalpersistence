import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch

def print_diagram(title, diagram):
    print(title)
    for dim, dgm in enumerate(diagram):
        print(f"Dimension {dim}:")
        if len(dgm) == 0:
            print("  No points")
        else:
            for point in dgm:
                print(f"  {point}")

def findclose(x, A, tol=1e-5):
    return ((x + tol) >= A) & ((x - tol) <= A)


def lipschitz(dX, dY):
    return np.max(dY / dX)


def matrix_size_from_condensed(dX):
    n = len(dX)
    return int(0.5 * (np.sqrt(8 * n + 1) - 1) + 1)


def to_condensed_form(i, j, m):
    return m * i + j - ((i + 2) * (i + 1)) // 2.0


def general_position_distance_matrix(X, perturb=1e-7):
    n = len(X)
    Xperturbation = perturb * np.random.rand((n * (n - 1) // 2))
    dX = pdist(X) + Xperturbation
    return dX


def conematrix(DX, DY, DY_fy, eps):
    n = len(DX)
    m = len(DY)

    D = np.zeros((n + m + 1, n + m + 1))
    D[0:n, 0:n] = DX
    D[n : n + m, n : n + m] = DY

    D[0:n, n : n + m] = DY_fy
    D[n : n + m, 0:n] = DY_fy.T

    R = np.inf
    # R = max(DX.max(), DY_fy.max()) + 1 instead of np.inf

    D[n + m, n : n + m] = R
    D[n : n + m, n + m] = R

    D[n + m, :n] = eps
    D[:n, n + m] = eps

    return D


def format_bars(bars):
    bars = [np.array(b) for b in bars]
    lens = list(map(len, bars))
    for i in range(len(bars)):
        if all(l == 0 for l in lens[i:]):
            bars = bars[:i]
            break
    return bars


def kercoker_bars(dgm, dgmX, dgmY, cone_eps, tol=1e-11):
    """
    Find cokernel and kernel bars in the persistence diagram.
    TODO: optimize
    """
    coker_dgm = [[] for _ in range(len(dgm))]
    ker_dgm = [[] for _ in range(len(dgm))]
    for k in range(len(dgm)): # dimension cone diagram
        for r in dgm[k]:
            b, d = r
            if d > cone_eps + tol:
                # coker
                # b_c = b_y_i
                # d_c = d_y_i
                m = findclose(b, dgmY[k][:, 0], tol) & findclose(d, dgmY[k][:, 1], tol)
                if sum(m):
                    coker_dgm[k].append((b, d))

                # b_c = b_y_i
                # d_c = b_x_j
                if any(findclose(b, dgmY[k][:, 0], tol)) and any(findclose(d, dgmX[k][:, 0], tol)):
                    coker_dgm[k].append((b, d))

                # ker
                if k > 0:
                    # b_c = b_x_i (dim-1)
                    # d_c = d_x_i (dim-1)
                    m = findclose(b, dgmX[k - 1][:, 0], tol) & findclose(d, dgmX[k - 1][:, 1], tol)
                    if sum(m):
                        ker_dgm[k - 1].append((b, d))

                    # b_c = d_y_i (dim-1)
                    # d_c = d_x_j (dim-1)
                    if any(findclose(b, dgmY[k - 1][:, 1], tol)) and any(findclose(d, dgmX[k - 1][:, 1], tol)):
                        ker_dgm[k - 1].append((b, d))

    coker_dgm = format_bars(coker_dgm)
    ker_dgm = format_bars(ker_dgm)
    return coker_dgm, ker_dgm




######################Funciones necesarias en torch#########################################

def general_position_distance_matrix_torch(X, perturb=1e-7,device='cuda'):
    n = len(X)
    Xperturbation = perturb * torch.rand((n * (n - 1) // 2),device=device)
    dX = torch.pdist(X) + Xperturbation
    return dX

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

def squareform_torch(X,force='no',checks=True): ############### revisar esta funcion si es correcta!!! #########
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
    #### imporante que vengan llamadas  general_position_distance_matrix para las matrices previamente

    ## es el f verdaderamente necesario que sea pasado por parametro?
    n = matrix_size_from_condensed_torch(dX)
    m = matrix_size_from_condensed_torch(dY)

    device = dX.device
    f = torch.arange(m,device=device)              

    # dY_ff = d(f(x_i),f(x_j)) para todo i,j
    i, j = torch.triu_indices(n,n, offset=1,device=device)
    f_i, f_j = f[i], f[j]
    # f_pos = to_condensed_form(f_i, f_j, m) # broken
    # dY_ff = dY[f_pos.astype(int)]

    ############################### 
    dY_ff = squareform_torch(dY,'tomatrix')[f_i, f_j]  ###    testear       ##
    
    #########################################
    # dY_fy = d(f(x_i),y_j) para todo i,j
    
    #### indices = np.indices((n, m)) lo de abajo es la supuesta traduccion "revisar " ###

    rows = torch.arange(n,device=device)
    cols = torch.arange(m,device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    indices = torch.stack((grid_y, grid_x)) 

    ######################################################################################
    i = indices[0].flatten()
    j = indices[1].flatten()
    f_i = f[i]

    # DY_fy = np.zeros((n, m))
    # DY_fy[i, j] = np.inf
    # DY_fy[i, j] = squareform(dY)[f_i, j]

    # La entrada:
    # DY_fy[i, j] entre un índice cualquiera "i" (de X) y un "j" (de la llegada Y)
    # tiene que ser infinito en caso de que "j" no esté en la imagen de f.

    #DY_fy = torch.ones((n, m), dtype=float,device=device) * torch.inf
    DY_fy = torch.full((n, m), float('inf'), dtype=dY.dtype, device=device)

   # ijs = [(ii, jj) for ii, jj in zip(i, j) if jj in f_i]
    # i, j = zip(*ijs)
    # i = torch.array(i, dtype=int,device=device)
    # j = torch.array(j, dtype=int,device=device).

    ##### esto tambien hacer un test #
    mask = torch.isin(j, f)
    i = i[mask]
    j = j[mask]
   
    f_i = f[i]
    ####################################
    DY_fy[i, j] = squareform_torch(dY,'tomatrix')[f_i, j]
    
    L = lipschitzTorch(dX, dY_ff)
    log(f"lipschitz constant: {L:.2f}")

    dY = dY / L

    # dX     DY_fy
    # DY_fy  dY

    D = make_conematrix_torch(squareform_torch(dX,'tomatrix'), squareform_torch(dY,'tomatrix'), DY_fy, cone_eps)
    return D


#####################################################################################
def conematrix_numpy(dX:torch.Tensor, dY:torch.Tensor, f:torch.Tensor, maxdim=1, cone_eps=0.0, tol=1e-11):
    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)

    # dY_ff = d(f(x_i),f(x_j)) para todo i,j
    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]
    # f_pos = to_condensed_form(f_i, f_j, m) # broken
    # dY_ff = dY[f_pos.astype(int)]
    
    dY_ff = squareform(dY)[f_i, f_j]

    # dY_fy = d(f(x_i),y_j) para todo i,j
    indices = np.indices((n, m))
    i = indices[0].flatten()
    j = indices[1].flatten()
    f_i = f[i]

    # DY_fy = np.zeros((n, m))
    # DY_fy[i, j] = np.inf
    # DY_fy[i, j] = squareform(dY)[f_i, j]

    # La entrada:
    # DY_fy[i, j] entre un índice cualquiera "i" (de X) y un "j" (de la llegada Y)
    # tiene que ser infinito en caso de que "j" no esté en la imagen de f.

    DY_fy = np.ones((n, m), dtype=float) * np.inf
    
    ijs = [(ii, jj) for ii, jj in zip(i, j) if jj in f_i]
    i, j = zip(*ijs)
    i = np.array(i, dtype=int)
    j = np.array(j, dtype=int)

    f_i = f[i]

    DY_fy[i, j] = squareform(dY)[f_i, j]
    
    L = lipschitz(dX, dY_ff)
    log(f"lipschitz constant: {L:.2f}")

    dY = dY / L

    # dX     DY_fy
    # DY_fy  dY

    D = conematrix(squareform(dX), squareform(dY), DY_fy, cone_eps)
    return D