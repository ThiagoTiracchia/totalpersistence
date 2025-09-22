import cProfile
import pstats
import io
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import *


def previa_ker_cone():
    dX, dY, f = create_test_data()
    n = matrix_size_from_condensed(dX)
    m = matrix_size_from_condensed(dY)

    f = np.array(f)

    
    i, j = np.triu_indices(n, k=1)
    f_i, f_j = f[i], f[j]
    
    
    dY_ff = squareform(dY)[f_i, f_j]

    
    indices = np.indices((n, m))
    i = indices[0].flatten()
    j = indices[1].flatten()
    f_i = f[i]

    

    DY_fy = np.ones((n, m), dtype=float) * np.inf
    
    ijs = [(ii, jj) for ii, jj in zip(i, j) if jj in f_i]
    i, j = zip(*ijs)
    i = np.array(i, dtype=int)
    j = np.array(j, dtype=int)

    f_i = f[i]

    DY_fy[i, j] = squareform(dY)[f_i, j]
    
    L = lipschitz(dX, dY_ff)
   

    dY = dY / L
    return squareform(dX), squareform(dY), DY_fy


def create_test_data():
    
    n_points = 100
    m_points = 100
    
    # Puntos aleatorios en 2D
    X = np.random.rand(n_points, 2)
    Y = np.random.rand(m_points, 2)
    print(X)
  
    dX = pdist(X)
    dY = pdist(Y)
    print(dX)
    
    f = np.random.randint(0, m_points, size=n_points)
    
    return dX, dY, f

def test_conematrix(dX, dY, DY_fy):
    """Función de prueba para conematrix"""
    print("Generando datos de prueba...")
   
    
    print("Ejecutando conematrix")
    result =  conematrix(dX, dY, DY_fy, 0.1)
    return result


def test_conematrix_numpy():
    """Función de prueba para conematrix_numpy"""
    print("Generando datos de prueba...")
    dX, dY, f = create_test_data()
    
    print("Ejecutando conematrix_numpy")
    result = conematrix_numpy(dX, dY, f, maxdim=1, cone_eps=0.1)
    return result

def test_kercoker_bars():

    coker_dgm, ker_dgm = kercoker_bars(cone_dgm, dgmX, dgmY, cone_eps, tol)

##----------------------------------------------------------------------------
def run_comprehensive_profiling():
    
    print("=== PROFILING COMPLETO DE UTILS.PY ===\n")
    
    # Profiling de conematrix_numpy
    print("1. Profiling de conematrix_numpy:")
    profiler = cProfile.Profile()
    profiler.enable()
    
    result1 = test_conematrix_numpy()
    
    profiler.disable()
    print("   ✓ Completado\n")
    

    print("2. Profiling de conematrix:")
    dX, dY, DY_fy = previa_ker_cone()
    

    profiler2 = cProfile.Profile()
    profiler2.enable()
    
    result2 = test_conematrix(dX, dY, DY_fy)
    
    profiler2.disable()


    print("   ✓ Completado\n")


    print("3. Profiling de kerkone:")
    dX, dY, DY_fy = previa_ker_cone()
    D = conematrix(dX, dY, DY_fy, 0.1)
    dgmX = ripser(dX, distance_matrix=True, maxdim=maxdim)["dgms"]
    dgmY = ripser(dY, distance_matrix=True, maxdim=maxdim)["dgms"]
    cone_dgm = ripser(D, maxdim=maxdim, distance_matrix=True)["dgms"]

    profiler3 = cProfile.Profile()
    profiler3.enable()
    
    result3 = test_kercoker_bars(cone_dgm, dgmX, dgmY, 0.1, 1e-11)
    
    profiler3.disable()

    
    print("   ✓ Completado\n")
    
    return profiler, profiler2

def analyze_profiling_results(profiler, name="Function"):
    
    print(f"\n=== RESULTADOS DE PROFILING: {name} ===")
    
    # Crear stream para capturar output
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    
    # Ordenar por tiempo total
    stats.sort_stats('cumulative')
    
    # Mostrar top 20 funciones
    print("\nTop 20 funciones por tiempo acumulado:")
    stats.print_stats(20)
    
   
    
    # Guardar resultados en archivo
    filename = f"profiling_results_{name.lower().replace(' ', '_')}.txt"
    with open(filename, 'w') as f:
        f.write(s.getvalue())
    
    print(f"Resultados guardados en {filename}")


def memory_profiling():
    """Profiling de memoria (requiere memory_profiler)"""

    from memory_profiler import profile
        
    @profile
    def test_memory():
            dX, dY, f = create_test_data()
            result = conematrix_numpy(dX, dY, f)
            return result
        
    print("=== MEMORY PROFILING ===")
    test_memory()
        
prof1 = run_comprehensive_profiling()        
analyze_profiling_results(prof1, "conematrix_numpy")

memory_profiling()