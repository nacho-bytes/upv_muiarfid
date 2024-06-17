import numpy as np

PFA =  np.zeros((5, 5, 2))
map_alphabet_to_idx = {
    "a": 0,
    "b": 1
}



PFA = np.array([
    # 0           1           2           3           4           5           6           7
    [[0.0, 0.0], [0.2, 0.0], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # 0
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.9], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # 1
    [[0.0, 0.0], [0.8, 0.0], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # 2
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.5], [0.3, 0.2], [0.0, 0.0], [0.0, 0.0]], # 3
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.5], [0.2, 0.0], [0.3, 0.0], [0.0, 0.0], [0.0, 0.0]], # 4
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.4, 0.6], [0.0, 0.0]], # 5
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.8, 0.0], [0.0, 0.0], [0.0, 0.2]], # 6
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], # 7
])

def crearMatrizM(pfa: np.array):
    return np.sum(pfa, axis=2)
    
def createVectorEpsilon(pfa: np.array):
    # xi(i) = - \sum_{v and j} P(i,v,j)log P(i,v,j)
    #el out y where es para cuando es == 0, así lo filtramos y devuelve 0 la multiplicación
    res =  -(pfa * np.log2(pfa, where=(pfa > 0), out=np.zeros_like(pfa))).sum(axis=(2, 1))
    #Nos aseguramos de que el último sea 0....
    res[-1] = 0
    return res

def calcularEntropiaDerivacional(pfa: np.array):
    #Creamos M y el vector epsilon
    M = crearMatrizM(pfa)
    epsilon_vec = createVectorEpsilon(pfa)
    print("La matriz característica M es:")
    print(np.round(M, 3))
    print("El vector epsilon es:")
    print(np.round(epsilon_vec, 3))
    #Matriz identidad
    I = np.identity(M.shape[0])
    #Nos aseguramos de que el eigen value sea estricamente menor que 1 
    eigen_values, _ = np.linalg.eig(M)
    print("Los valores propios de M son...")
    print(eigen_values)
    assert np.max(eigen_values) < 1, "El eigen value máximo es mayor que 1"
    #La fórmula
    return M, epsilon_vec, np.linalg.inv(I - M) @ epsilon_vec

M, epsilon_vec, res = calcularEntropiaDerivacional(PFA)


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)
r = bmatrix(np.round(res, 3))
print(np.round(res, 3))
print(r)
