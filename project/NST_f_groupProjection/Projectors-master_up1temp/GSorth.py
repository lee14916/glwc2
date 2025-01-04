from sympy.matrices.utilities import _get_intermediate_simp
from sympy.core.function import expand_mul
import numpy as np
from sympy import Matrix as spMatrix

#from scypy/matrices/decompositions.py
#adding an .expand() solve all the issues!

def _QRdecomposition(M):
    r"""Returns a QR decomposition.
    Explanation
    ===========
    A QR decomposition is a decomposition in the form $A = Q R$
    where
    - $Q$ is a column orthogonal matrix.
    - $R$ is a upper triangular (trapezoidal) matrix.
    A column orthogonal matrix satisfies
    $\mathbb{I} = Q^H Q$ while a full orthogonal matrix satisfies
    relation $\mathbb{I} = Q Q^H = Q^H Q$ where $I$ is an identity
    matrix with matching dimensions.
    For matrices which are not square or are rank-deficient, it is
    sufficient to return a column orthogonal matrix because augmenting
    them may introduce redundant computations.
    And an another advantage of this is that you can easily inspect the
    matrix rank by counting the number of columns of $Q$.
    If you want to augment the results to return a full orthogonal
    decomposition, you should use the following procedures.
    - Augment the $Q$ matrix with columns that are orthogonal to every
      other columns and make it square.
    - Augument the $R$ matrix with zero rows to make it have the same
      shape as the original matrix.
    The procedure will be illustrated in the examples section.
    Examples
    ========
    A full rank matrix example:
    >>> from sympy import Matrix
    >>> A = Matrix([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> Q, R = A.QRdecomposition()
    >>> Q
    Matrix([
    [ 6/7, -69/175, -58/175],
    [ 3/7, 158/175,   6/175],
    [-2/7,    6/35,  -33/35]])
    >>> R
    Matrix([
    [14,  21, -14],
    [ 0, 175, -70],
    [ 0,   0,  35]])
    If the matrix is square and full rank, the $Q$ matrix becomes
    orthogonal in both directions, and needs no augmentation.
    >>> Q * Q.H
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> Q.H * Q
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> A == Q*R
    True
    A rank deficient matrix example:
    >>> A = Matrix([[12, -51, 0], [6, 167, 0], [-4, 24, 0]])
    >>> Q, R = A.QRdecomposition()
    >>> Q
    Matrix([
    [ 6/7, -69/175],
    [ 3/7, 158/175],
    [-2/7,    6/35]])
    >>> R
    Matrix([
    [14,  21, 0],
    [ 0, 175, 0]])
    QRdecomposition might return a matrix Q that is rectangular.
    In this case the orthogonality condition might be satisfied as
    $\mathbb{I} = Q.H*Q$ but not in the reversed product
    $\mathbb{I} = Q * Q.H$.
    >>> Q.H * Q
    Matrix([
    [1, 0],
    [0, 1]])
    >>> Q * Q.H
    Matrix([
    [27261/30625,   348/30625, -1914/6125],
    [  348/30625, 30589/30625,   198/6125],
    [ -1914/6125,    198/6125,   136/1225]])
    If you want to augment the results to be a full orthogonal
    decomposition, you should augment $Q$ with an another orthogonal
    column.
    You are able to append an arbitrary standard basis that are linearly
    independent to every other columns and you can run the Gram-Schmidt
    process to make them augmented as orthogonal basis.
    >>> Q_aug = Q.row_join(Matrix([0, 0, 1]))
    >>> Q_aug = Q_aug.QRdecomposition()[0]
    >>> Q_aug
    Matrix([
    [ 6/7, -69/175, 58/175],
    [ 3/7, 158/175, -6/175],
    [-2/7,    6/35,  33/35]])
    >>> Q_aug.H * Q_aug
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> Q_aug * Q_aug.H
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    Augmenting the $R$ matrix with zero row is straightforward.
    >>> R_aug = R.col_join(Matrix([[0, 0, 0]]))
    >>> R_aug
    Matrix([
    [14,  21, 0],
    [ 0, 175, 0],
    [ 0,   0, 0]])
    >>> Q_aug * R_aug == A
    True
    A zero matrix example:
    >>> from sympy import Matrix
    >>> A = Matrix.zeros(3, 4)
    >>> Q, R = A.QRdecomposition()
    They may return matrices with zero rows and columns.
    >>> Q
    Matrix(3, 0, [])
    >>> R
    Matrix(0, 4, [])
    >>> Q*R
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])
    As the same augmentation rule described above, $Q$ can be augmented
    with columns of an identity matrix and $R$ can be augmented with
    rows of a zero matrix.
    >>> Q_aug = Q.row_join(Matrix.eye(3))
    >>> R_aug = R.col_join(Matrix.zeros(3, 4))
    >>> Q_aug * Q_aug.T
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> R_aug
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])
    >>> Q_aug * R_aug == A
    True
    See Also
    ========
    sympy.matrices.dense.DenseMatrix.cholesky
    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    LUdecomposition
    QRsolve
    """
    # Q,R=np.linalg.qr(np.matrix(M).astype('complex128'))
    # return spMatrix(Q),spMatrix(R)

    dps    = _get_intermediate_simp(expand_mul, expand_mul)
    mat    = M.as_mutable()
    n      = mat.rows
    m      = mat.cols
    ranked = list()

    # Pad with additional rows to make wide matrices square
    # nOrig keeps track of original size so zeros can be trimmed from Q
    if n < m:
        nOrig = n
        n     = m
        mat   = mat.col_join(mat.zeros(n - nOrig, m))
    else:
        nOrig = n


    Q, R = mat.zeros(n, m), mat.zeros(m)

    for j in range(m):  # for each column vector
        tmp = mat[:, j]  # take original v

        for i in range(j):
            # print('s8',i,j)
            # subtract the project of mat on new vector
            R[i, j]  = dps(Q[:, i].dot(mat[:, j], hermitian=True))
            # print('s9')
            tmp     -= (Q[:, i] * R[i, j]).expand()

        tmp = dps(tmp)

        # normalize it
        R[j, j] = tmp.norm()

        if not R[j, j].is_zero:
            ranked.append(j)
            Q[:, j] = tmp / R[j, j]


    Q = Q.extract(range(nOrig), ranked)
    R = R.extract(ranked, range(R.cols))
    return M.__class__(Q), M.__class__(R)

def orth_rows(spM):
    aux_M = spM.tolist()
    size  = len(aux_M)
    aux_M = [ row for row in aux_M if row!=[0,]*size  ]
    rank  = np.linalg.matrix_rank( np.array(aux_M,dtype=np.complex128) )
    
    aux_M = spMatrix(aux_M).T
    Q, _  = _QRdecomposition(aux_M)
    return Q.T   