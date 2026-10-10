"""The sole scPRISM mapping extension: a fixed spatial graph Laplacian."""
import numpy as np
from scipy import sparse


def spatial_graph_laplacian(adata_sp, normalized=True, return_sparse=False):
    """Read the prepared graph without changing or pruning its edges.

    Symmetric normalization gives isolated nodes zero rows and columns.
    """
    if 'spatial_connectivities' not in adata_sp.obsp:
        raise ValueError("Missing 'spatial_connectivities'; run build_spatial_graph() before mapping.")
    W = sparse.csr_matrix(adata_sp.obsp['spatial_connectivities'], dtype=np.float32).copy()
    W.eliminate_zeros()
    if not np.isfinite(W.data).all() or np.any(W.data < 0):
        raise ValueError('Spatial graph weights must be finite and nonnegative.')
    if (W != W.T).nnz or np.any(W.diagonal() != 0):
        raise ValueError('Prepared graph must be symmetric and have no self-loops.')
    degree = np.asarray(W.sum(axis=1)).ravel()
    if normalized:
        inv = np.zeros_like(degree)
        inv[degree > 0] = 1 / np.sqrt(degree[degree > 0])
        D = sparse.diags(inv)
        L = sparse.diags((degree > 0).astype(np.float32)) - D @ W @ D
    else:
        L = sparse.diags(degree) - W
    L = L.astype(np.float32).tocsr()
    return L if return_sparse else L.toarray()
