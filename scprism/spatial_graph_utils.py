"""KNN construction and expression pruning before Tangram-style mapping."""
import logging
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def build_spatial_graph(adata_sp, method="knn", n_neighbors=6,
                        prune_by_expression=False, sim_threshold=0.85,
                        expression_genes=None):
    """Build an undirected KNN graph and optionally prune low-cosine edges.

    Uses X directly. Omitted expression_genes uses the training panel saved by
    pp_adatas. Pass the exact training split when evaluating held-out genes.
    Stores the adjacency, matching Euclidean distances and pruning metadata.
    """
    if method != 'knn':
        raise ValueError("Only method='knn' is supported.")
    if 'spatial' not in adata_sp.obsm:
        raise ValueError("No coordinates in adata_sp.obsm['spatial'].")
    coords = np.asarray(adata_sp.obsm['spatial'])
    if coords.ndim != 2 or not np.isfinite(coords).all():
        raise ValueError('Spatial coordinates must be a finite 2D array.')
    if not isinstance(n_neighbors, (int, np.integer)) or not 0 < n_neighbors < adata_sp.n_obs:
        raise ValueError('n_neighbors must be positive and smaller than the number of spatial observations.')
    if prune_by_expression:
        if not np.isfinite(sim_threshold) or not -1 <= sim_threshold <= 1:
            raise ValueError('sim_threshold must be finite and between -1 and 1.')
        if expression_genes is None:
            expression_genes = adata_sp.uns.get('training_genes')
        if expression_genes is None:
            raise ValueError("Pass expression_genes or run pp_adatas() to save uns['training_genes'].")
        expression_genes = list(expression_genes)
        if not expression_genes or len(set(expression_genes)) != len(expression_genes):
            raise ValueError('expression_genes must be nonempty and unique.')
        indices = adata_sp.var_names.get_indexer(expression_genes)
        if np.any(indices < 0):
            raise ValueError('expression_genes must be present in adata_sp.var_names.')
        X = adata_sp.X[:, indices]
        X = X.astype(float) if sp.issparse(X) else np.asarray(X, dtype=float)
        if not np.isfinite(X.data if sp.issparse(X) else X).all():
            raise ValueError('Expression used for pruning must be finite.')

    W = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors_graph(mode='connectivity')
    W = W.maximum(W.T).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    initial_edges = sp.triu(W, k=1).nnz
    baseline = W.astype(bool)
    if prune_by_expression:
        edges = W.tocoo()
        if sp.issparse(X):
            X = X.tocsr()
            norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
            dots = np.asarray(X[edges.row].multiply(X[edges.col]).sum(axis=1)).ravel()
        else:
            norms = np.linalg.norm(X, axis=1)
            dots = np.einsum('ij,ij->i', X[edges.row], X[edges.col])
        similarity = dots / (norms[edges.row] * norms[edges.col] + 1e-9)
        keep = similarity >= sim_threshold
        W = sp.csr_matrix((edges.data[keep], (edges.row[keep], edges.col[keep])), shape=W.shape)
        W.eliminate_zeros()
    if (W != W.T).nnz or W.multiply(baseline).nnz != W.nnz:
        raise RuntimeError('Pruning must preserve symmetry and cannot introduce edges.')
    row, col = W.nonzero()
    distances = np.linalg.norm(coords[row] - coords[col], axis=1)
    D = sp.csr_matrix((distances, (row, col)), shape=W.shape)
    final_edges = sp.triu(W, k=1).nnz
    adata_sp.obsp['spatial_connectivities'] = W
    adata_sp.obsp['spatial_distances'] = D
    adata_sp.uns['scprism_graph_pruning'] = {
        'method': 'knn', 'n_neighbors': int(n_neighbors),
        'initial_edges': int(initial_edges), 'final_edges': int(final_edges),
        'removed_edges': int(initial_edges - final_edges),
        'expression_genes': list(expression_genes) if prune_by_expression else None,
        'sim_threshold': float(sim_threshold) if prune_by_expression else None,
    }
    logging.info('KNN spatial pruning: %d -> %d undirected edges; removed %d.',
                 initial_edges, final_edges, initial_edges - final_edges)
    return adata_sp


def plot_spatial_graph(adata_sp, basis='spatial', edge_color='black', edge_alpha=0.5, edge_linewidth=0.5,
                       node_color='red', node_alpha=0.8, node_size=5, ax=None, title="Spatial Graph"):
    """
    Visualizes the spatial graph on top of the physical coordinates.

    Args:
        adata_sp (AnnData): AnnData object with 'spatial_connectivities' built.
        basis (str): Key in `obsm` holding the coordinates. Default is 'spatial'.
        edge_color (str): Color of the edges.
        edge_alpha (float): Opacity of the edges.
        edge_linewidth (float): Line width of the edges.
        node_color (str or list): Color of the spots.
        node_alpha (float): Opacity of the spots.
        node_size (float): Size of the spots.
        ax (matplotlib.axes.Axes): Matplotlib axes to plot on.
        title (str): Title of the plot.

    Returns:
        matplotlib.axes.Axes: The axis with the plot.
    """
    if 'spatial_connectivities' not in adata_sp.obsp:
        raise ValueError("Run `build_spatial_graph` first.")
    if basis not in adata_sp.obsm:
        raise ValueError(f"Basis '{basis}' not found in `obsm`.")
        
    coords = adata_sp.obsm[basis]
    W = adata_sp.obsp['spatial_connectivities']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # Plot edges on top (using a LineCollection is much faster and cleaner)
    from matplotlib.collections import LineCollection
    W_triu = sp.triu(W, k=1).tocoo()
    
    segments = []
    for i, j in zip(W_triu.row, W_triu.col):
        segments.append([
            [coords[i, 0], coords[i, 1]], 
            [coords[j, 0], coords[j, 1]]
        ])
        
    lc = LineCollection(segments, colors=edge_color, alpha=edge_alpha, linewidths=edge_linewidth, zorder=1)
    ax.add_collection(lc)

    # Plot nodes on top of edges for clear anchors
    ax.scatter(coords[:, 0], coords[:, 1], c=node_color, alpha=node_alpha, s=node_size, zorder=2, edgecolors='none')
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    
    return ax
