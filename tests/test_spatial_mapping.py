"""Spatial graph integration tests without images or external data."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse
from anndata import AnnData
import scprism as prism
from scprism.mapping_optimizer import Mapper


def spatial_data():
    a = AnnData(sparse.csr_matrix([[5., 1.], [4., 1.], [1., 5.], [1., 4.]]),
                obs=pd.DataFrame({'domain': ['a', 'a', 'b', 'b']}, index=list('abcd')),
                var=pd.DataFrame(index=['g1', 'g2']))
    a.obsm['spatial'] = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    return a


def test_isolated_nodes_have_zero_laplacian():
    a = spatial_data()
    a.obsp['spatial_connectivities'] = sparse.csr_matrix((4, 4))
    np.testing.assert_array_equal(prism.spatial_graph_laplacian(a), np.zeros((4, 4)))
    a.obsp['spatial_connectivities'] = sparse.csr_matrix(([1., 1.], ([0, 1], [1, 0])), shape=(4, 4))
    L = prism.spatial_graph_laplacian(a)
    np.testing.assert_array_equal(L[2:], np.zeros((2, 4)))
    np.testing.assert_allclose(L[:2, :2], [[1., -1.], [-1., 1.]])


def test_preprocessing_preserves_prepared_graph():
    a, ref = spatial_data(), spatial_data()
    prism.build_spatial_graph(a, n_neighbors=2, prune_by_expression=True,
                              expression_genes=['g1', 'g2'], sim_threshold=0.9)
    W, D = a.obsp['spatial_connectivities'].copy(), a.obsp['spatial_distances'].copy()
    prism.pp_adatas(ref, a)
    np.testing.assert_array_equal(a.obsp['spatial_connectivities'].toarray(), W.toarray())
    np.testing.assert_array_equal(a.obsp['spatial_distances'].toarray(), D.toarray())


def test_mapper_consumes_prepared_graph_without_second_pruning():
    a, ref = spatial_data(), spatial_data()
    prism.build_spatial_graph(a, n_neighbors=2, prune_by_expression=True,
                              expression_genes=['g1', 'g2'], sim_threshold=0.9)
    prism.pp_adatas(ref, a)
    W = a.obsp['spatial_connectivities'].copy()
    expected = prism.spatial_graph_laplacian(a)
    captured = []
    original = Mapper.train
    def train(mapper, **kwargs):
        captured.append(mapper.spatial_graph_laplacian.to_dense().detach().cpu().numpy())
        return original(mapper, **kwargs)
    with patch.object(Mapper, 'train', train):
        result = prism.map_cells_to_space(ref, a, num_epochs=3, random_state=42,
                    density_prior=None, lambda_graph_laplacian=0.2,
                    verbose=False)
    np.testing.assert_allclose(captured[0], expected)
    np.testing.assert_array_equal(a.obsp['spatial_connectivities'].toarray(), W.toarray())
    assert np.isfinite(result.X).all()
    np.testing.assert_allclose(result.X.sum(axis=1), 1, atol=1e-6)
    projected = prism.project_genes(result, ref)
    assert projected.shape == (4, 2)
    assert np.isfinite(projected.X).all()
    np.testing.assert_allclose(projected.X, result.X.T @ np.asarray(ref.X), atol=1e-6)
    prism.project_cell_annotations(result, a, annotation='domain')
    assert a.obsm['tangram_ct_pred'].shape == (4, 2)


def test_expression_pruning_retains_only_existing_edges():
    a, baseline = spatial_data(), spatial_data()
    prism.build_spatial_graph(baseline, n_neighbors=2)
    prism.build_spatial_graph(a, n_neighbors=2, prune_by_expression=True,
                              expression_genes=['g1', 'g2'], sim_threshold=0.9)
    B, W = baseline.obsp['spatial_connectivities'], a.obsp['spatial_connectivities']
    assert W.multiply(B.astype(bool)).nnz == W.nnz
    assert (W != W.T).nnz == 0
    assert W[1, 2] == 0
    assert a.uns['scprism_graph_pruning']['removed_edges'] > 0


def test_expression_pruning_defaults_to_saved_training_genes():
    a = spatial_data()
    a = AnnData(sparse.hstack([a.X, sparse.csr_matrix(np.ones((4, 1)))]).tocsr(),
                obs=a.obs.copy(), var=pd.DataFrame(index=['g1', 'g2', 'heldout']),
                obsm={'spatial': a.obsm['spatial'].copy()})
    ref = spatial_data()
    prism.build_spatial_graph(a, n_neighbors=2)
    prism.pp_adatas(ref, a, genes=['g1', 'g2'])
    altered, explicit = a.copy(), a.copy()
    altered.X = sparse.csr_matrix(np.column_stack([
        a.X[:, :2].toarray(), [1000., 0., 1000., 0.]]))

    def prune(data, **kwargs):
        prism.build_spatial_graph(data, n_neighbors=2, prune_by_expression=True,
                                  sim_threshold=0.9, **kwargs)
        return data.obsp['spatial_connectivities'].toarray()

    expected = prune(explicit, expression_genes=['g1', 'g2'])
    assert explicit.uns['scprism_graph_pruning']['removed_edges'] > 0
    np.testing.assert_array_equal(prune(a), expected)
    np.testing.assert_array_equal(prune(altered), expected)


def test_expression_pruning_without_training_panel_fails_before_graph_write():
    a = spatial_data()
    prism.build_spatial_graph(a, n_neighbors=2)
    W, D = a.obsp['spatial_connectivities'].copy(), a.obsp['spatial_distances'].copy()
    with pytest.raises(ValueError, match='expression_genes'):
        prism.build_spatial_graph(a, n_neighbors=1, prune_by_expression=True)
    assert (a.obsp['spatial_connectivities'] != W).nnz == 0
    assert (a.obsp['spatial_distances'] != D).nnz == 0






def test_constrained_graph_loss_is_explicitly_rejected():
    with pytest.raises(ValueError, match='constrained'):
        prism.map_cells_to_space(spatial_data(), spatial_data(), mode='constrained',
                                 target_count=2, lambda_graph_laplacian=0.2)


