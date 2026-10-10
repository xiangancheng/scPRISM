"""Regression contracts for single-stage Tangram mapping plus Laplacian."""
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

import scprism as prism
from scprism.mapping_optimizer import Mapper


def pair():
    ref = AnnData(np.array([[4., 1.], [1., 5.], [3., 2.]], dtype=np.float32),
                  obs=pd.DataFrame(index=['r0', 'r1', 'r2']),
                  var=pd.DataFrame(index=['g0', 'g1']))
    space = AnnData(np.array([[5., 2.], [1., 4.]], dtype=np.float32),
                    obs=pd.DataFrame(index=['s0', 's1']),
                    var=pd.DataFrame(index=['g0', 'g1']))
    space.obsp['spatial_connectivities'] = sparse.csr_matrix([[0., 1.], [1., 0.]])
    space.obsp['spatial_distances'] = space.obsp['spatial_connectivities'].copy()
    prism.pp_adatas(ref, space)
    return ref, space


def test_dense_reference_and_custom_density():
    ref, space = pair()
    result = prism.map_cells_to_space(ref, space, num_epochs=2,
        density_prior=np.array([0.4, 0.6]), random_state=0, verbose=False)
    assert np.isfinite(result.X).all()
    assert np.isfinite(np.asarray(result.uns['training_history']['main_loss'], dtype=float)).all()
    np.testing.assert_allclose(result.X.sum(1), 1., atol=1e-6)


def test_sparse_graph_energy_and_gradient_match_trace_reference():
    ref, space = pair()
    L = np.array([[1., -1.], [-1., 1.]], dtype=np.float32)
    args = dict(S=ref.X, G=space.X, lambda_graph_laplacian=0.3, random_state=42)
    dense = Mapper(**args, spatial_graph_laplacian=L)
    sparse_mapper = Mapper(**args, spatial_graph_laplacian=sparse.csr_matrix(L))
    P = torch.softmax(dense.M, dim=1)
    expected = torch.trace(P @ torch.tensor(L) @ P.t()) * 2 / 3
    expected_grad = torch.autograd.grad(expected, dense.M)[0]
    assert sparse_mapper._loss_fn(False)[5] == pytest.approx(expected.item(), rel=1e-6)
    np.testing.assert_allclose(
        torch.autograd.grad(sparse_mapper._loss_fn(False)[0], sparse_mapper.M)[0],
        torch.autograd.grad(dense._loss_fn(False)[0], dense.M)[0], atol=1e-7)
    assert torch.isfinite(expected_grad).all()








@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0',
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_single_stage_linear_mapping_and_projection(device):
    ref, space = pair()
    S, G = ref.X.copy(), space.X.copy()
    W, D = space.obsp['spatial_connectivities'].copy(), space.obsp['spatial_distances'].copy()
    mapped = prism.map_cells_to_space(ref, space, num_epochs=5, device=device,
        lambda_g1=1., lambda_g2=0.2, lambda_d=1., density_prior='uniform',
        lambda_graph_laplacian=0.02, random_state=42, verbose=False)
    np.testing.assert_array_equal(ref.X, S)
    np.testing.assert_array_equal(space.X, G)
    assert (space.obsp['spatial_connectivities'] != W).nnz == 0
    assert (space.obsp['spatial_distances'] != D).nnz == 0
    assert 'stage1_mapping' not in mapped.layers
    history = mapped.uns['training_history']
    assert len(history['total_loss']) == 5 and 'stage' not in history
    for key in ['main_loss', 'vg_reg', 'kl_reg', 'graph_laplacian_reg']:
        assert np.isfinite(np.asarray(history[key], dtype=float)).all()
    np.testing.assert_allclose(mapped.X.sum(1), 1., atol=1e-6)
    projected = prism.project_genes(mapped, ref)
    np.testing.assert_allclose(projected.X, mapped.X.T @ S, atol=1e-6)


def test_mapper_loss_is_linear_expression_density_and_laplacian():
    ref, space = pair()
    L = torch.tensor([[1., -1.], [-1., 1.]])
    mapper = Mapper(S=ref.X, G=space.X, d=np.array([0.5, 0.5]),
        lambda_g1=1., lambda_g2=0.2, lambda_d=1.,
        lambda_graph_laplacian=0.03, spatial_graph_laplacian=L.numpy(), random_state=42)
    P = torch.softmax(mapper.M, dim=1)
    prediction = P.t() @ torch.tensor(ref.X)
    gv = torch.nn.functional.cosine_similarity(prediction, torch.tensor(space.X), dim=0).mean()
    vg = torch.nn.functional.cosine_similarity(prediction, torch.tensor(space.X), dim=1).mean()
    kl = torch.nn.functional.kl_div(torch.log(P.mean(0)), mapper.d, reduction='sum')
    energy = torch.trace(P @ L @ P.t()) * 2 / 3
    torch.testing.assert_close(mapper._loss_fn(False)[0], -gv-0.2*vg+kl+0.03*energy)


