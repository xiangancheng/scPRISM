"""Verify the pinned official Tangram reference, not the local tangram fork."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
import torch

import scprism as prism
from scprism import mapping_optimizer as optimizer
from scprism import mapping_utils as mapping

REFERENCE = Path(__file__).parent / 'reference' / 'tangram_1_0_4'


def official():
    name = '_official_tangram_104'
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(REFERENCE)]
        sys.modules[name] = pkg
        for child in ['mapping_optimizer', 'utils', 'mapping_utils']:
            spec = importlib.util.spec_from_file_location(name + '.' + child, REFERENCE / (child + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            # Only adapt input conversions that fail on current numpy/torch.
            source = (REFERENCE / (child + '.py')).read_text(encoding='utf-8')
            if child == 'mapping_utils':
                source = source.replace('S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)',
                    'S = np.array(adata_sc[:, training_genes].X.toarray() if hasattr(adata_sc.X, "toarray") else adata_sc[:, training_genes].X, dtype="float32",)')
                source = source.replace('if density_prior == "rna_count_based":', 'if isinstance(density_prior, str) and density_prior == "rna_count_based":')
                source = source.replace('elif density_prior == "uniform":', 'elif isinstance(density_prior, str) and density_prior == "uniform":')
                source = source.replace('d = density_prior', 'd = np.asarray(density_prior) if density_prior is not None else None')
                source = source.replace('d = adata_sp.obs["uniform_density"]', 'd = np.asarray(adata_sp.obs["uniform_density"])')
            exec(compile(source, str(REFERENCE / (child + '.py')), 'exec'), module.__dict__)
            setattr(pkg, child, module)
    return sys.modules[name]


def data():
    rng = np.random.RandomState(7)
    ref = AnnData(sparse.csr_matrix(rng.poisson(3., (12, 5)).astype('float32')),
        obs=pd.DataFrame({'type': ['a'] * 6 + ['b'] * 6}, index=[f'r{i}' for i in range(12)]),
        var=pd.DataFrame(index=[f'G{i}' for i in range(5)]))
    space = AnnData(sparse.csr_matrix(rng.poisson(4., (8, 5)).astype('float32')),
        obs=pd.DataFrame(index=[f's{i}' for i in range(8)]), var=ref.var.copy())
    space.obsm['spatial'] = rng.rand(8, 2)
    return ref, space


def test_only_laplacian_extends_official_mapping_signature():
    ref = official().mapping_utils
    assert list(inspect.signature(prism.pp_adatas).parameters) == list(inspect.signature(ref.pp_adatas).parameters)
    assert list(inspect.signature(prism.map_cells_to_space).parameters) == list(inspect.signature(ref.map_cells_to_space).parameters) + ['lambda_graph_laplacian']
    for name in ['project_genes', 'project_cell_annotations', 'compare_spatial_geneexp', 'cross_val']:
        assert ast.dump(ast.parse(inspect.getsource(getattr(prism, name)))) == ast.dump(ast.parse(inspect.getsource(getattr(official().utils, name))))
    assert not hasattr(prism, 'mapping_hyperparameter_tuning')
    assert not hasattr(prism, 'cell_type_mapping')


def test_reference_and_unchanged_utilities_match_official_hashes():
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    for name, meta in manifest['files'].items():
        if name == '__init__.py':
            continue
        assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == meta['sha256']
    root = Path(prism.__file__).parent
    for name in ['utils.py', 'plot_utils.py']:
        # Windows edits may change line endings; every source line is preserved.
        assert (root / name).read_text().splitlines() == (REFERENCE / name).read_text().splitlines()
    raw = ast.parse((REFERENCE / 'mapping_optimizer.py').read_text())
    cls = next(n for n in raw.body if isinstance(n, ast.ClassDef) and n.name == 'MapperConstrained')
    assert ast.dump(cls) == ast.dump(ast.parse(inspect.getsource(optimizer.MapperConstrained)).body[0])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('mode', ['cells', 'clusters', 'constrained'])
def test_zero_graph_mapping_matches_official(device, mode):
    ref, space = data()
    expected_ref, expected_space = ref.copy(), space.copy()
    official().mapping_utils.pp_adatas(expected_ref, expected_space)
    prism.pp_adatas(ref, space)
    np.testing.assert_array_equal(ref.X.toarray(), expected_ref.X.toarray())
    assert ref.uns == expected_ref.uns
    pd.testing.assert_frame_equal(space.obs, expected_space.obs)
    # No implicit spatial graph is introduced by Tangram preprocessing.
    assert not space.obsp
    args = dict(mode=mode, cluster_label='type' if mode == 'clusters' else None,
        device=device, num_epochs=12, random_state=42, verbose=False,
        density_prior='uniform', lambda_g1=1., lambda_g2=.2, lambda_d=1., lambda_r=.001)
    if mode == 'constrained':
        args.update(target_count=8)
    expected = official().mapping_utils.map_cells_to_space(expected_ref, expected_space, **args)
    actual = prism.map_cells_to_space(ref, space, lambda_graph_laplacian=0., **args)
    np.testing.assert_array_equal(actual.X, expected.X)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    pd.testing.assert_frame_equal(actual.uns['train_genes_df'], expected.uns['train_genes_df'])
    assert actual.uns['training_history'] == expected.uns['training_history']
    expected_gene = official().utils.project_genes(expected, expected_ref.copy(), cluster_label=args['cluster_label'])
    actual_gene = prism.project_genes(actual, ref.copy(), cluster_label=args['cluster_label'])
    np.testing.assert_array_equal(actual_gene.X, expected_gene.X)


def test_only_graph_energy_changes_loss_and_gradient():
    ref, space = data()
    S, G = ref.X.toarray(), space.X.toarray()
    L = np.eye(8, dtype='float32')
    L[:2, :2] = [[1., -1.], [-1., 1.]]
    args = dict(S=S, G=G, d=np.full(8, 1 / 8), random_state=42,
                lambda_g1=1., lambda_g2=.2, lambda_d=1., lambda_r=.001)
    native = official().mapping_optimizer.Mapper(**args)
    extended = optimizer.Mapper(**args, lambda_graph_laplacian=.03, spatial_graph_laplacian=sparse.csr_matrix(L))
    np.testing.assert_array_equal(native.M.detach(), extended.M.detach())
    P = torch.softmax(extended.M, dim=1)
    energy = torch.sum((P @ torch.tensor(L)) * P) * G.shape[0] / S.shape[0]
    old_loss = native._loss_fn(False)[0]
    new_loss = extended._loss_fn(False)[0]
    torch.testing.assert_close(new_loss, old_loss + .03 * energy, rtol=0, atol=0)
    native_grad = torch.autograd.grad(old_loss, native.M)[0]
    graph_grad = torch.autograd.grad(energy, extended.M, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(new_loss, extended.M)[0], native_grad + .03 * graph_grad)


@pytest.mark.parametrize('prior', ['uniform', 'rna_count_based', None, np.full(8, 1 / 8)])
def test_dense_input_and_density_conversion_match_official(prior):
    ref, space = data()
    ref.X, space.X = ref.X.toarray(), space.X.toarray()
    expected_ref, expected_space = ref.copy(), space.copy()
    official().mapping_utils.pp_adatas(expected_ref, expected_space)
    prism.pp_adatas(ref, space)
    args = dict(num_epochs=3, random_state=42, verbose=False, density_prior=prior)
    expected = official().mapping_utils.map_cells_to_space(expected_ref, expected_space, **args)
    actual = prism.map_cells_to_space(ref, space, **args)
    np.testing.assert_array_equal(actual.X, expected.X)
    assert actual.uns['training_history'] == expected.uns['training_history']


def test_graph_preprocessing_cannot_affect_zero_graph_mapping():
    ref, space = data()
    prism.pp_adatas(ref, space)
    before = prism.map_cells_to_space(ref, space, num_epochs=3, random_state=42, verbose=False)
    prism.build_spatial_graph(space, n_neighbors=3, prune_by_expression=True, sim_threshold=.9)
    after = prism.map_cells_to_space(ref, space, num_epochs=3, random_state=42, verbose=False)
    np.testing.assert_array_equal(before.X, after.X)
    assert before.uns['training_history'] == after.uns['training_history']


def test_graph_panel_must_match_mapping_split():
    ref, space = data()
    prism.pp_adatas(ref, space)
    prism.build_spatial_graph(space, n_neighbors=3, prune_by_expression=True)
    with pytest.raises(ValueError, match='Rebuild expression pruning'):
        prism.map_cells_to_space(ref, space, cv_train_genes=['g0', 'g1'],
                                 lambda_graph_laplacian=.02, num_epochs=3, verbose=False)
