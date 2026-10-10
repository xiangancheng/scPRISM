"""Public spatial-only API contract after removing image processing."""
import inspect
import importlib.util
import scprism as prism


def test_depth_alignment_api_and_module_are_removed():
    assert not hasattr(prism, 'align_expression_depth')
    assert importlib.util.find_spec('scprism.expression_preprocessing') is None


def test_mapping_api_is_single_stage_linear_with_laplacian():
    from scprism.mapping_optimizer import Mapper
    assert not hasattr(prism, 'fit_graph_attention')
    assert importlib.util.find_spec('scprism.graph_attention') is None
    assert not hasattr(Mapper, 'train_staged')
    parameters = inspect.signature(prism.map_cells_to_space).parameters
    for removed in ('reference_layer', 'spatial_layer', 'expression_transform',
                    'training_strategy', 'stage1_epochs', 'stage2_epochs',
                    'graph_warmup_epochs', 'graph_key'):
        assert removed not in parameters
    assert 'lambda_graph_laplacian' in parameters
    projection = inspect.signature(prism.project_genes).parameters
    assert 'expression_layer' not in projection
    assert 'apply_mapping_scale' not in projection


def test_package_exposes_only_spatial_mapping_features():
    removed = (
        'extract_pathology_features', 'apply_pathology_edge_weights',

    )
    assert not any(hasattr(prism, name) for name in removed)
    # These are official Tangram utilities, not additional scPRISM algorithms.
    for name in ('create_segment_cell_df', 'count_cell_annotations',
                 'deconvolve_cell_annotations', 'df_to_cell_types'):
        assert callable(getattr(prism, name))
    assert importlib.util.find_spec('scprism.pathology_encoder') is None
    assert not any('pathology' in name for name in inspect.signature(prism.build_spatial_graph).parameters)
    for name in ('pp_adatas', 'map_cells_to_space', 'build_spatial_graph',
                 'plot_spatial_graph', 'spatial_graph_laplacian',
                 'project_genes', 'project_cell_annotations'):
        assert callable(getattr(prism, name))
