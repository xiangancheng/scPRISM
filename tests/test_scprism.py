import sys
import os

# 优先导入当前目录下的包
sys.path.insert(0, r"D:\python_program\scPRISM")

import numpy as np
import pandas as pd
import anndata as ad
import scprism as sp

def test_scprism():
    print("=== 测试 D:\\python_program\\scPRISM 迁移与映射功能 ===")
    n_cells = 40
    n_spots = 15
    n_genes = 12

    np.random.seed(42)
    X_sc = np.random.poisson(lam=2.5, size=(n_cells, n_genes)).astype(np.float32)
    genes = [f"gene_{i}" for i in range(n_genes)]
    adata_sc = ad.AnnData(
        X=X_sc,
        obs=pd.DataFrame({"cell_type": np.random.choice(["TypeA", "TypeB"], size=n_cells)}),
        var=pd.DataFrame(index=genes),
    )

    X_sp = np.random.poisson(lam=4.0, size=(n_spots, n_genes)).astype(np.float32)
    coords = np.random.uniform(0, 300, size=(n_spots, 2)).astype(np.float32)
    adata_sp = ad.AnnData(
        X=X_sp,
        obs=pd.DataFrame(index=[f"spot_{j}" for j in range(n_spots)]),
        var=pd.DataFrame(index=genes),
    )
    adata_sp.obsm["spatial"] = coords

    print("1. 执行 sp.pp_adatas 预处理...")
    sp.pp_adatas(adata_sc, adata_sp, genes=genes)
    sp.build_spatial_graph(adata_sp, n_neighbors=6, prune_by_expression=True,
                          sim_threshold=0.25)

    print("2. 调用 sp.map_cells_to_space (开启图拉普拉斯损失)...")
    adata_map = sp.map_cells_to_space(
        adata_sc=adata_sc,
        adata_sp=adata_sp,
        mode="cells",
        num_epochs=20,
        learning_rate=0.1,
        lambda_g1=1.0,
        lambda_d=1.0,
        lambda_graph_laplacian=0.3,
        device="cpu",
        verbose=True,
    )

    print("\n3. 校验结果:")
    print("   scPRISM 包路径:", sp.__file__)
    print("   映射矩阵 shape:", adata_map.X.shape)
    print("   每行 Softmax 概率和:", np.mean(adata_map.X.sum(axis=1)))
    print("   图损失监控历史:", adata_map.uns["training_history"]["graph_laplacian_reg"][-1])

    assert adata_map.X.shape == (n_cells, n_spots)
    print("\n【全部测试通过！】scPRISM 包已成功独立建立并可直接使用。")

if __name__ == "__main__":
    test_scprism()
