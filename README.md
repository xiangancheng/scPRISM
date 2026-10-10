# scPRISM

scPRISM = 官方 Tangram 1.0.4 + KNN 图 + 表达剪枝 + 图拉普拉斯正则。

## 官方基准

仓库 https://github.com/broadinstitute/Tangram ，版本1.0.4，commit `d537ff936a1a26445a0f196c714d100d10e3e5ea`。原始源码、URL、SHA256和许可证存于 `tests/reference/tangram_1_0_4/`，与本项目旧`tangram/`目录独立。

`pp_adatas`、cluster聚合、初始化、GV/VG、密度、熵、Adam及gene/注释投影遵循官方。`utils.py`和`plot_utils.py`逐行一致（不计Windows换行）；`MapperConstrained`保持官方实现。官方已有分割后处理工具随utils恢复，没有新增病理或核分割模型。

## 使用流程

```python
import scprism as sp

# 数据已读取，ad_st.obsm['spatial']已有坐标。
# genes=None使用共同基因；有留出划分时传入实际训练基因。
sp.pp_adatas(ad_sc, ad_st, genes=None, gene_to_lowercase=True)
sp.build_spatial_graph(
    ad_st, method='knn', n_neighbors=6,
    prune_by_expression=True, sim_threshold=0.25,
)
adata_map = sp.map_cells_to_space(
    ad_sc, ad_st, mode='cells', device='cuda:0',
    num_epochs=500, learning_rate=0.1,
    density_prior='uniform', lambda_g1=1., lambda_d=1.,
    lambda_graph_laplacian=0.02, random_state=42,
)
P = adata_map.X
adata_ge = sp.project_genes(adata_map, ad_sc.copy())
```

图系数只展示语法，不是最优推荐。相似度阈值越高剪枝越严格。省略`expression_genes`时读`uns['training_genes']`；如之后改用更小的`cv_train_genes`，必须按那个实际面板重新剪枝。

`pp_adatas()`不自动构图，和官方一样。图函数独立使用欧氏KNN，取双向边并集后按训练表达余弦剪边，距离同步更新，孤立节点拉普拉斯为零。构图只支持KNN及表达剪枝；mapping只读准备好的图，不二次剪枝。

## 唯一新增损失

```text
P = softmax(M, dim=1)
G_pred = P.T @ S
loss = Tangram_loss + lambda_graph_laplacian * (n_spots/n_cells) * Tr(P L P.T)
```

L为对称归一化拉普拉斯。保留已有图项的n_spots/n_cells量级约定；表达预测不另乘尺度、不取log1p。用稀疏图乘法避免cell×cell中间矩阵。`lambda_graph_laplacian=0`无需图，严格走官方训练路径。graph支持cells/clusters；constrained仅支持官方原路径。

默认参数遵循Tangram：`lambda_g2=0`、`lambda_r=0`。提供密度先验且`lambda_d=0`时入口将其改为1。训练历史恢复官方字符串列表格式；启用图项时只新增`graph_laplacian_reg`。官方日志对关闭项反除0可能显示nan，不应将统计nan等同于总损失异常。

## 差异边界

唯一算法扩展是上述图项。映射入口另保留必要输入兼容修复：dense参考不调用`.toarray()`；密度字符串比较限定str类型；pandas密度转换成ndarray。均不改损失，对照基准施加相同输入修复，原始下载文件哈希不改。

已移除非官方的L1/L2、邻域表达、cell-type islands、Moran/Geary/Getis-Ord、Ray调参、cell_type_mapping、mapping_config、跨平台校正、注意力、去噪及两阶段扩展。`preserve_spatial_graph`、`graph_cell_type_aware`、`spatial_cluster_key`、`graph_sim_threshold`、表达层选择及radius/标签剪枝接口已移除。

为遵循官方，未保留之前的logsumexp密度改写或熵稳定化；极端softmax下沿用官方数值行为。原Notebook和H5AD不修改，重启内核并重跑预处理、构图。项目`tangram/`旧目录未改写，不用于官方对照。

## 验证

```bash
uv run --no-project --python 'D:/python_program/tangram_reproduction/.venv/Scripts/python.exe' python -m pytest -q
```

包含官方来源哈希、逐行源码、CPU/CUDA三种模式的零图映射/历史/投影一致性、dense和密度兼容、图项损失与梯度、剪枝支持集、训练面板和孤立节点校验。测试通过不代表生物学映射准确率提高。源码恢复前快照位于`D:/python_program/scPRISM_backups/`。
