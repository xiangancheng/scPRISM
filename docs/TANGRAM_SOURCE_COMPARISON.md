# scPRISM 与官方 Tangram 1.0.4 的源码对照

基准：broadinstitute/Tangram，tag 1.0.4，commit `d537ff936a1a26445a0f196c714d100d10e3e5ea`。下载来源与SHA256见 `tests/reference/tangram_1_0_4/manifest.json`。项目旧 `tangram/` 目录不是官方对照基准。

## 保持一致的内容

`pp_adatas`、`adata_to_cluster_expression`、`MapperConstrained` 保持官方源码。`utils.py`、`plot_utils.py`源码行一致（仅可能有Windows换行差异）。训练历史恢复官方字符串格式。GV/VG、KL密度、熵、初始化及Adam不改公式。

## 三种必要输入兼容修复

1. Dense单细胞参考直接转换为float32数组，不调用不存在的 `.toarray()`。
2. ndarray密度先验不与字符串直接比较，限定字符串类型。
3. pandas密度Series在传入Torch之前转换为ndarray。

官方数值对照使用同样的输入兼容修复；保存的官方文件自身不变。没有保留原scPRISM的logsumexp密度和熵稳定化重写。

## 唯一算法扩展

欧氏KNN → 对称并集 → 训练表达余弦剪枝 → 对称归一化L → 额外损失 `lambda_graph_laplacian * n_spots/n_cells * Tr(P L P.T)`。表达、坐标和图不在mapping中改写。图项只支持cells/clusters；constrained保留原版。

## 逐行差异

### mapping_optimizer.py

```diff
--- official/mapping_optimizer.py
+++ scprism/mapping_optimizer.py
@@ -31,6 +31,8 @@
         device="cpu",
         adata_map=None,
         random_state=None,
+        lambda_graph_laplacian=0,
+        spatial_graph_laplacian=None,
     ):
         """
         Instantiate the Tangram optimizer (without filtering).
@@ -52,6 +54,8 @@
             device (str or torch.device): Optional. Device is 'cpu'.
             adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
             random_state (int): Optional. pass an int to reproduce training. Default is None.
+            lambda_graph_laplacian (float): Strength of the only scPRISM loss extension.
+            spatial_graph_laplacian (array or sparse matrix): Prepared location Laplacian.
         """
         self.S = torch.tensor(S, device=device, dtype=torch.float32)
         self.G = torch.tensor(G, device=device, dtype=torch.float32)
@@ -84,6 +88,22 @@
             self.M, device=device, requires_grad=True, dtype=torch.float32
         )
 
+        # scPRISM: prepare the fixed Laplacian only when its loss is enabled.
+        if not np.isfinite(lambda_graph_laplacian) or lambda_graph_laplacian < 0:
+            raise ValueError('lambda_graph_laplacian must be finite and nonnegative.')
+        self.lambda_graph_laplacian = lambda_graph_laplacian
+        self.spatial_graph_laplacian = None
+        if lambda_graph_laplacian > 0:
+            if spatial_graph_laplacian is None:
+                raise ValueError('A prepared spatial_graph_laplacian is required.')
+            from scipy.sparse import coo_matrix
+            L = coo_matrix(spatial_graph_laplacian, dtype=np.float32)
+            if L.shape != (G.shape[0], G.shape[0]) or not np.isfinite(L.data).all():
+                raise ValueError('Laplacian must be finite and match the spatial observations.')
+            self.spatial_graph_laplacian = torch.sparse_coo_tensor(
+                np.vstack([L.row, L.col]), L.data, L.shape,
+                device=device, dtype=torch.float32).coalesce()
+
     def _loss_fn(self, verbose=True):
         """
         Evaluates the loss function.
@@ -129,10 +149,21 @@
 
         entropy_reg = (regularizer_term / self.lambda_r).tolist()
 
+        # scPRISM: Tr(P L P.T), evaluated without a cell-by-cell matrix.
+        graph_term = None
+        if self.lambda_graph_laplacian > 0:
+            LP = torch.sparse.mm(self.spatial_graph_laplacian, M_probs.t()).t()
+            graph_energy = (M_probs * LP).sum() * self.G.shape[0] / self.S.shape[0]
+            graph_term = self.lambda_graph_laplacian * graph_energy
+            graph_reg = graph_energy.detach().item()
+
         if verbose:
 
             term_numbers = [main_loss, vg_reg, kl_reg, entropy_reg]
             term_names = ["Score", "VG reg", "KL reg", "Entropy reg"]
+            if graph_term is not None:
+                term_numbers.append(graph_reg)
+                term_names.append("Graph Laplacian reg")
 
             d = dict(zip(term_names, term_numbers))
             clean_dict = {k: d[k] for k in d if not np.isnan(d[k])}
@@ -147,6 +178,9 @@
         if density_term is not None:
             total_loss = total_loss + density_term
 
+        if graph_term is not None:
+            total_loss = total_loss + graph_term
+            return total_loss, main_loss, vg_reg, kl_reg, entropy_reg, graph_reg
         return total_loss, main_loss, vg_reg, kl_reg, entropy_reg
 
     def train(self, num_epochs, learning_rate=0.1, print_each=100):
@@ -170,6 +204,8 @@
             logging.info(f"Printing scores every {print_each} epochs.")
 
         keys = ["total_loss", "main_loss", "vg_reg", "kl_reg", "entropy_reg"]
+        if self.lambda_graph_laplacian > 0:
+            keys.append("graph_laplacian_reg")
         values = [[] for i in range(len(keys))]
         training_history = {key: value for key, value in zip(keys, values)}
         for t in range(num_epochs):

```

### mapping_utils.py

```diff
--- official/mapping_utils.py
+++ scprism/mapping_utils.py
@@ -151,6 +151,7 @@
     random_state=None,
     verbose=True,
     density_prior='rna_count_based',
+    lambda_graph_laplacian=0,
 ):
     """
     Map single cell data (`adata_sc`) on spatial data (`adata_sp`).
@@ -172,6 +173,8 @@
         lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
         lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
         target_count (int): Optional. The number of cells to be filtered. Default is None.
+        lambda_graph_laplacian (float): scPRISM graph loss coefficient. Zero
+            exactly follows the Tangram training path and needs no spatial graph.
         random_state (int): Optional. pass an int to reproduce training. Default is None.
         verbose (bool): Optional. If print training details. Default is True.
         density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.
@@ -180,6 +183,12 @@
         a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
         The `uns` field of the returned AnnData contains the training genes.
     """
+
+    # scPRISM: the sole additional mapping objective consumes a prepared graph.
+    if not np.isfinite(lambda_graph_laplacian) or lambda_graph_laplacian < 0:
+        raise ValueError('lambda_graph_laplacian must be finite and nonnegative.')
+    if mode == 'constrained' and lambda_graph_laplacian > 0:
+        raise ValueError('Graph Laplacian is supported in cells/clusters, not constrained mode.')
 
     # check invalid values for arguments
     if lambda_g1 == 0:
@@ -232,13 +241,21 @@
                 "Given training genes list should be subset of two AnnDatas."
             )
 
+    L = None
+    if lambda_graph_laplacian > 0:
+        from .spatial_weights import spatial_graph_laplacian
+        panel = adata_sp.uns.get('scprism_graph_pruning', {}).get('expression_genes')
+        if panel is not None and set(panel) != set(training_genes):
+            raise ValueError('Rebuild expression pruning with the mapping training_genes before mapping.')
+        L = spatial_graph_laplacian(adata_sp, return_sparse=True)
+
     logging.info("Allocate tensors for mapping.")
     # Allocate tensors (AnnData matrix can be sparse or not)
 
     if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
         S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
     elif isinstance(adata_sc.X, np.ndarray):
-        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
+        S = np.array(adata_sc[:, training_genes].X, dtype="float32",)
     else:
         X_type = type(adata_sc.X)
         logging.error("AnnData X has unrecognized type: {}".format(X_type))
@@ -263,11 +280,11 @@
     if type(density_prior) is np.ndarray:
         d_str = "customized"
 
-    if density_prior == "rna_count_based":
+    if isinstance(density_prior, str) and density_prior == "rna_count_based":
         density_prior = adata_sp.obs["rna_count_based_density"]
 
     # define density_prior if 'uniform' is passed to the density_prior argument:
-    elif density_prior == "uniform":
+    elif isinstance(density_prior, str) and density_prior == "uniform":
         density_prior = adata_sp.obs["uniform_density"]
 
     if mode == "cells":
@@ -285,6 +302,10 @@
         if lambda_d is None or lambda_d == 0:
             lambda_d = 1
 
+    # Input compatibility: PyTorch requires an array rather than a pandas Series.
+    if d is not None:
+        d = np.asarray(d)
+
     # Choose device
     device = torch.device(device)  # for gpu
 
@@ -308,7 +329,9 @@
             )
         )
         mapper = mo.Mapper(
-            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
+            S=S, G=G, d=d, device=device, random_state=random_state,
+            lambda_graph_laplacian=lambda_graph_laplacian,
+            spatial_graph_laplacian=L, **hyperparameters,
         )
 
         # TODO `train` should return the loss function
@@ -380,4 +403,4 @@
 
     adata_map.uns["training_history"] = training_history
 
-    return adata_map
+    return adata_map
```

## 验证范围

自动测试对照CPU/CUDA下cells/clusters/constrained、稀疏/稠密输入和不同密度先验；零图映射、全部原生history与gene投影逐元素一致。graph开启时验证loss和gradient只增加图能量；图剪枝支持集、距离同步、输入保留、孤立节点和面板一致性单独校验。短程测试不支持准确率提升结论。


## 已执行结果

完整回归：31 passed（CPU与CUDA均执行，无CUDA跳过）。警告来自官方旧式SciPy导入和第三方索引转换。

真实数据短程检查沿用 `scPRISM_merfish_test.ipynb` 的文件、Scanpy抽样规则（10000，random_state=42）和原空间坐标。为避免整份参考X/Raw的内存峰值，逐行只读取映射所需的共同面板，保持抽样ID、行顺序和原X数值。不是全转录组投影或完整500步准确率评估。每组3步，官方与scPRISM均使用cuda:0、seed42、uniform密度、lambda_g1=1、lambda_d=1、learning_rate=0.1；graph开启组系数0.5。

```json
{
  "purpose": "real-data shared-panel compatibility smoke, not an accuracy benchmark",
  "torch": "2.8.0+cu128",
  "cuda": "12.8",
  "device": "NVIDIA GeForce RTX 5060 Ti",
  "reference_cells": 10000,
  "spatial_cells": 3780,
  "training_genes": 253,
  "epochs_each": 3,
  "zero_graph_max_absolute_difference": 0.0,
  "training_histories_identical": true,
  "graph_unchanged": true,
  "spatial_expression_unchanged": true,
  "projection_shape": [
    3780,
    253
  ],
  "graph_energy_history": [
    1.5191835165023804,
    1.212453007698059,
    0.9758380651473999
  ],
  "graph_counts": {
    "initial_edges": 13150,
    "final_edges": 7121,
    "removed_edges": 6029
  }
}
```
