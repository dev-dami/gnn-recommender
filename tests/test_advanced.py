import torch
import numpy as np
from scipy import sparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lightgcn import LightGCN
from src.models.ultralgcn import UltraGCN
from src.models.dropout import NodeDropout, EdgeDropout, DropEdge
from src.models.self_supervised import SelfSupervisedLoss, GraphAugmentor
from src.data.graph import build_adjacency_matrix, symmetric_normalization
from src.losses.bpr import BPRLoss
from src.explain.explainer import GNNExplainer, PathExplainer
from src.knowledge.kg import KnowledgeGraph
from src.training.mixed_precision import MixedPrecisionTrainer
from src.training.distributed import DistributedManager


def _make_test_data():
    num_users, num_items = 50, 30
    mat = sparse.random(num_users, num_items, density=0.15, format="csr")
    norm_adj = symmetric_normalization(build_adjacency_matrix(mat))
    return num_users, num_items, mat, norm_adj


def test_node_dropout():
    nd = NodeDropout(p=0.5)
    x = torch.randn(10, 16)
    nd.train()
    out = nd(x)
    assert out.shape == x.shape
    assert (out == 0).any()
    nd.eval()
    out2 = nd(x)
    assert torch.equal(out2, x)


def test_edge_dropout():
    ed = EdgeDropout(p=0.5)
    coo = sparse.random(20, 15, density=0.3, format="coo")
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    adj = torch.sparse_coo_tensor(indices, values, coo.shape)
    ed.train()
    out = ed(adj)
    assert out.shape == adj.shape
    ed.eval()
    out2 = ed(adj)
    assert torch.equal(out2.to_dense(), adj.to_dense())


def test_drop_edge():
    de = DropEdge(drop_rate=0.5, seed=42)
    mat = sparse.random(50, 30, density=0.2, format="csr")
    dropped = de(mat)
    assert dropped.shape == mat.shape
    assert dropped.nnz < mat.nnz


def test_ultralgcn():
    num_users, num_items, mat, norm_adj = _make_test_data()
    model = UltraGCN(num_users, num_items, 32, n_layers=3)
    user_emb, item_emb = model()
    assert user_emb.shape == (num_users, 32)
    assert item_emb.shape == (num_items, 32)

    norm_t = symmetric_normalization(build_adjacency_matrix(mat))
    coo = norm_t.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    norm_adj_t = torch.sparse_coo_tensor(indices, values, coo.shape)

    kernel_loss = model.compute_kernel_loss(norm_adj_t, user_emb, item_emb)
    assert kernel_loss.item() >= 0


def test_self_supervised_loss():
    ssl = SelfSupervisedLoss(temperature=0.1)
    z1 = torch.randn(32, 16)
    z2 = torch.randn(32, 16)
    loss = ssl.info_nce_loss(z1, z2)
    assert loss.item() > 0

    u_emb = torch.randn(10, 16)
    i_emb = torch.randn(20, 16)
    users = torch.arange(5)
    pos = torch.randint(0, 20, (5,))
    neg = torch.randint(0, 20, (5,))
    combined = ssl.bpr_ssl_loss(u_emb, u_emb, i_emb, i_emb, users, pos, neg)
    assert combined.item() > 0


def test_graph_augmentor():
    mat = sparse.random(50, 30, density=0.15, format="csr")
    aug = GraphAugmentor(mat, seed=42)
    v1 = aug.edge_perturbation(drop_rate=0.1, add_rate=0.05)
    assert v1.shape == mat.shape
    v2 = aug.node_dropout(drop_rate=0.1)
    assert v2.shape == mat.shape
    v3 = aug.generate_view("edge_perturbation")
    assert v3.shape == mat.shape


def test_knowledge_graph():
    kg = KnowledgeGraph(100)
    kg.add_item_item_edges(0, 1, "genre")
    kg.add_item_item_edges(1, 2, "genre")
    assert kg.num_edges == 2
    assert kg.get_item_neighbors(0) == {"genre": [1]}

    genre_matrix = np.random.randint(0, 2, (100, 5))
    kg2 = KnowledgeGraph(100)
    kg2.build_from_genre_matrix(genre_matrix, threshold=0.1)
    assert kg2.num_edges > 0

    mat = sparse.random(50, 100, density=0.1, format="csr")
    aug_adj = kg2.get_augmented_adjacency(build_adjacency_matrix(mat), kg_weight=0.3)
    assert aug_adj.shape[0] == 50 + 100


def test_gnn_explainer():
    num_users, num_items, mat, norm_adj = _make_test_data()
    model = LightGCN(num_users, num_items, 16, 2, norm_adj)
    user_emb, item_emb = model()
    user_emb = user_emb.detach().requires_grad_(True)
    item_emb = item_emb.detach().requires_grad_(True)
    coo = norm_adj.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    norm_adj_t = torch.sparse_coo_tensor(indices, values, coo.shape)

    explainer = GNNExplainer(model, norm_adj_t)
    neighbors = explainer.feature_importance(user_emb, item_emb, 0, 5, mat, top_k=5)
    assert len(neighbors) <= 5

    attn_items, attn_scores = explainer.attention_scores(user_emb, item_emb, 0, top_k=5)
    assert len(attn_items) == 5
    assert len(attn_scores) == 5

    info = explainer.embedding_similarity(user_emb, item_emb, 0, 5, mat)
    assert "score" in info
    assert "user_norm" in info


def test_path_explainer():
    num_users, num_items, mat, norm_adj = _make_test_data()
    pe = PathExplainer(norm_adj)
    explanation = pe.explain(0, 5, mat)
    assert "user" in explanation
    assert "item" in explanation
    assert "paths" in explanation
    assert "n_paths" in explanation


def test_mixed_precision():
    mp = MixedPrecisionTrainer(enabled=False, device="cpu")
    assert not mp.enabled
    ctx = mp.get_context()
    assert ctx is not None


def test_distributed_manager():
    dm = DistributedManager()
    assert dm.rank == 0
    assert dm.world_size == 1
    assert dm.is_main()
    status = dm.status
    assert "gpu_count" in status


def test_ultralgcn_predict():
    num_users, num_items, mat, norm_adj = _make_test_data()
    model = UltraGCN(num_users, num_items, 32, n_layers=2)
    user_emb, item_emb = model()
    users = torch.LongTensor([0, 1, 2])
    items = torch.LongTensor([0, 1, 2])
    scores = model.predict_scores(user_emb, item_emb, users, items)
    assert scores.shape == (3,)


if __name__ == "__main__":
    test_node_dropout()
    print("PASS: node dropout")
    test_edge_dropout()
    print("PASS: edge dropout")
    test_drop_edge()
    print("PASS: drop edge")
    test_ultralgcn()
    print("PASS: ultralgcn")
    test_self_supervised_loss()
    print("PASS: self-supervised loss")
    test_graph_augmentor()
    print("PASS: graph augmentor")
    test_knowledge_graph()
    print("PASS: knowledge graph")
    test_gnn_explainer()
    print("PASS: gnn explainer")
    test_path_explainer()
    print("PASS: path explainer")
    test_mixed_precision()
    print("PASS: mixed precision")
    test_distributed_manager()
    print("PASS: distributed manager")
    test_ultralgcn_predict()
    print("PASS: ultralgcn predict")
    print("\nAll advanced tests passed!")
