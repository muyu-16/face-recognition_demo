# -*- coding: utf-8 -*-
"""
similarity.py
=============
相似度计算与阈值判定模块。

为什么用余弦相似度？
    两张人脸的特征向量都是 512 维空间中的点。余弦相似度衡量的是"方向是否一致"，
    取值范围 [-1, 1]，越接近 1 越像。它对向量长度不敏感，
    所以即使两张照片亮度/对比度不同导致特征模长变化，也不会影响判断。

    如果特征已经做了 L2 归一化（本项目默认会做），那么：
        余弦相似度 == 两个单位向量的点积
    于是"跟人脸库里所有人比对"就退化成一次矩阵乘法，速度非常快。

阈值（threshold）怎么用？
    取出该查询人脸与人脸库中每个人的最高相似度 s：
        s >= threshold  -> 判为"已知人员"，输出对应姓名
        s <  threshold  -> 判为"陌生人"
    阈值是可调参数（默认 0.6）：
        调高 -> 更严格，别人不容易冒充，但本人可能被误判成陌生人（漏识）
        调低 -> 更宽松，本人容易通过，但容易把别人认成本人（误识）
    现场演示时可以拖动滑块，直观展示"误识率 / 漏识率"的权衡。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 默认判定阈值：低于它就认为是陌生人
DEFAULT_THRESHOLD = 0.6

# 阈值的合理范围，界面滑块可以直接用
THRESHOLD_RANGE = (0.0, 1.0)

UNKNOWN_LABEL = "陌生人"


# ---------------------------------------------------------------------------
# 基础向量运算
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """L2 归一化：把向量缩放到长度 1，避免零向量除零。"""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """计算两个向量（或两组对应向量）的余弦相似度，返回 Python float。"""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"两个特征维度不一致：{a.shape} vs {b.shape}，"
                         f"请确认是否更换过骨干网络或重新注册过人脸库。")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """一个查询向量 vs 整个人脸库矩阵。

    query   : (D,)      单张人脸特征
    gallery : (N, D)    人脸库里 N 条特征
    返回    : (N,)      每条特征的余弦相似度
    实现上用 矩阵乘法 + 归一化，等价于逐条算余弦相似度，但快得多。
    """
    q = np.asarray(query, dtype=np.float32).ravel()
    g = np.asarray(gallery, dtype=np.float32)
    if g.ndim == 1:                       # 允许传入单条特征
        g = g.reshape(1, -1)
    if g.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if g.shape[1] != q.shape[0]:
        raise ValueError(
            f"特征维度不匹配：查询 {q.shape[0]} 维，人脸库 {g.shape[1]} 维。"
            f"通常是更换了骨干网络（InceptionResnetV1 / ResNet18）但没有重新注册人脸。"
        )
    q_norm = q / max(float(np.linalg.norm(q)), 1e-10)
    g_norm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-10)
    return (g_norm @ q_norm).astype(np.float32)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """欧氏距离（补充工具）。特征做过 L2 归一化后满足：
       距离² = 2 - 2 * 余弦相似度
    所以两者是单调等价的，本 Demo 主用余弦相似度。"""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# 识别结果
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    """一次识别的完整结果（界面直接展示这个对象）。"""

    is_known: bool                       # 是否为已知人员
    name: str                            # 命中姓名；陌生人为 "陌生人"
    similarity: float                    # 最高相似度
    threshold: float                     # 本次使用的阈值
    top_k: List[Tuple[str, float]] = field(default_factory=list)  # [(姓名, 相似度)] 降序
    all_scores: Optional[np.ndarray] = None   # 与每个人脸库条目（逐条）的相似度
    matched_index: int = -1               # 命中的库内条目下标（-1 表示无）

    @property
    def margin(self) -> float:
        """第一名与第二名的分差，越大说明越可靠（可用于二次判断）。"""
        if len(self.top_k) < 2:
            return float("nan")
        return float(self.top_k[0][1] - self.top_k[1][1])

    def as_dict(self) -> Dict[str, object]:
        return {
            "是否已知": self.is_known,
            "识别结果": self.name,
            "最高相似度": round(self.similarity, 4),
            "阈值": round(self.threshold, 4),
            "判定": "通过" if self.is_known else "低于阈值 -> 陌生人",
            "Top-K": [(n, round(s, 4)) for n, s in self.top_k],
        }

    def __repr__(self) -> str:
        return (f"MatchResult(name='{self.name}', similarity={self.similarity:.4f}, "
                f"threshold={self.threshold:.2f}, is_known={self.is_known})")


# ---------------------------------------------------------------------------
# 识别主流程
# ---------------------------------------------------------------------------
def identify(
    query_embedding: np.ndarray,
    names: Sequence[str],
    gallery: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 5,
    aggregate: str = "max",
    mean_topk: int = 3,
) -> MatchResult:
    """把一张人脸特征与人脸库比对，返回判定结果。

    参数：
        query_embedding : (D,) 查询特征
        names           : 长度 N，gallery 每行对应的姓名（同一个人的多张图姓名重复）
        gallery         : (N, D) 人脸库特征矩阵
        threshold       : 判定阈值，默认 0.6
        top_k           : 返回前 K 名用于展示
        aggregate       : 同一个人的多张注册图如何汇总
                          'max'        -> 取最大相似度（默认，推荐）
                          'mean_topk'  -> 取该人前 mean_topk 张的平均，抗单张噪声
    返回：MatchResult
    """
    names = list(names)
    scores = cosine_similarity_matrix(query_embedding, gallery)

    # 人脸库为空：直接判为陌生人，避免后续 max() 崩溃
    if scores.size == 0 or len(names) == 0:
        return MatchResult(is_known=False, name=UNKNOWN_LABEL, similarity=0.0,
                           threshold=float(threshold), top_k=[],
                           all_scores=scores, matched_index=-1)

    # ---- 按人汇总：同一个人可能注册了多张图，这里合并成一个人的得分 ----
    per_person: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        per_person.setdefault(name, []).append(idx)

    person_scores: List[Tuple[str, float, int]] = []   # (姓名, 分数, 最佳条目下标)
    for name, idxs in per_person.items():
        person_sim = [float(scores[i]) for i in idxs]
        if aggregate == "mean_topk":
            k = max(1, min(int(mean_topk), len(person_sim)))
            top = sorted(person_sim, reverse=True)[:k]
            score = float(np.mean(top))
        else:
            score = float(max(person_sim))
        best_idx = idxs[int(np.argmax([scores[i] for i in idxs]))]
        person_scores.append((name, score, best_idx))

    person_scores.sort(key=lambda item: item[1], reverse=True)
    best_name, best_score, best_idx = person_scores[0]

    is_known = bool(best_score >= float(threshold))
    return MatchResult(
        is_known=is_known,
        name=best_name if is_known else UNKNOWN_LABEL,
        similarity=best_score,
        threshold=float(threshold),
        top_k=[(n, float(s)) for n, s, _ in person_scores[:max(1, int(top_k))]],
        all_scores=scores,
        matched_index=best_idx if is_known else -1,
    )


def is_same_person(similarity: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """单次比较的判定封装：相似度 >= 阈值 即认为是同一个人。"""
    return bool(float(similarity) >= float(threshold))


def verify_pair(embedding_a: np.ndarray, embedding_b: np.ndarray,
                threshold: float = DEFAULT_THRESHOLD, names: Tuple[str, str] = ("A", "B")) -> dict:
    """1:1 比对（人脸验证），常用于"是不是同一个人"的演示。"""
    sim = cosine_similarity(embedding_a, embedding_b)
    return {
        "比对": f"{names[0]} vs {names[1]}",
        "余弦相似度": round(sim, 4),
        "欧氏距离": round(euclidean_distance(embedding_a, embedding_b), 4),
        "阈值": threshold,
        "结论": "同一个人" if is_same_person(sim, threshold) else "不是同一个人",
    }


# ---------------------------------------------------------------------------
# 阈值评估工具（写实验报告 / 答辩加分项）
# ---------------------------------------------------------------------------
def similarity_stats(embeddings: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    """统计类内（同一人）与类间（不同人）相似度分布，用于说明阈值怎么选。

    返回：类内均值/最小/最大、类间均值/最大、以及二者之间的"最佳分割点"建议值。
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    names = list(names)
    intra, inter = [], []
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            sim = cosine_similarity(emb[i], emb[j])
            (intra if names[i] == names[j] else inter).append(sim)

    def _stat(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"数量": 0, "均值": float("nan"), "最小": float("nan"), "最大": float("nan")}
        return {"数量": len(values), "均值": float(np.mean(values)),
                "最小": float(np.min(values)), "最大": float(np.max(values))}

    intra_stat, inter_stat = _stat(intra), _stat(inter)
    suggestion = float("nan")
    if intra and inter:
        # 简单策略：取"类内最小值"与"类间最大值"的中点作为建议阈值
        suggestion = float((intra_stat["最小"] + inter_stat["最大"]) / 2.0)
    return {"类内相似度": intra_stat, "类间相似度": inter_stat, "建议阈值": suggestion}


def find_best_threshold(
    embeddings: np.ndarray,
    names: Sequence[str],
    steps: int = 199,
) -> Dict[str, float]:
    """在 [0,1] 上扫描阈值，按准确率给出最佳阈值（用于报告中画"阈值-准确率"曲线）。"""
    emb = np.asarray(embeddings, dtype=np.float32)
    names = list(names)
    if len(emb) < 2:
        return {"最佳阈值": DEFAULT_THRESHOLD, "最佳准确率": float("nan")}

    pairs = []
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            pairs.append((cosine_similarity(emb[i], emb[j]), names[i] == names[j]))
    if not pairs:
        return {"最佳阈值": DEFAULT_THRESHOLD, "最佳准确率": float("nan")}

    best_t, best_acc = DEFAULT_THRESHOLD, -1.0
    for t in np.linspace(0.0, 1.0, int(steps)):
        correct = sum(1 for sim, same in pairs if (sim >= t) == same)
        acc = correct / len(pairs)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return {"最佳阈值": float(best_t), "最佳准确率": float(best_acc)}


# ---------------------------------------------------------------------------
# 模块自测：python similarity.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 1) 余弦相似度的基本性质：自己跟自己比 = 1，正交向量 = 0
    v = rng.normal(size=512).astype(np.float32)
    print("自相似度（应为 1.0）：", round(cosine_similarity(v, v), 6))

    a = np.zeros(512, dtype=np.float32); a[0] = 1.0
    b = np.zeros(512, dtype=np.float32); b[1] = 1.0
    print("正交向量相似度（应为 0.0）：", round(cosine_similarity(a, b), 6))

    # 2) 构造一个 3 人的人脸库：每人 3 条特征，同人加小噪声
    names, feats = [], []
    for person in ["张三", "李四", "王五"]:
        base = rng.normal(size=512).astype(np.float32)
        base /= np.linalg.norm(base)
        for _ in range(3):
            noisy = base + 0.15 * rng.normal(size=512).astype(np.float32)
            names.append(person)
            feats.append(l2_normalize(noisy))
    gallery = np.stack(feats)

    # 3) 用张三的一条在库特征去查询，应当命中张三
    result = identify(gallery[0], names, gallery, threshold=0.6, top_k=3)
    print("识别结果：", result)
    print("详细：", result.as_dict())

    # 4) 陌生人查询（随机向量）应当低于阈值
    stranger = l2_normalize(rng.normal(size=512).astype(np.float32))
    print("陌生人结果：", identify(stranger, names, gallery, threshold=0.6))

    # 5) 阈值评估工具
    print("相似度统计：", similarity_stats(gallery, names))
    print("最佳阈值扫描：", find_best_threshold(gallery, names))
