from __future__ import annotations

import re


def keyword_overlap(pred: str, ref: str) -> float:
    stops = {"the", "a", "an", "is", "are", "to", "of", "in", "and", "or"}

    def kws(t: str) -> set:
        return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", t.lower()) if w not in stops}

    p, r = kws(pred), kws(ref)
    return len(p & r) / max(len(r), 1)


def rouge_l(pred: str, ref: str) -> float:
    def lcs(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    p_tokens = pred.lower().split()
    r_tokens = ref.lower().split()
    if not p_tokens or not r_tokens:
        return 0.0
    lcs_len = lcs(p_tokens, r_tokens)
    precision = lcs_len / len(p_tokens)
    recall = lcs_len / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bert_score_approx(pred: str, ref: str) -> float:
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
    recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
