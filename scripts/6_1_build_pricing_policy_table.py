'''

'''
from __future__ import annotations

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_one(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

# -- 得到持续时间 以及反应峰值 --
def _infer_effect_horizon(es: pd.DataFrame, post_only: bool = True) -> pd.DataFrame:
    """
    每个 store_id的es得到:
      - effect_horizon: 效应有效期
      - peak_time: 影响最强的那一周
    """
    if es is None or es.empty:
        return pd.DataFrame(columns=["store_id", "effect_horizon", "peak_time"])

    df = es.copy()
    # -- 只看 post（调价之后）--
    if post_only:
        df = df[df["event_time"] >= 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["store_id", "effect_horizon", "peak_time"])
    # -- 得到我们要的指标 --
    def _one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("event_time")
        sig = g[g["p"] < 0.05] # 只保留显著的效应
        effect_h = int(sig["event_time"].max()) if len(sig) else -1 # 持续期
        # 得到峰值
        peak_row = g.iloc[(g["coef"].abs()).values.argmax()]
        peak_t = int(peak_row["event_time"])
        return pd.Series({"effect_horizon": effect_h, "peak_time": peak_t})
    # -- 对store 单独算 --
    out = df.groupby("store_id", as_index=False).apply(_one).reset_index(drop=True)
    return out

# -- 把 DID 的统计结果（ATE / ES / pre-trend） 翻译成一个门店级、可执行的定价决策表 --
def _build_overall_table(did_dir: str, y_mode: str) -> pd.DataFrame:
    """
    商店层面的决策表
    使用:
      - ate_summary_sales_{y_mode}.csv
      - dynamic_effect_sales_{y_mode}.csv
      - pretrend_check_sales_{y_mode}.csv
    """
    ate_path = os.path.join(did_dir, f"ate_summary_sales_{y_mode}.csv")
    es_path = os.path.join(did_dir, f"dynamic_effect_sales_{y_mode}.csv")
    pre_path = os.path.join(did_dir, f"pretrend_check_sales_{y_mode}.csv")
    # -- 都是以store 作为前提单位 --
    ate = _read_one(ate_path) # 调价的总体平均影响是多少
    es = _read_one(es_path) # 调价效应随时间如何变化
    pre = _read_one(pre_path) # 验证 ES / ATE 是否可信

    if ate.empty:
        return pd.DataFrame()

    # -- ATE的显著性 storelevel 的 --
    ate = ate.copy()
    ate["ate_sig_05"] = (ate["p"] < 0.05).astype(int)
    ate["ate_sig_10"] = (ate["p"] < 0.10).astype(int)

    # -- 可信度过滤器 --
    pre_pass = pd.DataFrame(columns=["store_id", "pretrend_pass", "pretrend_fail_rate"])
    if not pre.empty:
        tmp = pre.copy()
        tmp["fail"] = (tmp["p"] < 0.05).astype(int)  # 如果在调价前，你就看到显著效应  那这次事件不可信
        # -- 这里对门店进行聚合 你调价前有效应可以 但是不能超 --
        pre_pass = (
            tmp.groupby("store_id", as_index=False)
            .agg(pretrend_fail_rate=("fail", "mean"))
        )
        pre_pass["pretrend_pass"] = (pre_pass["pretrend_fail_rate"] <= 0.2).astype(int)  # 允许一点噪声
    # 
    # -- ES：效果持续多久以及峰值在哪 --
    horizon = _infer_effect_horizon(es)
    # -- 合并成 store-level 表 --
    out = ate.merge(pre_pass, on="store_id", how="left").merge(horizon, on="store_id", how="left")
    out["pretrend_pass"] = out["pretrend_pass"].fillna(0).astype(int)
    out["pretrend_fail_rate"] = out["pretrend_fail_rate"].fillna(np.nan)

    # -- 一个 0–1 的信心分 --
    out["confidence"] = (
        0.0
        + 0.55 * out["ate_sig_05"]
        + 0.25 * out["ate_sig_10"]
        + 0.20 * out["pretrend_pass"]
    ).clip(0, 1)

    # -- 多久再调一次价 --
    out["cooldown_weeks"] = out["effect_horizon"].apply(lambda x: int(x) if pd.notna(x) and x >= 0 else 4)
    out["cooldown_weeks"] = out["cooldown_weeks"].clip(2, 12).astype(int)

    # -- 分级 --
    def _mode(row) -> str:
        if row["pretrend_pass"] == 0:
            return "audit_only" # 人工 或者方向细分
        if row["p"] < 0.05:
            return "normal" #  自动定价
        if row["p"] < 0.10: # 可用 但要保守一点
            return "conservative"
        return "hold_bias" # 基本不变
    
    # -- recommendation_mode：策略开关 --
    out["recommendation_mode"] = out.apply(_mode, axis=1)

    # --调价幅度候选池 --
    out["delta_pct_cut_candidates"] = "[-0.03,-0.05,-0.08]"
    out["delta_pct_raise_candidates"] = "[0.02,0.03]"
    out["scope"] = "overall"
    # -- 保留的cols --
    keep = [
        "store_id",
        "ate", "se", "t", "p", "n_events",
        "pretrend_pass", "pretrend_fail_rate",
        "effect_horizon", "peak_time",
        "confidence", "cooldown_weeks", "recommendation_mode",
        "delta_pct_cut_candidates", "delta_pct_raise_candidates",
        "scope",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep].sort_values("store_id").reset_index(drop=True)
    return out


def _build_direction_table(did_dir: str, y_mode: str) -> pd.DataFrame:
    """
    Direction-specific policy table at (store_id, direction).
    Uses:
      - ate_summary_sales_{y_mode}_by_direction.csv
      - dynamic_effect_sales_{y_mode}_by_direction.csv
      - pretrend_check_sales_{y_mode}_by_direction.csv
    """
    ate_path = os.path.join(did_dir, f"ate_summary_sales_{y_mode}_by_direction.csv")
    es_path = os.path.join(did_dir, f"dynamic_effect_sales_{y_mode}_by_direction.csv")
    pre_path = os.path.join(did_dir, f"pretrend_check_sales_{y_mode}_by_direction.csv")

    ate = _read_one(ate_path)
    es = _read_one(es_path)
    pre = _read_one(pre_path)

    if ate.empty:
        return pd.DataFrame()

    ate = ate.copy()
    ate["ate_sig_05"] = (ate["p"] < 0.05).astype(int)
    ate["ate_sig_10"] = (ate["p"] < 0.10).astype(int)

    pre_pass = pd.DataFrame(columns=["store_id", "direction", "pretrend_pass", "pretrend_fail_rate"])
    if not pre.empty:
        tmp = pre.copy()
        tmp["fail"] = (tmp["p"] < 0.05).astype(int)
        pre_pass = (
            tmp.groupby(["store_id", "direction"], as_index=False)
            .agg(pretrend_fail_rate=("fail", "mean"))
        )
        pre_pass["pretrend_pass"] = (pre_pass["pretrend_fail_rate"] <= 0.2).astype(int)

    # horizon per (store, direction)
    if es is None or es.empty:
        horizon = pd.DataFrame(columns=["store_id", "direction", "effect_horizon", "peak_time"])
    else:
        def _one(g: pd.DataFrame) -> pd.Series:
            g = g[g["event_time"] >= 0].sort_values("event_time")
            sig = g[g["p"] < 0.05]
            effect_h = int(sig["event_time"].max()) if len(sig) else -1
            peak_row = g.iloc[(g["coef"].abs()).values.argmax()]
            return pd.Series({"effect_horizon": effect_h, "peak_time": int(peak_row["event_time"])})
        horizon = es.groupby(["store_id", "direction"], as_index=False).apply(_one).reset_index(drop=True)

    out = ate.merge(pre_pass, on=["store_id", "direction"], how="left").merge(horizon, on=["store_id", "direction"], how="left")
    out["pretrend_pass"] = out["pretrend_pass"].fillna(0).astype(int)
    out["pretrend_fail_rate"] = out["pretrend_fail_rate"].fillna(np.nan)

    out["confidence"] = (
        0.0
        + 0.55 * out["ate_sig_05"]
        + 0.25 * out["ate_sig_10"]
        + 0.20 * out["pretrend_pass"]
    ).clip(0, 1)

    out["cooldown_weeks"] = out["effect_horizon"].apply(lambda x: int(x) if pd.notna(x) and x >= 0 else 4)
    out["cooldown_weeks"] = out["cooldown_weeks"].clip(2, 12).astype(int)

    def _mode(row) -> str:
        if row["pretrend_pass"] == 0:
            return "audit_only"
        if row["p"] < 0.05:
            return "normal"
        if row["p"] < 0.10:
            return "conservative"
        return "hold_bias"

    out["recommendation_mode"] = out.apply(_mode, axis=1)
    out["scope"] = "by_direction"

    keep = [
        "store_id", "direction",
        "ate", "se", "t", "p", "n_events",
        "pretrend_pass", "pretrend_fail_rate",
        "effect_horizon", "peak_time",
        "confidence", "cooldown_weeks", "recommendation_mode",
        "scope",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep].sort_values(["store_id", "direction"]).reset_index(drop=True)
    return out


def parse_args():
    ap = argparse.ArgumentParser("Build pricing policy tables from DID results")
    ap.add_argument("--did_results_dir", type=str, default="output/did_results", help="directory containing did csv outputs")
    ap.add_argument("--out_dir", type=str, default="output/pricing_policy", help="output directory")
    ap.add_argument("--y_mode", type=str, default="log1p_sales", help="same y_mode you used in run_did")
    return ap.parse_args()


def main():
    # -- 命令行参数 --
    args = parse_args()
    ensure_dir(args.out_dir)

    # -- 生成策略表 --
    overall = _build_overall_table(args.did_results_dir, args.y_mode)
    bydir = _build_direction_table(args.did_results_dir, args.y_mode)

    # -- 分别写出 csv --
    if not overall.empty:
        p = os.path.join(args.out_dir, "policy_table_overall.csv")
        overall.to_csv(p, index=False)
        print(f"[OUT] {p} rows={len(overall):,}")
    else:
        print("[WARN] overall table empty (missing overall DID outputs?)")

    if not bydir.empty:
        p = os.path.join(args.out_dir, "policy_table_by_direction.csv")
        bydir.to_csv(p, index=False)
        print(f"[OUT] {p} rows={len(bydir):,}")
    else:
        print("[WARN] direction table empty (missing *_by_direction outputs?)")

    # -- 合并一个方便用的总表 --
    merged = pd.DataFrame()
    if not overall.empty and not bydir.empty:
        merged = pd.concat([overall, bydir], ignore_index=True)
    elif not overall.empty:
        merged = overall.copy()
    elif not bydir.empty:
        merged = bydir.copy()

    if not merged.empty:
        p = os.path.join(args.out_dir, "policy_table_merged.csv")
        merged.to_csv(p, index=False)
        print(f"[OUT] {p} rows={len(merged):,}")

    print("[DONE] build_pricing_policy_table finished.")


if __name__ == "__main__":
    main()
