# 5_run_did.py
# ------------------------------------------------------------
# 在脚本4生成的did datasets的基础上 运行 did(事件研究 + 2*2did +pre trend checks)
# per-store incremental aggregation -> small regression tables （单个store回归）
# 输入:
#   output/did_parts/store_id=*/did_part_*.parquet
# 
# 输出：
#       - dynamic_effect_sales.csv
#       - pretrend_check_sales.csv
#       - ate_summary_sales.csv
#       - store_level_sales.csv
# 需要注意的是：
#       我在 DID 阶段没有直接跑全量行级回归，而是利用了 event fixed effect 的结构，
#       先在每个事件×时间点上计算 treated–control 的差，再对这个差做事件研究回归。
#       这个做法在数学上与标准 event-study DID 等价，但在工程上更高效、更稳定，
#       也方便做分 store 并行和大规模数据处理。
#
# ------------------------------------------------------------
from __future__ import annotations

import os
import glob
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as e:
    raise RuntimeError(
        "statsmodels is required for run_did. Please install it in your venv:\n"
        "  pip install statsmodels\n"
        f"Original import error: {e}"
    )

# plot is optional
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


'''
    IO
'''
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def list_store_ids(did_dir: str) -> List[str]:
    store_dirs = sorted(glob.glob(os.path.join(did_dir, "store_id=*")))
    return [os.path.basename(p).split("store_id=")[-1] for p in store_dirs]


def list_did_parts_for_store(did_dir: str, store_id: str) -> List[str]:
    pdir = os.path.join(did_dir, f"store_id={store_id}")
    return sorted(glob.glob(os.path.join(pdir, "did_part_*.parquet")))


'''
    因变量的构造
'''
def _compute_y(df: pd.DataFrame, y_mode: str) -> pd.Series:
    if y_mode == "sales":
        return df["sales"].astype(float)
    if y_mode == "log1p_sales":
        return np.log1p(df["sales"].astype(float))
    raise ValueError(f"Unknown y_mode={y_mode}")

'''
event_id | event_time | treat | item_id | sales                     event_id | event_time | treat | sum_y | n
------------------------------------------------                    -----------------------------------------
  101    |    0       |  1    |   A     |  12          ---->        101    |    0       |  1    |  22   |  2   
  101    |    0       |  1    |   B     |  10                       101    |    0       |  0    |  17   |  2
  101    |    0       |  0    |   C     |   9
  101    |    0       |  0    |   D     |   8

其实就是从did的micro level 变成 事件 时间 是否处理过 销量 这样可以用来回归的表
'''
def aggregate_event_time_treat(
    parquet_files: List[str],
    y_mode: str,
    by_direction: bool = False,
) -> pd.DataFrame:
    acc: Optional[pd.DataFrame] = None  # 作为一个stack容器

    # -- 最小所需 cols --
    cols = ["event_id", "event_time", "treat", "sales"]
    # -- 如果需要 那就加上 --
    if by_direction: 
        cols.append("direction")

    for fp in parquet_files:
        # -- 读取数据 --
        df = pd.read_parquet(fp, columns=cols)
        # -- 确保类型 --
        df["event_id"] = df["event_id"].astype(np.int64)
        df["event_time"] = df["event_time"].astype(np.int16)
        df["treat"] = df["treat"].astype(np.int8)

        if by_direction:
            df["direction"] = df["direction"].astype(str)

        # -- 计算因变量 --
        y = _compute_y(df, y_mode)
        # -- 如果分direction的话 我们将复合键加上direction --
        # -- 为什么要弄一个tmp 就是temporal 暂时的意思 不想 在df上 append y
        if by_direction:
            tmp = pd.DataFrame(
                {
                    "direction": df["direction"].values,
                    "event_id": df["event_id"].values,
                    "event_time": df["event_time"].values,
                    "treat": df["treat"].values,
                    "y": y.values,
                }
            )
            # -- agg --
            g = tmp.groupby(["direction", "event_id", "event_time", "treat"], sort=False).agg(
                sum_y=("y", "sum"),
                n=("y", "size"),
            )
        else:
            tmp = pd.DataFrame(
                {
                    "event_id": df["event_id"].values,
                    "event_time": df["event_time"].values,
                    "treat": df["treat"].values,
                    "y": y.values,
                }
            )
            g = tmp.groupby(["event_id", "event_time", "treat"], sort=False).agg(
                sum_y=("y", "sum"),
                n=("y", "size"),
            )

        acc = g if acc is None else acc.add(g, fill_value=0) # 需要注意micro level是以事件为中心的 所以不需要担心会 一个event的数据被两个parquet切开 然后分头agg 造成错误

        del df, tmp, g # 释放缓存

    if acc is None:
        return pd.DataFrame()

    out = acc.reset_index()
    out["sum_y"] = out["sum_y"].astype(float)
    out["n"] = out["n"].astype(np.int64)
    return out

# -- 做第一次差分 --
# -- 组间差分 --
'''
    这里的做法是对agg之后的进行处理 类似与对其进行melt 然后 为了求同一个event 里面 treat 和control组的sale mean 以及 diff
'''
def build_event_level_diff_table(agg: pd.DataFrame, by_direction: bool = False) -> pd.DataFrame:
    '''
        属于treat和control的差分
    '''
    # -- 空就返回空 --
    if agg is None or agg.empty:
        base_cols = ["event_id", "event_time", "treated_mean", "control_mean", "diff", "n_treat", "n_ctrl", "weight"]
        if by_direction:
            base_cols = ["direction"] + base_cols
        return pd.DataFrame(columns=base_cols) 

    # -- 求个mean -- 
    agg = agg.copy()
    agg["mean_y"] = agg["sum_y"] / agg["n"].replace(0, np.nan)
    # -- 它这里索引欸 我感觉melt 一下会更好 --
    # -- 感觉这样索引很慢 --
    if by_direction:
        tr = agg[agg["treat"] == 1][["direction", "event_id", "event_time", "mean_y", "n"]].rename(
            columns={"mean_y": "treated_mean", "n": "n_treat"}
        )
        ct = agg[agg["treat"] == 0][["direction", "event_id", "event_time", "mean_y", "n"]].rename(
            columns={"mean_y": "control_mean", "n": "n_ctrl"}
        )
        m = tr.merge(ct, on=["direction", "event_id", "event_time"], how="inner")
    else:
        tr = agg[agg["treat"] == 1][["event_id", "event_time", "mean_y", "n"]].rename(
            columns={"mean_y": "treated_mean", "n": "n_treat"}
        )
        ct = agg[agg["treat"] == 0][["event_id", "event_time", "mean_y", "n"]].rename(
            columns={"mean_y": "control_mean", "n": "n_ctrl"}
        )
        m = tr.merge(ct, on=["event_id", "event_time"], how="inner")

    m["diff"] = m["treated_mean"] - m["control_mean"] # 组间差值
    # -- 拓展的 不必管它 为了以后的多treat情况进行拓展 --
    m["weight"] = np.minimum(m["n_treat"].values, m["n_ctrl"].values).astype(float)
    m = m[m["weight"] > 0].copy()

    m["event_id"] = m["event_id"].astype(np.int64)
    m["event_time"] = m["event_time"].astype(np.int16)
    if by_direction:
        m["direction"] = m["direction"].astype(str)

    return m


# -- 事件研究型 DID 的“第二步回归” --  
def fit_event_study(diff_df: pd.DataFrame, baseline_time: int = -1) -> Tuple[pd.DataFrame, Dict]:
    # -- 看看第一次差分的表格变不变 -- 
    if diff_df is None or diff_df.empty:
        return pd.DataFrame(), {"ok": False, "reason": "empty_diff_df"}

    times = sorted(diff_df["event_time"].unique().tolist()) # 得到事件范围
    times = [t for t in times if t != baseline_time] # 去掉基准事件
    # -- 构造X变量 -- 
    X = pd.DataFrame({"const": 1.0}, index=diff_df.index)
    for t in times:
        X[f"tau_{t}"] = (diff_df["event_time"].values == t).astype(float) # 构建调价效应

    y = diff_df["diff"].astype(float).values # 因变量
    w = diff_df["weight"].astype(float).values # 权重
    clusters = diff_df["event_id"].values # 聚类
    # -- 进行回归 --
    res = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": clusters})

    rows = []
    for t in times:
        name = f"tau_{t}"
        # -- 不同event time 的数据 --
        rows.append(
            {
                "event_time": int(t),
                "coef": float(res.params.get(name, np.nan)),
                "se": float(res.bse.get(name, np.nan)),
                "t": float(res.tvalues.get(name, np.nan)),
                "p": float(res.pvalues.get(name, np.nan)),
            }
        )
    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

    # -- 画图的参数 --
    raw = (
        diff_df.groupby("event_time", as_index=False)
        .apply(lambda g: pd.Series({
            "raw_mean_diff": np.average(g["diff"].values, weights=g["weight"].values) if len(g) else np.nan, # 把所有事件的 diff 做一个加权平均
            "n_events": g["event_id"].nunique(), # 把所有事件的 diff 做一个加权平均
            "weight_sum": float(g["weight"].sum()),# 权重
        }))
        .reset_index()
    )
    # -- 防bug --
    if "event_time" not in raw.columns and "level_0" in raw.columns:
        raw = raw.rename(columns={"level_0": "event_time"})
    raw["event_time"] = raw["event_time"].astype(int)
    # -- raw 统计量 融合到out -- 
    out = out.merge(raw[["event_time", "raw_mean_diff", "n_events", "weight_sum"]], on="event_time", how="left")

    info = {
        "ok": True,
        "n_obs": int(len(diff_df)),
        "n_events": int(diff_df["event_id"].nunique()),
        "baseline_time": int(baseline_time),
    }
    return out, info

# 把 event study 结果里，调价“之前”的那些系数单独拎出来，用来检查平行趋势
def pretrend_check(event_study_df: pd.DataFrame) -> pd.DataFrame:
    if event_study_df is None or event_study_df.empty:
        return pd.DataFrame(columns=["event_time", "coef", "se", "t", "p"])
    return event_study_df[event_study_df["event_time"] < 0].sort_values("event_time").reset_index(drop=True)


def compute_2x2_ate(diff_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Event-level ATE: (post mean diff) - (pre mean diff), then intercept-only WLS with cluster SE.
    """
    # -- 确认 --
    if diff_df is None or diff_df.empty:
        return pd.DataFrame(), {"ok": False, "reason": "empty_diff_df"}
    # -- 切分 调价发生前后 
    pre_mask = diff_df["event_time"] < 0
    post_mask = diff_df["event_time"] >= 0

    # -- 加权均值 --
    def wavg(v, w):
        ww = np.asarray(w, dtype=float)
        vv = np.asarray(v, dtype=float)
        s = ww.sum()
        return float((vv * ww).sum() / s) if s > 0 else np.nan

    rows = []
    # -- 对每个事件 event_id 先算一个“事件级 ATE” -- 
    for eid, g in diff_df.groupby("event_id", sort=False):
        pre = g[pre_mask.loc[g.index]]
        post = g[post_mask.loc[g.index]]
        pre_mean = wavg(pre["diff"].values, pre["weight"].values) if len(pre) else np.nan
        post_mean = wavg(post["diff"].values, post["weight"].values) if len(post) else np.nan
        w_event = float(min(pre["weight"].sum(), post["weight"].sum())) # 权重
        ate_event = (post_mean - pre_mean) if (np.isfinite(pre_mean) and np.isfinite(post_mean)) else np.nan # ATE 两次差分
        rows.append({"event_id": int(eid), "ate_event": ate_event, "weight_event": w_event}) # 将事件级别ATE添加到rows

    # -- 筛选和保留 --
    ev = pd.DataFrame(rows)
    ev = ev[np.isfinite(ev["ate_event"]) & (ev["weight_event"] > 0)].copy()
    if ev.empty:
        return pd.DataFrame(), {"ok": False, "reason": "no_valid_events_for_ate"}

    X = pd.DataFrame({"const": 1.0}, index=ev.index)
    y = ev["ate_event"].astype(float).values
    w = ev["weight_event"].astype(float).values
    clusters = ev["event_id"].values
    # -- 再把所有事件的 ATE 做一个“加权平均 + 有标准误 --
    res = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    # 总结一下信息
    summary = pd.DataFrame([{
        "ate": float(res.params["const"]),
        "se": float(res.bse["const"]),
        "t": float(res.tvalues["const"]),
        "p": float(res.pvalues["const"]),
        "n_events": int(ev["event_id"].nunique()),
    }])

    return summary, {"ok": True, "n_events": int(ev["event_id"].nunique())}


'''
    plot
'''
def save_event_study_plot(df: pd.DataFrame, out_png: str, title: str) -> None:
    if plt is None:
        print("[WARN] matplotlib not available, skip plotting.")
        return
    if df is None or df.empty:
        print(f"[WARN] empty df for plot: {out_png}")
        return

    d = df.sort_values("event_time").copy()
    x = d["event_time"].values.astype(int)
    y = d["coef"].values.astype(float)
    se = d["se"].values.astype(float)

    # 95% CI
    lo = y - 1.96 * se
    hi = y + 1.96 * se

    ensure_dir(os.path.dirname(out_png))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(0.0)
    ax.axvline(0.0)
    ax.plot(x, y, marker="o")
    ax.fill_between(x, lo, hi, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("event_time (weeks)")
    ax.set_ylabel("effect on y (coef)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


'''
    对某一个门店，把 DID 数据读进来 → 做 event study → 做 pre-trend → 算 2×2 ATE，
    是否按涨/降价拆开，由 by_direction 决定。
'''
def run_for_store_one_group(
    did_dir: str,
    store_id: str,
    y_mode: str,
    baseline_time: int,
    by_direction: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    # -- 找数据 --
    files = list_did_parts_for_store(did_dir, store_id)
    if not files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"ok": False, "reason": "no_files"}
    # 聚合 + 做 treated−control 差分 （第一步差分）
    agg = aggregate_event_time_treat(files, y_mode=y_mode, by_direction=by_direction)
    diff = build_event_level_diff_table(agg, by_direction=by_direction)

    # -- 是否分方向 不分就跑一整套  --
    if not by_direction:
        es, es_info = fit_event_study(diff, baseline_time=baseline_time)
        pre = pretrend_check(es) # 平行趋势 
        ate, ate_info = compute_2x2_ate(diff) # 

        info = {
            "ok": True,
            "store_id": store_id,
            "n_files": len(files),
            "n_events": int(diff["event_id"].nunique()) if not diff.empty else 0,
            "n_obs_event_time": int(len(diff)),
            "event_study": es_info,
            "ate": ate_info,
        }
        if not es.empty:
            es.insert(0, "store_id", store_id)
        if not pre.empty:
            pre.insert(0, "store_id", store_id)
        if not ate.empty:
            ate.insert(0, "store_id", store_id)
        return es, pre, ate, info

    
    out_es = []
    out_pre = []
    out_ate = []
    infos = []

    # -- 分就拆成多组跑 --
    for direction, g in diff.groupby("direction", sort=False):
        es, es_info = fit_event_study(g, baseline_time=baseline_time) # 跑完得到动态效应 以及相关信息
        pre = pretrend_check(es)
        ate, ate_info = compute_2x2_ate(g)

        if not es.empty:
            es.insert(0, "store_id", store_id)
            es.insert(1, "direction", direction)
            out_es.append(es)
        if not pre.empty:
            pre.insert(0, "store_id", store_id)
            pre.insert(1, "direction", direction)
            out_pre.append(pre)
        if not ate.empty:
            ate.insert(0, "store_id", store_id)
            ate.insert(1, "direction", direction)
            out_ate.append(ate)

        infos.append({
            "store_id": store_id,
            "direction": direction,
            "event_study": es_info,
            "ate": ate_info,
            "n_events": int(g["event_id"].nunique()) if not g.empty else 0,
            "n_obs_event_time": int(len(g)),
        })

    # -- 打包结果 + 诊断信息，返回给 main --
    info = {"ok": True, "store_id": store_id, "n_files": len(files), "groups": infos}
    es_df = pd.concat(out_es, ignore_index=True) if out_es else pd.DataFrame()
    pre_df = pd.concat(out_pre, ignore_index=True) if out_pre else pd.DataFrame()
    ate_df = pd.concat(out_ate, ignore_index=True) if out_ate else pd.DataFrame()
    return es_df, pre_df, ate_df, info


def parse_args():
    ap = argparse.ArgumentParser("Run DID regressions (event study + 2x2) on did_parts")
    ap.add_argument("--did_dir", type=str, default="output/did_parts", help="Directory with store_id=*/did_part_*.parquet") # 输入数据的地址
    ap.add_argument("--out_dir", type=str, default="output/did_results", help="Output directory for did results") # 输出结果的地址
    ap.add_argument("--y_mode", type=str, default="log1p_sales", choices=["sales", "log1p_sales"], help="Outcome transform") # 因变量怎么选
    ap.add_argument("--baseline_time", type=int, default=-1, help="Baseline event_time to omit in event study (default -1)") # 
    ap.add_argument("--stores", type=str, default="", help="Comma-separated store_ids, empty=all (e.g. CA_1,CA_2)") # 门店id 用来并行都可以
    ap.add_argument("--by_direction", action="store_true", help="Split results by direction (up/down)") # 是否分direction进行did
    ap.add_argument("--plot", action="store_true", help="Save event-study plots as PNG under out_dir/plots/") # 是否画画
    return ap.parse_args()


def main():
    # -- 命令行参数 --
    args = parse_args()
    ensure_dir(args.out_dir) # 确保可以输出

    # -- 所有门店id的名单 --
    all_stores = list_store_ids(args.did_dir)
    stores = [s.strip() for s in args.stores.split(",") if s.strip()] if args.stores else all_stores

    # -- 基本信息 -- （主要是用来确认）
    print(f"[INFO] did_dir={args.did_dir}")
    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] y_mode={args.y_mode} baseline_time={args.baseline_time}")
    print(f"[INFO] by_direction={args.by_direction} plot={args.plot}")
    print(f"[INFO] stores={stores} (n={len(stores)})")

    es_all = [] # event-study 动态效应
    pre_all = [] # pre-trend 检验
    ate_all = [] # 2x2 ATE
    store_level = [] # 诊断信息

    for i, store_id in enumerate(stores, start=1): # 索引以及门店
        print(f"\n[STORE {i}/{len(stores)}] running: {store_id}") #进度条
        # -- 对该门店跑一个did -- 
        es_df, pre_df, ate_df, info = run_for_store_one_group(
            did_dir=args.did_dir,
            store_id=store_id,
            y_mode=args.y_mode,
            baseline_time=args.baseline_time,
            by_direction=args.by_direction,
        )
        # -- 储存门店信息 --
        store_level.append(info)

        # -- 是否不ok --
        if not info.get("ok"):
            print(f"[SKIP] store={store_id} reason={info.get('reason')}")
            continue
        
        # -- 动态效应是否为空 不空就append --
        if not es_df.empty:
            es_all.append(es_df)
        # -- pre trend 检验是否为空 --
        if not pre_df.empty:
            pre_all.append(pre_df)
        # -- 2×2 DID 汇总 是否为空
        if not ate_df.empty:
            ate_all.append(ate_df)

        # -- 画图 --
        # -- 画出 event 的结果 你可以选择是否 是否按照direction分 
        if args.plot and not es_df.empty:
            plot_dir = os.path.join(args.out_dir, "plots") # 地址
            if args.by_direction:
                for direction, g in es_df.groupby("direction", sort=False):
                    png = os.path.join(plot_dir, f"event_study_{store_id}_dir={direction}_{args.y_mode}.png")
                    save_event_study_plot(
                        g,
                        png,
                        title=f"Event Study: store={store_id}, direction={direction}, y={args.y_mode}"
                    )
                    print(f"[PLOT] {png}")
            else:
                png = os.path.join(plot_dir, f"event_study_{store_id}_{args.y_mode}.png")
                save_event_study_plot(
                    es_df,
                    png,
                    title=f"Event Study: store={store_id}, y={args.y_mode}"
                )
                print(f"[PLOT] {png}")

    # 写入输出文件
    # -- 文件名 --
    suffix = f"{args.y_mode}"
    if args.by_direction: # 是否按照direction来
        suffix = f"{args.y_mode}_by_direction"

    if es_all:
        es = pd.concat(es_all, ignore_index=True) # 每个 store 的 event study 结果
        out_path = os.path.join(args.out_dir, f"dynamic_effect_sales_{suffix}.csv")
        es.to_csv(out_path, index=False)
        print(f"[OUT] {out_path} rows={len(es):,}")
    else:
        print("[WARN] no event-study results produced.")

    if pre_all:
        pre = pd.concat(pre_all, ignore_index=True) # 汇总 + 输出 pre-trend 检验结果
        out_path = os.path.join(args.out_dir, f"pretrend_check_sales_{suffix}.csv")
        pre.to_csv(out_path, index=False)
        print(f"[OUT] {out_path} rows={len(pre):,}")
    else:
        print("[WARN] no pretrend results produced.")

    if ate_all: # 汇总 + 输出 ATE（2×2 DID）结果
        ate = pd.concat(ate_all, ignore_index=True)
        out_path = os.path.join(args.out_dir, f"ate_summary_sales_{suffix}.csv")
        ate.to_csv(out_path, index=False)
        print(f"[OUT] {out_path} rows={len(ate):,}")
    else:
        print("[WARN] no ATE summary produced.")

    # -- 输出 store 级诊断信息 --
    diag_path = os.path.join(args.out_dir, f"store_level_sales_{suffix}.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(store_level, f, ensure_ascii=False, indent=2)
    print(f"[OUT] {diag_path}")

    print("\n[DONE] run_did finished.")


if __name__ == "__main__":
    main()

