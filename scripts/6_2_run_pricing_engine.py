# 6_2_run_pricing_engine.py
'''
    读策略表 + 读最新周数据 + 算每个商品近期状态 + 套规则出建议价 + 输出推荐表&诊断
'''
from __future__ import annotations

import os
import glob
import json
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# -- 确保类型 --
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

#  -- 获取 store id list --
def list_store_ids(weekly_store_dir: str) -> List[str]:
    dirs = sorted(glob.glob(os.path.join(weekly_store_dir, "store_id=*")))
    return [os.path.basename(d).split("store_id=")[-1] for d in dirs]

# -- 获得 周级轻表 --
def load_weekly_store(weekly_store_dir: str, store_id: str) -> pd.DataFrame:
    pdir = os.path.join(weekly_store_dir, f"store_id={store_id}")
    parts = sorted(glob.glob(os.path.join(pdir, "week_part_*.parquet")))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    # -- 防御性去重 --
    gcols = ["store_id", "item_id", "wm_yr_wk"]
    df = df.groupby(gcols, as_index=False).agg(
        weekly_sales=("weekly_sales", "sum"),
        weekly_price=("weekly_price", "mean"),
        dept_id=("dept_id", "first"),
        cat_id=("cat_id", "first"),
        state_id=("state_id", "first"),
    )
    df["store_id"] = df["store_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["wm_yr_wk"] = df["wm_yr_wk"].astype(int)
    return df

# -- 加载策略表
def load_policy_tables(policy_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_path = os.path.join(policy_dir, "policy_table_overall.csv")
    bydir_path = os.path.join(policy_dir, "policy_table_by_direction.csv")

    overall = pd.read_csv(overall_path) if os.path.exists(overall_path) else pd.DataFrame()
    bydir = pd.read_csv(bydir_path) if os.path.exists(bydir_path) else pd.DataFrame()

    if not overall.empty:
        overall["store_id"] = overall["store_id"].astype(str)
    if not bydir.empty:
        bydir["store_id"] = bydir["store_id"].astype(str)
        bydir["direction"] = bydir["direction"].astype(str)

    return overall, bydir

# -- 简单线性斜率 --
def simple_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x.astype(float)
    y = y.astype(float)
    xm = x.mean()
    ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / denom)

# -- 决策 --
'''
decide_action 是一个“因果约束下的定价决策器”：
    用 DID 判断“能不能动、往哪动”，
    用近期销量趋势判断“现在该不该动”，
    最后给出一个可解释的价格动作。
'''
def decide_action(
    current_price: float,
    recent_sales_mean: float,
    recent_sales_slope: float,
    overall_policy: Optional[pd.Series],
    bydir_policy: Optional[pd.DataFrame],
) -> Tuple[str, float, str, float]:
    """
    Input：current_price recent_sales_mean recent_sales_slope 以及两个策略biao

    Returns:
      action: "hold" | "cut" | "raise"
      delta_pct: 建议的价格调整（ 如果是hold 就是0 ）
      reason: 简短地解释
      confidence: 对于这次调价的可信度
    """
    # -- 默认决策 --
    action = "hold"
    delta_pct = 0.0
    reason = "no_policy"
    confidence = 0.0

    if overall_policy is None or len(overall_policy) == 0:
        return action, delta_pct, reason, confidence

    confidence = float(overall_policy.get("confidence", 0.0))
    pre_ok = int(overall_policy.get("pretrend_pass", 0))
    pval = float(overall_policy.get("p", 1.0))
    ate = float(overall_policy.get("ate", 0.0))
    mode = str(overall_policy.get("recommendation_mode", "hold_bias"))

    '''
        pretrend 不通过 或 ATE 不显著 → 直接 hold
    '''

    # -- 如果pre trend过不了 只能audit了
    if pre_ok == 0:
        return "hold", 0.0, "pretrend_fail_hold", confidence * 0.5

    # 如果p值不够 那就也不改变价格
    if pval >= 0.10:
        return "hold", 0.0, "ate_not_significant_hold", confidence

    # -- 再看调价方向（可选增强）-- 
    prefer_cut = False
    prefer_raise = False
    if bydir_policy is not None and not bydir_policy.empty:
        # -- 选择更可信的方向 --
        # -- 如果降价更有利于升价 那就是降价 --
        d = bydir_policy.set_index("direction")
        if "down" in d.index and "up" in d.index:
            down_conf = float(d.loc["down"].get("confidence", 0.0))
            up_conf = float(d.loc["up"].get("confidence", 0.0))
            down_ate = float(d.loc["down"].get("ate", 0.0))
            up_ate = float(d.loc["up"].get("ate", 0.0))
            # -- heuristic 启发式 --
            # -- 有两个判断 一个是confidence要够 另一个是系数要够大
            if down_conf >= 0.6 and abs(down_ate) >= abs(up_ate) * 1.1:
                prefer_cut = True
            if up_conf >= 0.6 and abs(up_ate) >= abs(down_ate) * 1.1:
                prefer_raise = True

    # -- 基于最近销量的核心启发 --
    # -- 销量在上涨 -> 只在证据很强时小幅涨价 --
    # -- 销量在下滑 -> 倾向降价 --
    # -- 横盘 -> 更保守 --
    falling = (recent_sales_slope < 0)
    rising = (recent_sales_slope > 0)

    # 调价候选池
    cut_cands = [-0.03, -0.05, -0.08]
    raise_cands = [0.02, 0.03]

    # -- 输出一套决策 --
    if falling: # 如果销量下跌
        if prefer_raise: # 如果推荐升价
            # -- 很稀有的 可能是关于价格和质量相关的信号 但由于是反直觉的 所以要处理地非常谨慎 --
            action, delta_pct = "raise", raise_cands[0]
            reason = "sales_falling_but_up_effect_stronger_small_raise"
        else:
            action, delta_pct = "cut", cut_cands[1] if mode == "normal" else cut_cands[0]
            reason = "sales_falling_cut_to_recover_with_did_support"
    elif rising:
        if prefer_cut:
            action, delta_pct = "hold", 0.0
            reason = "sales_rising_hold_even_if_down_effect_stronger"
        else:
            # -- 同样的 反直觉 只有did证据很明显的时候才涨价 --
            if pval < 0.05: # 这里要求 0.05 要求更严格了
                action, delta_pct = "raise", raise_cands[0]
                reason = "sales_rising_small_raise_with_strong_evidence"
            else:
                action, delta_pct = "hold", 0.0
                reason = "sales_rising_hold"
    else:
        # -- 如何是平的话 那就正常处理 --
        if prefer_cut and pval < 0.05:
            action, delta_pct = "cut", cut_cands[0]
            reason = "flat_sales_cut_small_prefer_down"
        elif prefer_raise and pval < 0.05:
            action, delta_pct = "raise", raise_cands[0]
            reason = "flat_sales_raise_small_prefer_up"
        else:
            action, delta_pct = "hold", 0.0
            reason = "flat_sales_hold"

    return action, float(delta_pct), reason, confidence

# -- 命令行参数 --
def parse_args():
    ap = argparse.ArgumentParser("Run weekly pricing engine using DID policy tables")
    ap.add_argument("--weekly_store_dir", type=str, default="output/weekly_store_parts")
    ap.add_argument("--policy_dir", type=str, default="output/pricing_policy")
    ap.add_argument("--out_dir", type=str, default="output/pricing_reco")
    ap.add_argument("--stores", type=str, default="", help="comma-separated store ids, empty=all")
    ap.add_argument("--lookback_weeks", type=int, default=4, help="use last N weeks to compute mean+trend")
    ap.add_argument("--top_n", type=int, default=200, help="output top-N recommendations per store by score")
    return ap.parse_args()


def main():
    # -- 命令行参数 --
    args = parse_args()
    ensure_dir(args.out_dir) # 确保output类型 不然run完了发现output不出来 

    overall, bydir = load_policy_tables(args.policy_dir) # 加载策略表
    if overall.empty:
        raise FileNotFoundError(f"Missing overall policy table under {args.policy_dir}. Run 6_1 first.")
    
    # -- store list --
    all_stores = list_store_ids(args.weekly_store_dir)
    stores = [s.strip() for s in args.stores.split(",") if s.strip()] if args.stores else all_stores # 去掉空格

    out_rows = [] # output stack 容器
    diag = [] # 诊断
    #  -- 对store迭代 --
    for store_id in stores:
        wk = load_weekly_store(args.weekly_store_dir, store_id) # 周级轻表
        if wk.empty:
            diag.append({"store_id": store_id, "ok": False, "reason": "no_weekly_data"})
            continue

        # -- 策略切分 --
        pol_overall = overall[overall["store_id"] == store_id] # 取出单store的策略
        overall_row = pol_overall.iloc[0] if len(pol_overall) else None # 列名

        pol_bydir = bydir[bydir["store_id"] == store_id] if not bydir.empty else pd.DataFrame() # by direction的

        # -- 把最新周当成本周数据 --
        cur_wk = int(wk["wm_yr_wk"].max()) # 最新周
        wk_cur = wk[wk["wm_yr_wk"] == cur_wk].copy() # 最新周周级数据

        # 这里的假设：wk单调是成立的 --
        # -- 近期窗口 --
        wks_sorted = np.sort(wk["wm_yr_wk"].unique()) # wkid
        recent_wks = wks_sorted[-args.lookback_weeks:] if len(wks_sorted) >= args.lookback_weeks else wks_sorted # 最近的几周
        wk_recent = wk[wk["wm_yr_wk"].isin(recent_wks)].copy() # 获取所有门店 所有商品的周级数据

        # -- 算特征 -> 套策略 -> 得到动作 --
        def _item_stats(g: pd.DataFrame) -> pd.Series:
            g = g.sort_values("wm_yr_wk") # 周级数据
            x = np.arange(len(g), dtype=int) # 构造一个时间轴
            y = g["weekly_sales"].to_numpy(dtype=float) # 销量
            return pd.Series({
                "recent_sales_mean": float(np.mean(y)) if len(y) else 0.0, # 窗口平均销量
                "recent_sales_slope": float(simple_slope(x, y)) if len(y) else 0.0, # 斜率
                "recent_price_mean": float(np.nanmean(g["weekly_price"].to_numpy(dtype=float))) if len(y) else np.nan, # 窗口平均价格
            })

        stats = wk_recent.groupby("item_id", as_index=False).apply(_item_stats).reset_index(drop=True) # 算出 store * item 近期的状态
        cur = wk_cur[["item_id", "weekly_price", "weekly_sales", "dept_id", "cat_id", "state_id"]].copy() # current周级数据 取需要cols
        cur = cur.rename(columns={"weekly_price": "current_price", "weekly_sales": "current_week_sales"}) # rename

        df = cur.merge(stats, on="item_id", how="left") # append上去
        # -- 添加门店和时间 后期stack在一起就可以了 
        df["store_id"] = store_id
        df["wm_yr_wk"] = cur_wk

        # decide actions
        actions = []
        # -- 把每个商品（item）过一遍规则引擎 --
        for r in df.itertuples(index=False):
            item_id = str(r.item_id)
            current_price = float(r.current_price) if np.isfinite(r.current_price) else np.nan # 获取近期价格
            # -- 如果价格确实 或者异常 那就不变
            if not np.isfinite(current_price) or current_price <= 0:
                actions.append(("hold", 0.0, "missing_price", 0.0))
                continue
            # -- 获取另外两个信息就是销量的mean 以及斜率
            recent_sales_mean = float(r.recent_sales_mean) if np.isfinite(r.recent_sales_mean) else 0.0
            recent_sales_slope = float(r.recent_sales_slope) if np.isfinite(r.recent_sales_slope) else 0.0

            # -- 区分的调价策略 --
            bydir_slice = pol_bydir[["direction", "ate", "p", "confidence"]] if not pol_bydir.empty else pd.DataFrame()

            action, delta_pct, reason, conf = decide_action(
                current_price=current_price,
                recent_sales_mean=recent_sales_mean,
                recent_sales_slope=recent_sales_slope,
                overall_policy=overall_row,
                bydir_policy=bydir_slice,
            )
            actions.append((action, delta_pct, reason, conf))
        df["action"] = [a[0] for a in actions]
        df["delta_pct"] = [a[1] for a in actions]
        df["reason"] = [a[2] for a in actions]
        df["confidence"] = [a[3] for a in actions]

        # -- 输出：推荐表 + 诊断信息 --
        df["suggested_price"] = (df["current_price"] * (1.0 + df["delta_pct"])).round(4)
        # -- 这里算一个得分 --
        df["score"] = df["confidence"] * df["delta_pct"].abs()

        # -- 一般不要hold的 但如果真的太少 那就留几个hold audit一下价格 --
        df2 = df.sort_values("score", ascending=False).head(args.top_n).reset_index(drop=True)
        out_rows.append(df2)

        diag.append({
            "store_id": store_id,
            "ok": True,
            "current_wm_yr_wk": cur_wk,
            "n_items_cur_week": int(len(df)),
            "n_out": int(len(df2)),
            "policy_overall": {
                "pretrend_pass": int(overall_row["pretrend_pass"]) if overall_row is not None else None,
                "p": float(overall_row["p"]) if overall_row is not None else None,
                "ate": float(overall_row["ate"]) if overall_row is not None else None,
                "confidence": float(overall_row["confidence"]) if overall_row is not None else None,
                "mode": str(overall_row["recommendation_mode"]) if overall_row is not None else None,
            }
        })

    if not out_rows:
        print("[WARN] no recommendations produced.")
        return
    # -- 这里就是输出的一些东西 --
    out = pd.concat(out_rows, ignore_index=True)

    out_path = os.path.join(args.out_dir, "reco_weekly.csv")
    out.to_csv(out_path, index=False)
    print(f"[OUT] {out_path} rows={len(out):,}")

    diag_path = os.path.join(args.out_dir, "reco_diag.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"[OUT] {diag_path}")

    print("[DONE] run_pricing_engine finished.")


if __name__ == "__main__":
    main()
