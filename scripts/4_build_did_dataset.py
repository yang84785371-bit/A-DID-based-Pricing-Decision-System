# 4_build_did_dataset.py
from __future__ import annotations

import os
import math
import glob
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import time


'''
    整个文件的用途是 以调价事件为中心 生成control组和treat组的相关数据 一般的数据是以事件的数据做add 其他的话就是post eventtime 以及treat 作为调价效应
    省内存：
    1、用parquet省内存 同时从全局的变成store分去计算did 还可以并行
    2、将batch 变小 逻辑没变 内存小了
    3、将某些不必要的扫描操作具体化 避免了浪费时间与消耗内存 (355-356行)
    4、对周级轻表 我们重构了 按照store来分 避免扫描的时候需要扫描所有的parquet
    加速：
    1、events_df 被 store 过滤后，主循环复杂度直接下降 （归功于周级轻表的重建 可以直接按照store进行事件筛选）
    2、select_controls_for_event 的候选池被极大缩小 将candi level进行提前物化 不用每次都判断 把条件判断变成了索引访问
    3、不用iterrow 用itertuple
    其实加速和生内存 有时候也是混合着:
    1、周级轻表按 store 切分
    2、DID 主循环按 store 运行
    3、events_df 在 store 维度被提前过滤
    三者闭环

'''

# -- 配置/基础设置 --
# -- 作用是定义方法层参数 避免散落在代码中 --
@dataclass # 特殊的类类型 不用写initial 标注为数据行为 并且有可复现性 
class DidBuildConfig:
    # -- 事件窗口有多大 --
    K: int = 8
    # -- 每个事件contral的强度 -- 
    N_CTRL: int = 20

    # control的相似性层级
    # dept 更严格 cat更 松弛
    POOL_LEVEL: str = "dept"  # "dept" or "cat"

    # --- 哪些 event / control 有资格进入 DID ---
    # -- 最小覆盖
    MIN_PRE_WEEKS: int = 6      # 至少要有前X周可观测
    MIN_WIN_WEEKS: int = 12     # 窗口内至少X周可观测

    # -- 相对容忍 --
    MAX_REL_DIFF_PRE_SALES: float = 0.60   # |ctrl - tr| / (tr+eps)
    MAX_REL_DIFF_PRE_PRICE: float = 0.25

    # -- 斜率容忍 --
    MAX_ABS_DIFF_SLOPE: float = 5.0

    # --- 工程与统计风险控制 ---
    DISALLOW_CONTROL_REUSE: bool = True
    REUSE_GUARD_WEEKS: int = 8  

    # IO / chunking
    EVENTS_BATCH_SIZE: int = 100  
    WEEKLY_AGG_FLUSH_ROWS: int = 2_000_000  

'''
    tools
'''

# -- 确保目录存在 -- 
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# -- 将财务周映射成唯一的财务周id 作为后续的时间基准--
def read_calendar_week_index(calendar_csv: str) -> Dict[int, int]:
    cal = pd.read_csv(calendar_csv, usecols=["date", "wm_yr_wk"])  # 读取
    cal["date"] = pd.to_datetime(cal["date"]) # 确保date的类型
    wk_order = cal.groupby("wm_yr_wk")["date"].min().reset_index().sort_values("date") # 只留财务周的最小日期
    wk_order["wk_idx"] = np.arange(len(wk_order), dtype=np.int32) # 生成id
    return dict(zip(wk_order["wm_yr_wk"].astype(int).values, wk_order["wk_idx"].astype(int).values)) # zip 财务周编码以及wkid 并输出

# -- 计算一定时间内销量的线性趋势斜率 --
# -- 主要用于对pre-trend的可比性筛选 --
def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
        简单的线性回归
    """
    if len(x) < 2:
        return 0.0
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

# --带保护的均值计算 避免matching阶段被nan或者空值炸掉 --
def safe_mean(arr):
    if arr is None or len(arr) == 0:
        return np.nan
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))

# -- 计算相对差异 --
def rel_diff(a: float, b: float, eps: float = 1e-6) -> float:
    return abs(b - a) / (abs(a) + eps)

# -- 判断窗口内 是否发生过调价 --
def has_event_in_window(event_wk_list_sorted: List[int], left: int, right: int) -> bool:
    """
        返回一个值 就是 该item 在窗口时间内有无调价 
    """
    if not event_wk_list_sorted:
        return False
    # binary search
    import bisect
    i = bisect.bisect_left(event_wk_list_sorted, left)
    return i < len(event_wk_list_sorted) and event_wk_list_sorted[i] <= right

# -- 找到所有曾经建立过的panel part --
def list_panel_parts(panel_parts_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(panel_parts_dir, "part_*.parquet"))) # glob.glob就是找的意思 这里是找文件名
    if not files:
        raise FileNotFoundError(f"No parquet parts found under: {panel_parts_dir}")
    return files


'''
    第一步 构建周级轻表
'''
# -- 构建周级轻表 并且按照store进行切分 -- 
def build_weekly_store_parts(
    panel_parts_dir: str,
    weekly_store_dir: str,
    usecols: Optional[List[str]] = None,
) -> None:
    """
        对每一个(store,item,wm_yr_wk)进行聚合.
        并且对每一个store进行分块储存.

        输出格式:
        output/weekly_store_parts/
            store_id=CA_1/week_part_0000.parquet
            store_id=CA_1/week_part_0001.parquet
            ...
    """
    ensure_dir(weekly_store_dir) # 确保输出输出格式正确
    # -- 不传usetool就默认最小安全规模 --
    if usecols is None:
        usecols = [
            "store_id", "item_id", "wm_yr_wk",
            "sales", "sell_price",
            "dept_id", "cat_id", "state_id"
        ]
    # -- 获得所有parquet --
    files = list_panel_parts(panel_parts_dir)

    # -- 每一个store的缓冲 （周级别数据） --
    store_buffers: Dict[str, List[pd.DataFrame]] = {} # List[pd.DataFrame] ：该 store 已聚合出的若干周级 DataFrame
    store_row_counts: Dict[str, int] = {} # 记录每个 store 当前 buffer 里“已经攒了多少行”
    store_part_counters: Dict[str, int] = {} # 每个 store 已经写出了多少个 parquet 分块

    # -- 某一个 store的周级数据 攒够了我们就进行 生成parquet --
    def flush_store(store_id: str) -> None:
        # -- 看门店是不是我们想要的 --
        if store_id not in store_buffers or not store_buffers[store_id]:
            return
        # -- 连接想要转parquet的小表 -- 
        df = pd.concat(store_buffers[store_id], ignore_index=True)
        # -- 单独记录信息 --
        store_buffers[store_id] = []
        store_row_counts[store_id] = 0

        # -- 确认输出地址 --
        out_dir = os.path.join(weekly_store_dir, f"store_id={store_id}")
        # -- 确认类型 --
        ensure_dir(out_dir)
        # -- 这是我们做出来的第几个parquet --
        part_no = store_part_counters.get(store_id, 0) # 因为id是连续完整的 作为number（no.1 那个no）
        out_path = os.path.join(out_dir, f"week_part_{part_no:04d}.parquet") # 输出目录加上文件名
        store_part_counters[store_id] = part_no + 1 # 记录总共有多少个parquet

        # 进行编写
        df.to_parquet(out_path, index=False)

    # -- 读取索引 以及 日级parquet文件名 --
    for fi, path in enumerate(files):
        df = pd.read_parquet(path, columns=usecols)

        gcols = ["store_id", "item_id", "wm_yr_wk"] # 目标cols
        agg = df.groupby(gcols, sort=False).agg(
            weekly_sales=("sales", "sum"), # 对一周的销量求sum
            weekly_price=("sell_price", "mean"), # 对价格求mean
            dept_id=("dept_id", "first"), # 对dept求frist 因为都是一样的
            cat_id=("cat_id", "first"), # 同上
            state_id=("state_id", "first"), # 同上
        ).reset_index() # 根据目标的cols进行聚合

        '''
            我们这里的日级parquet是根据时间来分的
            由于我们现在要按照store来分 所以对于数据 我们需要攒一下 攒够了在加入buffer然后 到了一定数量再给parquet
            但这里一周可能被两个parquet分割 所以会有一点偏差（有待优化）
        '''

        # -- 添加到每一个store的缓冲区 -- 
        for store_id, sdf in agg.groupby("store_id", sort=False): # 按照store进行分小表 sdf是subset df 
            store_id = str(store_id) # 获取store 名称
            store_buffers.setdefault(store_id, []).append(sdf) # 初始化 如果类似于get
            store_row_counts[store_id] = store_row_counts.get(store_id, 0) + len(sdf) # 记录该store id 已经存了多少条

            # -- 如果太多了就将其转化为parquet --
            if store_row_counts[store_id] >= DidBuildConfig().WEEKLY_AGG_FLUSH_ROWS:
                flush_store(store_id)
        
        # -- 每5次print一下是什么情况 --
        if (fi + 1) % 5 == 0:
            print(f"[WEEKLY] processed {fi+1}/{len(files)} parts")

    '''
        兜底：刚刚我们转换成parquet的条件是大于某个数 那不满足这个数的 也不会抛弃 我们都求一次就可以了 上面的逻辑 parquet是连续的 所以这里基本上就是最后一个parquet文件 最后几个数据了
    '''
    for store_id in list(store_buffers.keys()):
        flush_store(store_id)

    print(f"[OK] weekly store parts saved to: {weekly_store_dir}")

# -- 一次性加载某一个 store 的所有周级数据 -- 
def load_weekly_store(store_dir: str, store_id: str) -> pd.DataFrame:
    pdir = os.path.join(store_dir, f"store_id={store_id}")
    parts = sorted(glob.glob(os.path.join(pdir, "week_part_*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No weekly parts for store: {store_id} under {pdir}")

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) # stack到一起
    gcols = ["store_id", "item_id", "wm_yr_wk"] # 进行聚合
    # -- 为了安全 分块有风险 很可能这个分块一行 那个分块一行的 --
    df = df.groupby(gcols, as_index=False).agg(
        weekly_sales=("weekly_sales", "sum"),
        weekly_price=("weekly_price", "mean"),
        dept_id=("dept_id", "first"),
        cat_id=("cat_id", "first"),
        state_id=("state_id", "first"),
    )
    return df


# -----------------------------
# Step B/C: Build DID dataset
# -----------------------------

# -- 两个目的一个是标注好时间和事件id  一个是对每个item发生过的时间列表 -- 
def build_events_index(price_events_csv: str, wk_map: Dict[int, int]) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], List[int]]]:
    """
        events_df 包括 wk_idx 和 event_id
        events_by_item: (store,item) -> 发生过事件的时间id的列表
    """
    ev = pd.read_csv(price_events_csv) # 调价事件表

    # -- ev中需要的cols --
    # -- 分别是 store_id, item_id, event_wm_yr_wk, old_price, new_price, pct_change, direction --
    required = ["store_id", "item_id", "event_wm_yr_wk", "old_price", "new_price", "pct_change", "direction"]
    missing = [c for c in required if c not in ev.columns]
    if missing:
        raise ValueError(f"price_events.csv missing columns: {missing}") # 兜底

    # -- 确保类型 --
    ev["store_id"] = ev["store_id"].astype(str)
    ev["item_id"] = ev["item_id"].astype(str)
    ev["event_wm_yr_wk"] = ev["event_wm_yr_wk"].astype(int)
    ev["wk_idx"] = ev["event_wm_yr_wk"].map(wk_map).astype("Int64")

    # -- 确保时间id没问题 --
    bad = ev["wk_idx"].isna().sum()
    if bad > 0:
        ev = ev.dropna(subset=["wk_idx"]).copy()
    ev["wk_idx"] = ev["wk_idx"].astype(int)

    ev = ev.reset_index(drop=True)
    ev["event_id"] = np.arange(len(ev), dtype=np.int64) # 赋予事件id

    events_by_item: Dict[Tuple[str, str], List[int]] = {} # 构造一个双key的字典
    for (s, i), sdf in ev.groupby(["store_id", "item_id"], sort=False):
        lst = sorted(sdf["wk_idx"].tolist())
        events_by_item[(s, i)] = lst

    return ev, events_by_item

# -- 对事件前 K 周计算 --
# -- 输出的是 斜率
def compute_pre_stats(window_df: pd.DataFrame, event_wk_idx: int, K: int) -> Tuple[float, float, float, int]:
    """
        计算 处理窗口时间的(pre_mean_sales, pre_mean_price, pre_slope_sales, pre_n_weeks)
    """
    # -- 取窗口数据的事件发生前的df --
    pre = window_df[(window_df["wk_idx"] >= event_wk_idx - K) & (window_df["wk_idx"] <= event_wk_idx - 1)].copy()
    pre = pre.sort_values("wk_idx") # 排序
    pre_n = len(pre) # 计算可用观测

    pre_mean_sales = safe_mean(pre["weekly_sales"].to_numpy()) # 平均销量
    pre_mean_price = safe_mean(pre["weekly_price"].to_numpy()) # 平均价格

    # slope: x = -K..-1 mapped from wk_idx
    if pre_n >= 2:
        x = (pre["wk_idx"].to_numpy() - event_wk_idx).astype(int)  # 这里只能取例如 -3 -2 -1
        y = pre["weekly_sales"].to_numpy() # 这几周的价格
        pre_slope = linear_slope(x, y)  # 得到斜率
    else:
        pre_slope = 0.0

    return pre_mean_sales, pre_mean_price, pre_slope, pre_n

# -- 构造 [-K, +K] 的事件窗口数据 --
def make_event_window(
    item_df: Optional[pd.DataFrame],
    event_wk_idx: int,
    K: int,
) -> pd.DataFrame:
    """
    item_df: 单个 item 的周级数据（已经按 wk_idx 排序）
    """
    if item_df is None or item_df.empty:
        return pd.DataFrame()

    idx = item_df["wk_idx"].to_numpy()

    l = np.searchsorted(idx, event_wk_idx - K, side="left")
    r = np.searchsorted(idx, event_wk_idx + K, side="right")

    win = item_df.iloc[l:r].copy()
    if win.empty:
        return win

    win["event_time"] = win["wk_idx"] - event_wk_idx
    win["post"] = (win["event_time"] >= 0).astype(int)
    return win


# -- 选择对照组item --
def select_controls_for_event(
    item_groups: Dict[str, pd.DataFrame],
    events_by_item: Dict[Tuple[str, str], List[int]],
    store_id: str,
    treated_item: str,
    event_wk_idx: int,
    K: int,
    N_CTRL: int,
    pool_level: str,
    cfg: DidBuildConfig,
    control_last_used: Dict[Tuple[str, str], int],
    pool_dept_items: Dict[str, np.ndarray],
    pool_cat_items: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, float]]:

    """
    Return:
      controls_selected_df: weekly window rows for selected control items (stacked)
      treated_stats: dict for debug
    """
    # -- 处理组的窗口数据 --
    item_df = item_groups.get(treated_item)
    tr_win = make_event_window(item_df, event_wk_idx, K)


    if tr_win.empty:
        return pd.DataFrame(), {}

    # -- treated的资格审查 --
    tr_pre_mean_sales, tr_pre_mean_price, tr_pre_slope, tr_pre_n = compute_pre_stats(tr_win, event_wk_idx, K) # 得到这几个数先 
    tr_cov = len(tr_win) # 看看窗口覆盖了多少

    # -- 处理组的数据 一般用来debug 看有没有选错处理组 --
    treated_stats = dict(
        tr_pre_mean_sales=tr_pre_mean_sales,
        tr_pre_mean_price=tr_pre_mean_price,
        tr_pre_slope=tr_pre_slope,
        tr_pre_n=tr_pre_n,
        tr_win_n=tr_cov,
    )

    #  -- 如果pre太小 或者cov 太小 那这个treat 做不了 要舍弃 --
    if tr_pre_n < cfg.MIN_PRE_WEEKS or tr_cov < cfg.MIN_WIN_WEEKS:
        return pd.DataFrame(), treated_stats

    # -- 确定 control 候选池的分组维度 --
    tr_dept = str(tr_win["dept_id"].iloc[0])
    tr_cat  = str(tr_win["cat_id"].iloc[0])
    pool_val = tr_dept if pool_level == "dept" else tr_cat


    # -- 先拿到candidate的 也就是相同的商店以及相同候选池的 --
    if pool_level == "dept":
        cand_items = pool_dept_items.get(pool_val, [])
    else:
        cand_items = pool_cat_items.get(pool_val, [])


    # drop treated itself
    cand_items = [cid for cid in cand_items if cid != treated_item]

    if not cand_items:
        return pd.DataFrame(), treated_stats

    # -- 硬约束 如果你想成为control 你必须要在treat item的窗口内无调价 甚至更严格
    left = event_wk_idx - K
    right = event_wk_idx + K

    eligible: List[str] = [] # 可用的
    for cid in cand_items: # 遍历
        # -- 干净的item 在窗口事件内没有调价 --
        ev_list = events_by_item.get((store_id, cid), []) # item发生事件的周数
        if has_event_in_window(ev_list, left, right): # 是否在窗口内
            continue

        # -- 反重复使用避免隐藏的加权 --
        if cfg.DISALLOW_CONTROL_REUSE:
            last = control_last_used.get((store_id, cid), None) # 发生事件的事件
            if last is not None and abs(event_wk_idx - last) <= cfg.REUSE_GUARD_WEEKS: # 这里必须要求连续的使用事件要大于某个值
                continue

        eligible.append(cid) # 进入可用名单

    # -- 如果一个都不可用 那没办法 --
    if not eligible:
        return pd.DataFrame(), treated_stats

    # -- 软过滤：可比性 matching + 打分排序 --
    scored: List[Tuple[float, str]] = [] # 打分

    # -- 先计算处理组统计量避免nan --
    tr_sales_ref = tr_pre_mean_sales if not math.isnan(tr_pre_mean_sales) else 0.0 # 如果是nan就变0
    tr_price_ref = tr_pre_mean_price if not math.isnan(tr_pre_mean_price) else 0.0

    # -- 对于可用的 进行软约束 --
    for cid in eligible:
        c_item_df = item_groups.get(cid)
        c_win = make_event_window(c_item_df, event_wk_idx, K)

        if c_win.empty or len(c_win) < cfg.MIN_WIN_WEEKS: # 如果没有 或者太短 也不行
            continue
        
        # 计算调价前的的统计量 --
        c_pre_mean_sales, c_pre_mean_price, c_pre_slope, c_pre_n = compute_pre_stats(c_win, event_wk_idx, K)
        if c_pre_n < cfg.MIN_PRE_WEEKS:# 总的太少不行
            continue

        # -- 相对过滤 如果销量 价格 斜率 出入太大 不行 --
        if rel_diff(tr_sales_ref, c_pre_mean_sales) > cfg.MAX_REL_DIFF_PRE_SALES: # 与
            continue
        if rel_diff(tr_price_ref, c_pre_mean_price) > cfg.MAX_REL_DIFF_PRE_PRICE:
            continue
        if abs(c_pre_slope - tr_pre_slope) > cfg.MAX_ABS_DIFF_SLOPE:
            continue

        # -- 距离得分 越小越好 --
        # -- 进行混合 --
        d = 0.0
        d += rel_diff(tr_sales_ref, c_pre_mean_sales)
        d += rel_diff(tr_price_ref, c_pre_mean_price)
        d += abs(c_pre_slope - tr_pre_slope) / (abs(tr_pre_slope) + 10.0)  # mild scaling

        scored.append((float(d), cid))

    if not scored:
        return pd.DataFrame(), treated_stats

    scored.sort(key=lambda x: x[0])
    picked = [cid for _, cid in scored[:N_CTRL]] # 取得分前n个高的

    # -- 为anti reuse 记录使用时间 --
    if cfg.DISALLOW_CONTROL_REUSE:
        for cid in picked:
            control_last_used[(store_id, cid)] = event_wk_idx # 如果使用了 就把这些数据打上 目标事件的时间标签

    # -- 并且将窗口cat和返回 -- 
    ctrl_windows = []
    for cid in picked:
        c_item_df = item_groups.get(cid)
        w = make_event_window(c_item_df, event_wk_idx, K)
        if not w.empty:
            ctrl_windows.append(w)

    if not ctrl_windows:
        return pd.DataFrame(), treated_stats

    out = pd.concat(ctrl_windows, ignore_index=True) # 将所有control的窗口数据cat在一起
    return out, treated_stats

# -- 拼 DID 数据集 -- 
# -- 这个是最复杂最关键也是最重要的一环 还是最终目的 --

'''
    最后会生成这样一个长表
    [event_id, store_id, item_id, wk_idx, event_time,
    treat, post,
    sales, price,
    old_price, new_price, pct_change, direction,
    dept_id, cat_id, state_id]

'''

def build_did_parts(
    events_df: pd.DataFrame,
    events_by_item: Dict[Tuple[str, str], List[int]],
    wk_map: Dict[int, int],
    weekly_store_dir: str,
    out_did_dir: str,
    cfg: DidBuildConfig,
    stores_filter: Optional[List[str]] = None,
) -> None:
    # -- ensure一下格式有没有问题 --
    ensure_dir(out_did_dir)

    # -- 找到store的列表 --
    store_dirs = sorted(glob.glob(os.path.join(weekly_store_dir, "store_id=*")))
    all_stores = [os.path.basename(p).split("store_id=")[-1] for p in store_dirs]

    # optional filter
    if stores_filter is not None:
        stores_set = set(stores_filter)
        stores = [s for s in all_stores if s in stores_set]
    else:
        stores = all_stores

    print(f"[DID] stores to run: {stores} (n={len(stores)})")

    # ---- iterate stores ----
    for si, store_id in enumerate(stores, start=1):
        print(f"[DID] loading weekly store table: {store_id} ({si}/{len(stores)})")

        # -- 每个 store 写到自己的子目录，避免并行撞文件 --
        out_store_dir = os.path.join(out_did_dir, f"store_id={store_id}")
        ensure_dir(out_store_dir)

        # --每个 store 自己的 part_no，从 0 开始 --
        part_no = 0

        # -- 读取 --
        wk_files = sorted(glob.glob(os.path.join(weekly_store_dir, f"store_id={store_id}", "week_part_*.parquet")))
        if not wk_files:
            print(f"[DID] skip {store_id}: no weekly parts found")
            continue

        # -- 你原来怎么读就怎么读，这里给个常见写法： --
        store_weekly = pd.concat([pd.read_parquet(fp) for fp in wk_files], ignore_index=True)

        # -- 防御性收口：确保 (store,item,week) 唯一 --
        gcols = ["store_id", "item_id", "wm_yr_wk"]
        store_weekly = store_weekly.groupby(gcols, as_index=False).agg(
            weekly_sales=("weekly_sales", "sum"),
            weekly_price=("weekly_price", "mean"),
            dept_id=("dept_id", "first"),
            cat_id=("cat_id", "first"),
            state_id=("state_id", "first"),
        )

        # -- 增加 wkid --
        store_weekly = store_weekly.copy()
        store_weekly["wk_idx"] = store_weekly["wm_yr_wk"].astype(int).map(wk_map)
        store_weekly = store_weekly.dropna(subset=["wk_idx"]).copy()
        store_weekly["wk_idx"] = store_weekly["wk_idx"].astype(int)
        store_weekly["store_id"] = store_weekly["store_id"].astype(str)
        store_weekly["item_id"] = store_weekly["item_id"].astype(str)

        item_groups: Dict[str, pd.DataFrame] = {}

        for item_id, g in store_weekly.groupby("item_id", sort=False):
            g = g.sort_values("wk_idx").reset_index(drop=True)
            item_groups[str(item_id)] = g


        for c in ["item_id", "dept_id", "cat_id", "state_id"]:
            if c in store_weekly.columns:
                store_weekly[c] = store_weekly[c].astype("category")


        pool_dept_items: Dict[str, np.ndarray] = {}
        for dept, g in store_weekly.groupby("dept_id", sort=False):
            pool_dept_items[str(dept)] = g["item_id"].astype(str).unique()

        # cat-level pool: cat_id -> np.array(item_id)
        pool_cat_items: Dict[str, np.ndarray] = {}
        for cat, g in store_weekly.groupby("cat_id", sort=False):
            pool_cat_items[str(cat)] = g["item_id"].astype(str).unique()



        # -- 该门店的事件 --
        store_id = str(store_id)
        store_events = events_df[events_df["store_id"] == store_id].copy()
        store_events = store_events[[
            "event_id","item_id","wk_idx","event_wm_yr_wk",
            "old_price","new_price","pct_change","direction"
        ]].copy()

        if store_events.empty:
            print(f"[DID] skip {store_id}: no events")
            continue

        n_events = len(store_events)
        t_store0 = time.time()
        last_print = t_store0

        seen = 0
        kept = 0
        drop_tr_win = 0      # treated window empty
        drop_tr_cov = 0      # treated coverage不足（pre或win周数不足）
        drop_ctrl = 0

        # -- 反重复使用列表 --
        control_last_used: Dict[Tuple[str, str], int] = {}

        # -- 行缓冲区 --
        rows_buffer: List[pd.DataFrame] = []

        for ev in store_events.itertuples(index=False):
            seen += 1

            event_id = int(ev.event_id)
            item_id = str(ev.item_id)
            event_wk_idx = int(ev.wk_idx)
            event_wm_yr_wk = int(ev.event_wm_yr_wk)

            item_df = item_groups.get(item_id)
            tr_win = make_event_window(item_df, event_wk_idx, cfg.K)


            if tr_win.empty:
                drop_tr_win += 1
                continue

            # 选 control（clean window + matching + anti-reuse + fixed N）
            ctrl_win, tr_stats = select_controls_for_event(
                item_groups=item_groups,
                events_by_item=events_by_item,
                store_id=store_id,
                treated_item=item_id,
                event_wk_idx=event_wk_idx,
                K=cfg.K,
                N_CTRL=cfg.N_CTRL,
                pool_level=cfg.POOL_LEVEL,
                cfg=cfg,
                control_last_used=control_last_used,
                pool_dept_items=pool_dept_items,
                pool_cat_items=pool_cat_items,
            )
            if not tr_stats:
                # treated window empty（理论上不会到这，因为外面已判断 tr_win.empty）
                drop_tr_win += 1
                continue

            # treated coverage 不够（select 内部判掉了）
            if tr_stats.get("tr_pre_n", 0) < cfg.MIN_PRE_WEEKS or tr_stats.get("tr_win_n", 0) < cfg.MIN_WIN_WEEKS:
                drop_tr_cov += 1
                continue

            # 固定 N 控制（不够就丢事件）
            unique_ctrl_items = ctrl_win["item_id"].nunique() if not ctrl_win.empty else 0
            if unique_ctrl_items < cfg.N_CTRL:
                drop_ctrl += 1
                continue

            '''
                给 treated / control 打标签（treat=1/0）并补齐事件锚点信息
            ''' 

            tr_out = tr_win.copy()
            tr_out["event_id"] = event_id
            tr_out["treat"] = 1
            tr_out["store_id"] = store_id
            tr_out["event_wm_yr_wk"] = event_wm_yr_wk
            tr_out["treated_item_id"] = item_id

            ctrl_out = ctrl_win.copy()
            ctrl_out["event_id"] = event_id
            ctrl_out["treat"] = 0
            ctrl_out["store_id"] = store_id
            ctrl_out["event_wm_yr_wk"] = event_wm_yr_wk
            ctrl_out["treated_item_id"] = item_id


            '''
                附加事件元信息 + 统一字段（sales/price）
            '''
            for df_ in (tr_out, ctrl_out):
                df_["old_price"] = float(ev.old_price)
                df_["new_price"] = float(ev.new_price)
                df_["pct_change"] = float(ev.pct_change)
                df_["direction"] = str(ev.direction)


                # attach debug stats for audit (optional but helps “defend”)
                df_["tr_pre_mean_sales"] = float(tr_stats.get("tr_pre_mean_sales", np.nan))
                df_["tr_pre_mean_price"] = float(tr_stats.get("tr_pre_mean_price", np.nan))
                df_["tr_pre_slope"] = float(tr_stats.get("tr_pre_slope", np.nan))
                df_["tr_pre_n"] = int(tr_stats.get("tr_pre_n", 0))
                df_["tr_win_n"] = int(tr_stats.get("tr_win_n", 0))

                # normalize column names for DID stage
                df_["sales"] = df_["weekly_sales"].astype(float)
                df_["price"] = df_["weekly_price"].astype(float)

            out = pd.concat([tr_out, ctrl_out], ignore_index=True)

            # -- 只保留需要的列 --
            keep = [
                "event_id", "store_id", "treated_item_id", "item_id",
                "wm_yr_wk", "wk_idx", "event_wm_yr_wk",
                "event_time", "post", "treat",
                "sales", "price",
                "dept_id", "cat_id", "state_id",
                "old_price", "new_price", "pct_change", "direction",
                "tr_pre_mean_sales", "tr_pre_mean_price", "tr_pre_slope", "tr_pre_n", "tr_win_n",
            ]
            out = out[keep]

            rows_buffer.append(out)
            kept += 1
            now = time.time()
            if (now - last_print) >= 5.0:  # 每5秒打印一次
                elapsed = now - t_store0
                speed = seen / max(elapsed, 1e-9)
                remain = n_events - seen
                eta_sec = remain / max(speed, 1e-9)

                print(
                    f"[DID][{store_id}] {seen:,}/{n_events:,} "
                    f"kept={kept:,} drop_tr_win={drop_tr_win:,} drop_tr_cov={drop_tr_cov:,} drop_ctrl={drop_ctrl:,} "
                    f"speed={speed:.2f} ev/s ETA={eta_sec/60:.1f} min "
                    f"buffer_events={len(rows_buffer):,}"
                )
                last_print = now


            # -- 根据一定的大小 我们进行分块
            if len(rows_buffer) >= cfg.EVENTS_BATCH_SIZE:
                did_part = pd.concat(rows_buffer, ignore_index=True)
                out_path = os.path.join(out_store_dir, f"did_part_{part_no:04d}.parquet")
                did_part.to_parquet(out_path, index=False)
                print(f"[OK] wrote {out_path} rows={len(did_part):,}")
                rows_buffer = []
                part_no += 1

        # store event loop 结束后，先把剩余 buffer 写掉
        if rows_buffer:
            did_part = pd.concat(rows_buffer, ignore_index=True)
            out_path = os.path.join(out_store_dir, f"did_part_{part_no:04d}.parquet")
            did_part.to_parquet(out_path, index=False)
            print(f"[OK] wrote {out_path} rows={len(did_part):,}")
            rows_buffer = []
            part_no += 1

        elapsed = time.time() - t_store0
        print(
            f"[DID][{store_id}] DONE events={n_events:,} seen={seen:,} kept={kept:,} "
            f"drop_tr_win={drop_tr_win:,} drop_ctrl={drop_ctrl:,} "
            f"elapsed={elapsed/60:.1f} min"
        )



    print(f"[DONE] did parts saved to: {out_did_dir}")

def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/raw_data") # raw data地址
    ap.add_argument("--output_dir", type=str, default="output") # 输出地址
    ap.add_argument("--K", type=int, default=8) # 事件窗口半径
    ap.add_argument("--n_ctrl", type=int, default=20) # 每一个窗口事件选多少个control item
    ap.add_argument("--pool_level", type=str, default="dept", choices=["dept", "cat"]) # control 的候选池层级
    ap.add_argument("--rebuild_weekly", action="store_true", help="force rebuild weekly_store_parts") # 是否强制重构周级轻表
    ap.add_argument("--disallow_reuse", action="store_true", help="disallow control reuse near in time") # 是否禁止同一个 control item 被多个 event 在“相近时间”重复使用
    ap.add_argument("--stores", type=str, default="", help="comma-separated store_ids, empty=all (e.g. CA_1,CA_2)")
    args = ap.parse_args()

    # -- 参数赋值 --
    cfg = DidBuildConfig() # 获得配置
    cfg.K = int(args.K)
    cfg.N_CTRL = int(args.n_ctrl)
    cfg.POOL_LEVEL = str(args.pool_level)
    cfg.DISALLOW_CONTROL_REUSE = bool(args.disallow_reuse)

    # -- 制造calendar/event/panel地址 -- 
    calendar_csv = os.path.join(args.data_dir, "calendar.csv")
    price_events_csv = os.path.join(args.output_dir, "price_events.csv")
    panel_parts_dir = os.path.join(args.output_dir, "panel_parts")
    # -- 制造门店周表/did数据输出 地址 --
    weekly_store_dir = os.path.join(args.output_dir, "weekly_store_parts")
    out_did_dir = os.path.join(args.output_dir, "did_parts")

    # -- 确认这个输出地址存在 --
    ensure_dir(args.output_dir)

    # 将财务周映射成id 把带业务编码的周 → 严格递增、无断裂的时间轴
    wk_map = read_calendar_week_index(calendar_csv)

    # -- events_by_item ：每个 (store, item) 在历史上“所有调价发生的周”列表 --
    # -- events_df 把财务周换成时间id 为每个事件弄一个可以进行索引的事件id
    events_df, events_by_item = build_events_index(price_events_csv, wk_map)

    # -- 因果预处理: 将日级数据整合成周级别 并进行切分--
    # -- 只要满足你明确要求重建”或“周级结果不存在/不完整
    if args.rebuild_weekly or (not os.path.exists(weekly_store_dir)) or (len(glob.glob(os.path.join(weekly_store_dir, "store_id=*"))) == 0):
        print("[STEP] building weekly_store_parts ...")
        build_weekly_store_parts(panel_parts_dir=panel_parts_dir, weekly_store_dir=weekly_store_dir)

    # -- 获得store的列表 --
    stores_filter = [s.strip() for s in args.stores.split(",") if s.strip()] if args.stores else None

    # -- 在control的情况下 建立一个did 数据集 --
    print("[STEP] building did_parts ...")
    build_did_parts(
        events_df=events_df,  # 所有的调价事件的总表 有事件id 和时间 id 
        events_by_item=events_by_item,  # 分item展示调价的参数
        wk_map=wk_map, # id 对照 财务周
        weekly_store_dir=weekly_store_dir, # 周级轻表 其实就是价格表
        out_did_dir=out_did_dir, # 输出目录
        cfg=cfg, # 配置参数
        stores_filter=stores_filter
    )


if __name__ == "__main__":
    main()
