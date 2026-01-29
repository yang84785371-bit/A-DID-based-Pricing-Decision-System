# 2_check_price.py
'''
    该脚本文件用于检查生成的chunk是否合规
'''
import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Quick sanity checks for pricing panel parts")
    p.add_argument("--parts_dir", type=str, required=True, help="Directory with part_*.parquet") # 分块的地址
    p.add_argument("--part", type=str, default="", help="Specific part file name, e.g. part_0000.parquet (optional)") # 具体哪个分块
    p.add_argument("--n_series", type=int, default=2000, help="How many (store_id,item_id) series to sample for checks") # 抽样多少条
    p.add_argument("--top_k", type=int, default=10, help="Show top K series by number of price changes") # 显示价格变动次数最多的前K条
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling series") # 随机种子
    return p.parse_args()

# -- 简单的地址合成 -- 
# -- 最后得到的是分块的地址 --
def pick_part(parts_dir: Path, part_name: str) -> Path:
    if part_name:
        p = parts_dir / part_name
        if not p.exists():
            raise FileNotFoundError(f"part not found: {p}")
        return p
    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part_*.parquet found in {parts_dir}")
    return parts[0]


def main():
    # -- 命令行参数 --
    args = parse_args()
    parts_dir = Path(args.parts_dir) # 总的地址

    part_path = pick_part(parts_dir, args.part) # 制造一个取分块的地址
    df = pd.read_parquet(part_path) # 读取

    need_cols = ["store_id", "item_id", "date", "sales", "sell_price"] # 需要的cols
    missing_cols = [c for c in need_cols if c not in df.columns] # 缺少需要的cols
    # -- 兜底：没有必要的字段 --
    if missing_cols:
        raise ValueError(f"Missing required columns in {part_path.name}: {missing_cols}")

    # -- 展示基本信息 --
    print(f"[INFO] loaded: {part_path}")
    print(f"[INFO] shape: {df.shape}")
    print(f"[INFO] date range: {df['date'].min()} -> {df['date'].max()}")

    # -- 缺失 -- 
    miss_price = df["sell_price"].isna().mean() # isna得到一个bool列 然后求mean 得到率
    miss_sales = df["sales"].isna().mean()
    print(f"[CHECK] sell_price missing rate: {miss_price:.4%}")
    print(f"[CHECK] sales missing rate:      {miss_sales:.4%}")

    # -- 抽取样本序列 -- 
    series = df[["store_id", "item_id"]].drop_duplicates() # 去掉重复的 为什么是store和item呢 因为这个复合键已经是唯一的id指向 → 个分块里一共有多少条独立的时间序列”
    n_all = len(series) # 一共有多少个独立的时间序列
    n_take = min(args.n_series, n_all) # 可以拿的数量 all少拿all 参少就直接拿
    series_s = series.sample(n=n_take, random_state=args.seed) # 抽取若干个时间序列 series的定义已经保证其完整性

    # -- 只保留内连接的键 就是两者都要有的 其实就是相对df进行左连接 但是 series没有的 df也不要 --
    df_s = df.merge(series_s, on=["store_id", "item_id"], how="inner").copy()
    df_s = df_s.sort_values(["store_id", "item_id", "date"])

    # -- 统计时间序列进行了多少次的调价 --
    def count_changes(s: pd.Series) -> int:
        # Count changes ignoring NaN; a change occurs when current != prev among valid prices
        x = s.dropna() # 不要na
        if len(x) <= 1:
            return 0
        return int((x != x.shift(1)).sum() - 1 if len(x) > 0 else 0)

    change_cnt = (
        df_s.groupby(["store_id", "item_id"])["sell_price"] # 对每个个体
        .apply(count_changes) # 统计有多少次调价
        .rename("n_price_changes") # 作为新的字段
        .reset_index() # 重置索引编号
    )

    # -- 总结 --
    pct_any = (change_cnt["n_price_changes"] > 0).mean()
    print(f"[CHECK] sampled series: {n_take}/{n_all}")
    print(f"[CHECK] series with >=1 price change: {pct_any:.2%}")
    print("[CHECK] n_price_changes quantiles:")
    print(change_cnt["n_price_changes"].quantile([0, 0.5, 0.9, 0.99, 1.0]).to_string())

    # -- 调价次数最多的前k个个体 --
    top = change_cnt.sort_values("n_price_changes", ascending=False).head(args.top_k)
    print("\n[TOP] series with most price changes (sampled):")
    print(top.to_string(index=False))

    # -- 展示一个例子 --
    if len(top) > 0 and top.iloc[0]["n_price_changes"] > 0: # 必须有 并且 第一个要调价大于0
        st = top.iloc[0]["store_id"] 
        it = top.iloc[0]["item_id"]
        ex = df[df["store_id"].eq(st) & df["item_id"].eq(it)].sort_values("date")[
            ["date", "sales", "sell_price"]
        ] # 从当前分块里，把这条序列的数据捞出来
        print(f"\n[EXAMPLE] price & sales over time for store_id={st}, item_id={it} (first 30 rows):")
        print(ex.head(30).to_string(index=False))
    # -- 看调价比例 如果调价比例太小 就无法进行后续的定价分析了 --
    print("\n[NEXT] If price does change (series_with_change not near 0), we can define 'price change events' and proceed to DID.")


if __name__ == "__main__":
    main()
