# 3_extract_price_events.py
'''
    回答的问题是 怎么才算一次调价 
    生成的是 store_id | item_id | event_wm_yr_wk | old_price | new_price | direction 这样的一个表
    本质是 定价系统的“历史决策日志 也就是 decision log
'''
import argparse
from pathlib import Path
import pandas as pd

# -- 命令行参数 --
def parse_args():
    p = argparse.ArgumentParser("Extract price change events from chunked M5 panel parts")
    p.add_argument("--parts_dir", type=str, required=True, help="Directory with part_*.parquet files")
    p.add_argument("--glob", type=str, default="part_*.parquet", help="Glob pattern for part files") # 全取part
    p.add_argument("--out_path", type=str, required=True, help="Output path for events (csv or parquet)")
    p.add_argument("--out_format", type=str, default="csv", choices=["csv", "parquet"], help="Output format")
    p.add_argument("--min_abs_change", type=float, default=0.0, help="Min absolute price change to count as event") # 定义调价标准
    p.add_argument("--min_pct_change", type=float, default=0.0, help="Min pct price change to count as event (e.g., 0.05)")
    p.add_argument("--max_parts", type=int, default=-1, help="For debug: limit number of part files. -1 means all") # 用于debug 最大的part 选取
    return p.parse_args()

# -- 更新写入状态 并且生成文件 --
def _write_events(df_events: pd.DataFrame, out_path: Path, out_format: str, first_write: bool):
    # -- 如果事件为空 那就返回写入状态 --
    if df_events.empty:
        return first_write
    #  -- 确保目录存在 -- 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -- CSV 不支持“结构化追加”，只能靠约定 --
    if out_format == "csv":
        header = first_write
        df_events.to_csv(out_path, index=False, mode="w" if first_write else "a", header=header)
    else:
        # -- 一个 chunk → 一个 parquet 文件 --
        part_dir = out_path.parent / (out_path.stem + "_parts")
        part_dir.mkdir(parents=True, exist_ok=True) # 建一个专门存放“分片 parquet”的目录。
        idx = 0
        # -- 找空文件名 --
        while (part_dir / f"events_{idx:06d}.parquet").exists():
            idx += 1
        df_events.to_parquet(part_dir / f"events_{idx:06d}.parquet", index=False)
    # -- 不然后续都会标记为非第一次写入 --
    return False 

def main():
    # -- 命令行参数 初始化一些值 --
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    out_path = Path(args.out_path)

    part_files = sorted(parts_dir.glob(args.glob)) # glob就是全取的意思
    # -- 兜底一下 如果找不到某个part的话 --
    if not part_files:
        raise FileNotFoundError(f"No files matched {args.glob} under {parts_dir}")

    # -- 用于debug的最大part
    if args.max_parts != -1:
        part_files = part_files[: args.max_parts]

    print(f"[INFO] found {len(part_files)} part files under {parts_dir}")

    # -- state是用来保证不同chunk之间价格的连续性的 --
    # -- 字典state的结构 key: (store_id, item_id) -> (last_wm_yr_wk, last_price) --
    last_state = {}

    # -- 控制输出 为了不重复写表头 --
    first_write = True

    # -- 只需要少量的cols --
    need_cols = ["store_id", "item_id", "dept_id", "cat_id", "state_id", "date", "wm_yr_wk", "sell_price"]

    # -- 按照chunk 进行循环
    for pi, part_path in enumerate(part_files, 1): # 1代表从1开始索引 因为part的命名是这样
        print(f"[PART {pi}/{len(part_files)}] reading {part_path.name}") # 进度条
        df = pd.read_parquet(part_path, columns=need_cols) #读取单个chunk的数据 只要need cols

        # -- 保证类型
        df["wm_yr_wk"] = df["wm_yr_wk"].astype("int32")
        # date is string like 'YYYY-MM-DD' in your panel; convert once for min/max and sorting
        df["date"] = pd.to_datetime(df["date"])

        # -- 用来保存每个 (store_id, item_id) 对应的稳定属性标签 不随时间进行变化的 --
        meta = df[["store_id", "item_id", "dept_id", "cat_id", "state_id"]].drop_duplicates()

        # -- 把“日级的价格记录”，压缩成“每个商品 × 门店 × 财务周的一条价格记录” --
        weekly = (
            df.groupby(["store_id", "item_id", "wm_yr_wk"], as_index=False) # 作为唯一标签 对着复合键 进行分表
              .agg(
                  week_start_date=("date", "min"), # 聚合 用日期的最小值和价格的最大值
                  sell_price=("sell_price", "max"), # 这里 sell_price是新的字段名 "sell_price"用来聚合的字段 "max"是函数
              ) 
        )

        # -- 只保留有价格的week --
        weekly = weekly[weekly["sell_price"].notna()].copy() # copy 就是 索引/切片 之后打算修改值就要copy 这个主要是因为pandas对于切片/索引输出是view还是copy有不确定性
        if weekly.empty:
            print("[INFO] no observed prices in this part (all NaN after aggregation). skip.")
            continue
        # -- 排序 --
        weekly = weekly.sort_values(["store_id", "item_id", "wm_yr_wk"]).reset_index(drop=True)

        # -- 对于每个(store_id, item_id) 检测其变化 --
        events = []
        for (st, it), g in weekly.groupby(["store_id", "item_id"], sort=False): # g是一个分表
            g = g.sort_values("wm_yr_wk") # 对时间排序
            key = (st, it) # 个体

            # -- 对上一个chunk的历史状态赋予初值 -- 
            prev_wk, prev_price = last_state.get(key, (None, None))

            for row in g.itertuples(index=False): # 按行遍历
                # -- initialize一下 -- 
                wk = int(row.wm_yr_wk)
                price = float(row.sell_price)
                dt = row.week_start_date

                # -- 如果历史价格为空 直接载入 只有第一次会有 --
                if prev_price is None:
                    # first observed price for this series -> not a "price change event"
                    prev_wk, prev_price = wk, price
                    continue

                #  -- 如果价格连续 那就更新时间 --
                if price == prev_price:
                    prev_wk = wk
                    continue
                
                '''
                    上面这两个的意思就是 没有价格没有改变 没有改变就不用管他 记录一下历史财务周就可
                '''
                # -- 到这里 就是 如果价格发生改变了 --
                abs_change = price - prev_price # 得到差值
                pct_change = abs_change / prev_price if prev_price != 0 else float("inf") # 计算比例

                # -- 这里的逻辑是一个标准 工程化来说 你可以随时改变 超过了多少 或者加上规则 怎么才算一次调价
                if abs(abs_change) >= args.min_abs_change and abs(pct_change) >= args.min_pct_change:
                    # --确定算调价就记录一下 -- 
                    events.append({
                        "store_id": st,
                        "item_id": it,
                        "event_wm_yr_wk": wk,
                        "event_date": dt,            # first day of that week in this chunk
                        "old_price": prev_price,
                        "new_price": price,
                        "abs_change": abs_change,
                        "pct_change": pct_change,
                        "direction": "up" if abs_change > 0 else "down",
                    })

                # -- 更新一下历史价格 --
                prev_wk, prev_price = wk, price

            # -- 更新last state的字典 保持跨chunk的连续性 -- 
            last_state[key] = (prev_wk, prev_price)

        # -- 将所有的个体的调价情况变成df
        df_events = pd.DataFrame(events)
        # -- 极端情况兜底 --
        if df_events.empty:
            print("[INFO] no price change events found in this part.")
            continue

        # --附加一些meta信息 --
        df_events = df_events.merge(meta, on=["store_id", "item_id"], how="left")

        # -- 整理一下字段的顺序 --
        df_events = df_events[[
            "store_id", "state_id",
            "item_id", "dept_id", "cat_id",
            "event_wm_yr_wk", "event_date",
            "old_price", "new_price",
            "abs_change", "pct_change", "direction",
        ]].sort_values(["store_id", "item_id", "event_wm_yr_wk"]).reset_index(drop=True)

        print(f"[OK] events in {part_path.name}: {len(df_events)}")
        first_write = _write_events(df_events, out_path, args.out_format, first_write)

    print("[DONE] price event extraction finished.")
    if args.out_format == "csv":
        print(f"[OUT] {out_path}")
    else:
        print(f"[OUT] parquet parts written under: {out_path.parent / (out_path.stem + '_parts')}")


if __name__ == "__main__":
    main()
