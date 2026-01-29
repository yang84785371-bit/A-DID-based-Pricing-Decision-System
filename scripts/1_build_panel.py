# 1_build_panel.py
'''
    该文件是为了将sale calendar 以及price的表格merge在一起
    但为了防止oom 我们进行的是一个chunk分块
'''
'''
    calendar：['date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'] 时间表 以及当前日期一些基本的属性 每一行代表一天
    price ：store_id  item_id  wm_yr_wk  sell_pric 每一行也代表一天
    sale ：id        item_id    dept_id   cat_id store_id state_id  d_1  d_2  d_3  d_4  d_5  d_6  d_7 .....d_end  一行 = 某个商品在某个门店的若干天的「销量（件数）
    将三个表连接在一起 我们得到 id          item_id      dept_id     cat_id store_id state_id     d  sales        date  wm_yr_wk    weekday  wday  month  year event_name_1 event_type_1 event_name_2 event_type_2  snap_CA  snap_TX  snap_WI  sell_price
    就是将sale melt过来 然后按照时间连接calendar和price 但由于内存吃不消 所以我们要分分块 这里面属于是 天数*商品id*门店id 其中 state_id 和dept id没有独立性 不用管
    这里state被store指定 dept被item指定 也就是没有独立性的 可以不管 只不过是作为group的依据罢了

'''
import argparse
import math
from pathlib import Path

import pandas as pd

# -- 命令行参数 --
def parse_args():
    p = argparse.ArgumentParser("Build full panel (streaming, chunked) for M5 pricing project")

    # -- 这里的参数就是关于raw data的 --
    p.add_argument("--data_dir", type=str, required=True, help="Directory containing raw csv files") # raw数据地址
    p.add_argument("--sales_file", type=str, default="sales_train_validation.csv") # 销量文件
    p.add_argument("--calendar_file", type=str, default="calendar.csv") # 日历
    p.add_argument("--price_file", type=str, default="sell_prices.csv") # 价格文件

    # -- 关于输出的 --
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for partition files")
    p.add_argument("--out_format", type=str, default="parquet", choices=["parquet", "csv"])

    # -- 分块与时间控制 防止oom --
    p.add_argument("--chunk_days", type=int, default=56, help="How many d_* columns to process per chunk (e.g., 28/56/112)") # 每次处理多少天的数据
    p.add_argument("--start_d", type=int, default=1, help="Start day index (e.g., 1)") # start day
    p.add_argument("--end_d", type=int, default=-1, help="End day index (e.g., 1913). -1 means auto-detect max") # end day

    # -- 字段控制 --
    p.add_argument("--keep_calendar_cols", type=str, default="date,wm_yr_wk,weekday,wday,month,year,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI",
                   help="Comma-separated calendar columns to keep (must include date, wm_yr_wk)")
    # -- id列 --
    p.add_argument("--id_cols", type=str, default="id,item_id,dept_id,cat_id,store_id,state_id",
                   help="Comma-separated id columns in sales file")

    return p.parse_args()

# -- 找一个最大的d -- 
def detect_max_d(sales_path: Path) -> int:
    # Read only header to find max d_*
    cols = pd.read_csv(sales_path, nrows=0).columns.tolist()
    d_cols = [c for c in cols if c.startswith("d_")]
    # d_1 ... d_1913
    return max(int(c.split("_")[1]) for c in d_cols)


def main():
    # -- 命令行参数 --
    args = parse_args()
    data_dir = Path(args.data_dir)
    sales_path = data_dir / args.sales_file
    cal_path = data_dir / args.calendar_file
    price_path = data_dir / args.price_file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    id_cols = [x.strip() for x in args.id_cols.split(",") if x.strip()] #得到id的列 
    cal_keep_cols = [x.strip() for x in args.keep_calendar_cols.split(",") if x.strip()] # 日历保留的字段

    # -- 载入日历 --
    calendar = pd.read_csv(cal_path, usecols=["d"] + cal_keep_cols)
    # -- 确保有date和财务周 --
    if "date" not in calendar.columns or "wm_yr_wk" not in calendar.columns:
        raise ValueError("calendar must include 'date' and 'wm_yr_wk' in keep_calendar_cols")

    calendar["d"] = calendar["d"].astype(str) # calendar的d就是表内的顺序

    # -- 这里需要用到的是门店*物品*财务周*价格 --
    prices = pd.read_csv(price_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"]) 

    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")

    # -- 做一个索引 主要是为了更快的连接表 --
    prices = prices.set_index(["store_id", "item_id", "wm_yr_wk"]).sort_index()

    # -- 确定最大的d的范围 --
    max_d = detect_max_d(sales_path) if args.end_d == -1 else args.end_d 
    start_d = args.start_d
    end_d = max_d
    # -- 兜底 --
    if start_d < 1 or end_d < start_d:
        raise ValueError(f"Invalid d range: start_d={start_d}, end_d={end_d}")

    total_days = end_d - start_d + 1 # 一共有多少天
    num_chunks = math.ceil(total_days / args.chunk_days) # 确定分块

    print(f"[INFO] d range: d_{start_d} ... d_{end_d}  (total_days={total_days})")
    print(f"[INFO] chunk_days={args.chunk_days} => num_chunks={num_chunks}")
    print(f"[INFO] out_format={args.out_format} out_dir={out_dir}")

    # 
    for chunk_idx in range(num_chunks):
        chunk_start = start_d + chunk_idx * args.chunk_days # 块的开始
        chunk_end = min(end_d, chunk_start + args.chunk_days - 1) # 块的结束
        d_cols = [f"d_{i}" for i in range(chunk_start, chunk_end + 1)] # 取出需要的内部编码

        usecols = id_cols + d_cols # 这次chunk在sale表中取数的行/日期
        print(f"[CHUNK {chunk_idx+1}/{num_chunks}] reading sales cols: d_{chunk_start}..d_{chunk_end} ({len(d_cols)} days)")

        sales_chunk = pd.read_csv(sales_path, usecols=usecols)
        # -- 将其拉平成可用于分析的时间序列数据 --
        long_chunk = sales_chunk.melt(
            id_vars=id_cols, #哪些行不动
            value_vars=d_cols,# 哪些行也被拉平
            var_name="d", # 原来列名现在叫什么
            value_name="sales", # 原来列名的值叫什么
        )

        # -- 连接 -- 
        long_chunk = long_chunk.merge(calendar, on="d", how="left")

        # -- 确保类型 --
        long_chunk["wm_yr_wk"] = long_chunk["wm_yr_wk"].astype("int32")

        # -- 将["store_id", "item_id", "wm_yr_wk"]变成一个复合键 --
        key_index = pd.MultiIndex.from_frame(long_chunk[["store_id", "item_id", "wm_yr_wk"]])
        long_chunk["sell_price"] = prices.reindex(key_index)["sell_price"].to_numpy()

        # -- 排序 --
        long_chunk = long_chunk.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        # -- 分区 --
        part_name = f"part_{chunk_idx:04d}"
        if args.out_format == "parquet":
            out_path = out_dir / f"{part_name}.parquet"
            long_chunk.to_parquet(out_path, index=False)
        else:
            out_path = out_dir / f"{part_name}.csv"
            long_chunk.to_csv(out_path, index=False)

        print(f"[OK] wrote {out_path}  shape={long_chunk.shape}")

        # -- 清除内存 --
        del sales_chunk, long_chunk

    print("[DONE] All chunks written.")
    print("Tip: you can later read all parts and concatenate if needed (prefer parquet scan).")


if __name__ == "__main__":
    main()

