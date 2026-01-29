# 6.1 生成策略表
python scripts/6_1_build_pricing_policy_table.py --y_mode log1p_sales

# 6.2 生成按周的调价建议
python scripts/6_2_run_pricing_engine.py --lookback_weeks 4 --top_n 200
