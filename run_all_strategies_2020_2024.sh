#!/bin/bash
# 滚动窗口回测脚本
# 对所有策略进行 6 个月滚动窗口回测（每月滚动一次）
# 时间范围: 2017-01-01 到 2025-12-31
# 用法: bash run_all_strategies_2020_2024.sh

# ============================================================================
# 配置参数
# ============================================================================
WINDOW_START_DATE="2017-01-01"
WINDOW_END_DATE="2025-12-31"
WINDOW_MONTHS=6           # 6 个月的投资窗口
ROLLING_MONTHS=1          # 每月滚动一次
PARALLEL_JOBS=4           # 并发执行数量
TIMEFRAMES=("h4" "d1")  # 所有时间框架

echo "================================================================"
echo "滚动窗口回测 - 所有策略和时间框架"
echo "================================================================"
echo "数据范围: $WINDOW_START_DATE 至 $WINDOW_END_DATE"
echo "窗口大小: $WINDOW_MONTHS 个月"
echo "滚动步长: $ROLLING_MONTHS 个月"
echo "时间框架: ${TIMEFRAMES[@]}"
echo "并发数量: $PARALLEL_JOBS"
echo "================================================================"
echo ""

# 创建结果目录
mkdir -p backtest_results

# 生成所有滚动窗口的日期对（使用Python处理日期以确保跨平台兼容）
generate_rolling_windows() {
    local start_date=$1
    local end_date=$2
    local window_months=$3
    local rolling_months=$4

    python3 - "$start_date" "$end_date" "$window_months" "$rolling_months" << 'PYTHON_EOF'
import sys
from datetime import datetime

def add_months(date, months):
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return datetime(year, month, day)

start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
end = datetime.strptime(sys.argv[2], "%Y-%m-%d")
window_months = int(sys.argv[3])
rolling_months = int(sys.argv[4])

current = start
while current < end:
    window_end = add_months(current, window_months)
    if window_end > end:
        window_end = end

    print(f"{current.strftime('%Y-%m-%d')} {window_end.strftime('%Y-%m-%d')}")
    current = add_months(current, rolling_months)
PYTHON_EOF
}

# 并发控制函数
run_with_limit() {
    local timeframe=$1
    local strategy=$2
    local window_start=$3
    local window_end=$4

    # 等待直到有可用的并发槽位
    while [ $(jobs -r | wc -l) -ge $PARALLEL_JOBS ]; do
        sleep 0.5
    done

    # 后台运行策略
    python bt_main.py --start "$window_start" --end "$window_end" --timeframe "$timeframe" --atom "$strategy" --no-plot &
}

# ============================================================================
# 所有策略定义
# ============================================================================

INTRADAY=(
    "orb" "orb_15min" "orb_30min" "orb_60min" "orb_30min_no_close" "orb_45min" "orb_aggressive" "orb_conservative"
    "intraday_mom" "intraday_mom_0_5" "intraday_mom_1_0" "intraday_mom_1_5" "intraday_mom_2_0" "intraday_mom_0_3"
    "intraday_mom_aggressive" "intraday_mom_conservative" "intraday_mom_moderate"
    "intraday_reversal" "intraday_rev_1_5" "intraday_rev_1_0" "intraday_rev_2_0" "intraday_rev_aggressive" "intraday_rev_conservative"
    "vwap_reversion" "vwap_rev_1_0" "vwap_rev_1_5" "vwap_rev_2_0" "vwap_rev_aggressive" "vwap_rev_conservative"
)

TREND=(
    "sma_cross" "sma_5_20" "sma_10_30" "sma_20_60"
    "triple_ma" "triple_ma_5_20_50" "triple_ma_10_30_60" "triple_ma_8_21_55" "triple_ma_12_26_52"
    "adx_trend" "adx_14_25" "adx_14_30" "adx_14_20" "adx_21_25" "adx_10_25"
    "macd_trend"
)

MEANREV=(
    "rsi_reversal"
    "boll_mr" "boll_mr_20_2" "boll_mr_20_2_5" "boll_mr_20_1_5" "boll_mr_30_2" "boll_mr_10_2" "boll_mr_strict"
    "cci_channel" "cci_20_100" "cci_20_150" "cci_20_80" "cci_14_100" "cci_30_100" "cci_strict"
)

BREAKOUT=(
    "donchian_channel" "donchian_20_10" "donchian_55_20" "donchian_20_20" "donchian_10_5" "donchian_5_3"
    "donchian_40_15" "donchian_turtle_sys1" "donchian_turtle_sys2" "donchian_aggressive" "donchian_conservative"
    "keltner_channel" "keltner_20_10_1_5" "keltner_20_10_2_0" "keltner_20_10_1_0" "keltner_20_14_1_5"
    "keltner_30_10_1_5" "keltner_10_10_1_5" "keltner_tight"
    "atr_breakout" "atr_breakout_20_14_2" "atr_breakout_20_14_3" "atr_breakout_20_14_1_5" "atr_breakout_20_10_2"
    "atr_breakout_50_14_2" "atr_breakout_10_14_2" "atr_breakout_aggressive" "atr_breakout_conservative"
    "vol_breakout" "vol_breakout_14_2" "vol_breakout_14_2_5" "vol_breakout_14_1_5" "vol_breakout_20_2"
    "vol_breakout_10_2" "vol_breakout_10_3" "vol_breakout_aggressive" "vol_breakout_conservative"
    "new_hl" "new_hl_20" "new_hl_50" "new_hl_100" "new_hl_250" "new_hl_10" "new_hl_5" "new_hl_aggressive" "new_hl_conservative"
)

VOLATILITY=(
    "const_vol" "const_vol_10" "const_vol_15" "const_vol_20" "const_vol_conservative" "const_vol_aggressive"
    "vol_expansion" "vol_expansion_standard" "vol_expansion_sensitive" "vol_expansion_conservative" "vol_expansion_short" "vol_expansion_long"
    "vol_regime" "vol_regime_standard" "vol_regime_sensitive" "vol_regime_conservative" "vol_regime_short" "vol_regime_long"
)

CLASSIC=(
    "turtle" "turtle_sys1" "turtle_sys1_conservative" "turtle_sys1_aggressive" "turtle_sys2" "turtle_es" "turtle_mnq"
)

# 组合所有策略
ALL_STRATEGIES=("${INTRADAY[@]}" "${TREND[@]}" "${MEANREV[@]}" "${BREAKOUT[@]}" "${VOLATILITY[@]}" "${CLASSIC[@]}")

# ============================================================================
# 主回测循环
# ============================================================================

# 生成所有滚动窗口
WINDOWS=$(generate_rolling_windows "$WINDOW_START_DATE" "$WINDOW_END_DATE" "$WINDOW_MONTHS" "$ROLLING_MONTHS")
WINDOW_COUNT=$(echo "$WINDOWS" | wc -l)

echo "生成的滚动窗口数量: $WINDOW_COUNT 个"
echo ""

WINDOW_INDEX=0

# 对每个滚动窗口执行回测
while IFS=' ' read -r WINDOW_START WINDOW_END; do
    WINDOW_INDEX=$((WINDOW_INDEX + 1))
    echo "=========================================="
    echo "处理窗口 $WINDOW_INDEX/$WINDOW_COUNT: $WINDOW_START 至 $WINDOW_END"
    echo "=========================================="

    # 对每个时间框架
    for timeframe in "${TIMEFRAMES[@]}"; do
        echo "  时间框架: $timeframe"

        # 对每个策略
        for strategy in "${ALL_STRATEGIES[@]}"; do
            run_with_limit "$timeframe" "$strategy" "$WINDOW_START" "$WINDOW_END"
        done
    done

    echo "窗口 $WINDOW_INDEX 的所有任务已提交，等待完成..."
    wait  # 等待当前窗口的所有任务完成
    echo ""

done <<< "$WINDOWS"

# ============================================================================
echo "================================================================"
echo "所有策略回测完成!"
echo "================================================================"
echo ""
echo "回测统计信息:"
echo "  策略类型数量:"
echo "    - 日内策略: ${#INTRADAY[@]} 个"
echo "    - 趋势策略: ${#TREND[@]} 个"
echo "    - 均值回归: ${#MEANREV[@]} 个"
echo "    - 突破策略: ${#BREAKOUT[@]} 个"
echo "    - 波动率策略: ${#VOLATILITY[@]} 个"
echo "    - 经典系统: ${#CLASSIC[@]} 个"
echo "    ─────────────────"
echo "    总策略数: ${#ALL_STRATEGIES[@]} 个"
echo ""
echo "  时间框架数: ${#TIMEFRAMES[@]} 个"
echo "  滚动窗口数: $WINDOW_COUNT 个"
echo ""
TOTAL_BACKTESTS=$((${#ALL_STRATEGIES[@]} * ${#TIMEFRAMES[@]} * $WINDOW_COUNT))
echo "  总回测次数: ${#ALL_STRATEGIES[@]} 策略 × ${#TIMEFRAMES[@]} 时间框架 × $WINDOW_COUNT 窗口 = $TOTAL_BACKTESTS 次"
echo ""
echo "结果文件位置:"
echo "  backtest_results/ 目录下的各类文件:"
echo "    - daily_values_*.csv - 每日投资组合价值"
echo "    - trades_*.csv - 交易记录"
echo ""
