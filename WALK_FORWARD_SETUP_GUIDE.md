# Walk-Forward验证器 - 运行前准备指南

## 问题总结

用户遇到的错误：
```
✗ portfolio_signal_weighted_sma_5_20_keltner_10_10_1_5_etc: 窗口内无数据 (20200501 ~ 20201031)
```

这个错误说明某个组合的策略在指定窗口内没有可用的数据。

---

## 必备准备工作

### 第1步：确保所有基础策略都有完整期间的daily_values文件

**需要的文件格式：**
```
daily_values_{strategy_name}_d1_20200101_20241231.csv
```

其中：
- `{strategy_name}` = 策略名称（如 `rsi_reversal`, `sma_cross`, `keltner_10_10_1_5` 等）
- `d1` = 日线时间周期（walk_forward_validator默认使用d1）
- `20200101_20241231` = 完整的数据范围（从2020年1月1日到2024年12月31日）

**当前状态：**
✓ 你已有 **198个策略** 的完整daily_values文件（d1, 20200101-20241231）

**如何生成这些文件（如果需要）：**

```bash
# 运行所有策略的完整期间回测（2020-2024）
# 这会生成daily_values_*_d1_20200101_20241231.csv文件

# 方法1：使用批处理脚本
bash run_all_strategies_2020_2024.sh

# 方法2：逐个策略运行
python bt_main.py --atom rsi_reversal --timeframe d1 --start 2020-01-01 --end 2024-12-31
python bt_main.py --atom sma_cross --timeframe d1 --start 2020-01-01 --end 2024-12-31
# ...对所有策略重复
```

---

## Walk-Forward验证器工作流程

### 用到的基础策略范围

walk_forward_validator会：

1. **扫描backtest_results目录** 查找所有 `daily_values_*_d1_20200101_20241231.csv` 文件
2. **提取策略列表** 从文件名中（示例：`daily_values_rsi_reversal_d1_20200101_20241231.csv` → `rsi_reversal`）
3. **计算指标** 从daily_values文件中计算夏普、收益、回撤等
4. **质量筛选** 按夏普≥0.5、收益≥1%等条件筛选高质量策略
5. **相关性分析** 计算策略间相关性，选出低相关的组合
6. **权重优化** 为每个组合生成5种权重方案

### 具体流程图

```
完整期间数据 (2020-2024)
         ↓
daily_values_*.csv 文件 (198个策略)
         ↓
【窗口1】 【窗口2】 【窗口3】...【窗口N】
   ↓         ↓         ↓          ↓
[训练期]  [训练期]  [训练期]   [训练期]
  │         │         │          │
  ├→ 质量筛选 ← 使用这198个策略的数据
  ├→ 相关性分析
  ├→ 权重优化
  └→ 选出Top10组合
         ↓
   [测试期验证]
```

---

## 关键参数说明

### 数据范围参数

```bash
python walk_forward_validator.py \
  --data-start 20200101 \     # 完整数据开始 - 用于查找文件
  --data-end 20241231 \       # 完整数据结束 - 用于查找文件
  --timeframe d1              # 时间周期 - 必须与daily_values文件匹配
```

**重要：**
- `--data-start` 和 `--data-end` 必须与你的 daily_values_*.csv 文件名中的日期匹配
- 当前文件都是 `_20200101_20241231.csv`，所以必须用这些日期

### 窗口参数

```bash
python walk_forward_validator.py \
  --train-months 12 \         # 每个训练窗口12个月
  --test-months 6 \           # 每个测试窗口6个月
  --step-months 6             # 每6个月滚动一次
```

这会生成：
```
窗口1: 训练 2020-01~2020-12 → 测试 2021-01~2021-06
窗口2: 训练 2020-07~2021-06 → 测试 2021-07~2021-12
窗口3: 训练 2021-01~2021-12 → 测试 2022-01~2022-06
...以此类推
```

---

## "窗口内无数据"错误排查

### 错误原因

错误信息格式：
```
✗ portfolio_signal_weighted_{strategy_combo}_etc: 窗口内无数据 ({window_start} ~ {window_end})
```

**可能的原因：**

1. **某个基础策略的daily_values文件缺失**
   - 例如：如果组合使用了 `sma_5_20` 但缺少 `daily_values_sma_5_20_d1_20200101_20241231.csv`

2. **daily_values文件格式错误**
   - 缺少必要列：`datetime`, `portfolio_value`, `daily_return`
   - datetime格式不是标准的 YYYY-MM-DD HH:MM:SS

3. **窗口时间段在daily_values数据范围之外**
   - daily_values文件中没有该窗口期间的数据
   - 应该不会发生，因为daily_values是20200101-20241231，窗口应该在此范围内

### 排查步骤

**步骤1：检查是否有missing的策略**

```bash
# 从质量筛选后的组合中提取所有策略名称
# 检查每个策略是否有对应的daily_values文件

# 例如，错误中提到 sma_5_20 和 keltner_10_10_1_5
ls backtest_results/daily_values_sma_5_20_d1_20200101_20241231.csv
ls backtest_results/daily_values_keltner_10_10_1_5_d1_20200101_20241231.csv

# 如果文件不存在，需要运行对应的回测
python bt_main.py --atom sma_5_20 --timeframe d1 --start 2020-01-01 --end 2024-12-31
```

**步骤2：检查daily_values文件格式**

```bash
# 查看第一行确认有正确的列
head -2 backtest_results/daily_values_sma_5_20_d1_20200101_20241231.csv

# 应该看到类似这样的：
# datetime,portfolio_value,daily_return,cumulative_return,...
# 2020-01-02 09:30:00,99500.25,-0.0050,...
```

**步骤3：检查daily_values数据范围**

```bash
# 查看数据的开始和结束日期
head -2 backtest_results/daily_values_sma_5_20_d1_20200101_20241231.csv
tail -2 backtest_results/daily_values_sma_5_20_d1_20200101_20241231.csv

# 应该包含 2020-01-02 到 2024-12-31 的数据
```

---

## 推荐的完整运行流程

### 方案A：使用已有的198个策略（推荐）

```bash
# 1. 生成walk_forward分析报告（自动并行执行）
python walk_forward_validator.py \
  --timeframe d1 \
  --train-months 12 \
  --test-months 6 \
  --step-months 6 \
  --top-n 10

# 2. 查看结果
cat backtest_results/walk_forward/portfolio_robustness_d1.csv

# 3. 选择稳健的组合（过拟合指数<0.3）并进一步验证
```

### 方案B：如果缺少某些策略

```bash
# 1. 先运行缺失的策略
python bt_main.py --atom missing_strategy --timeframe d1 --start 2020-01-01 --end 2024-12-31

# 2. 再运行walk_forward_validator
python walk_forward_validator.py --timeframe d1 --train-months 12 --test-months 6 --step-months 6
```

### 方案C：添加新的d1周期daily_values

```bash
# 如果你有其他日期范围的daily_values（如20240101-20241231）
# 需要重新运行回测生成完整期间的文件

python bt_main.py --atom all_strategies --timeframe d1 --start 2020-01-01 --end 2024-12-31
```

---

## 性能优化建议

walk_forward_validator现在支持自动并行化：

```bash
# 自动检测CPU核数，使用 (核数-1) 个进程
python walk_forward_validator.py --timeframe d1

# 指定并行进程数
python walk_forward_validator.py --timeframe d1 --workers 4

# 串行执行（调试用）
python walk_forward_validator.py --timeframe d1 --workers 1
```

**预期性能：**
- 单核串行：30-60分钟（6个窗口 × 10个组合 × 测试）
- 4核并行：8-15分钟（每个窗口的组合并行测试）
- 8核并行：4-8分钟

---

## 快速参考

### 验证daily_values文件是否完整

```bash
# 计数d1周期的完整数据文件
ls backtest_results/daily_values_*_d1_20200101_20241231.csv 2>/dev/null | wc -l

# 应该输出 100+ （你目前有198个）

# 列出所有可用的策略
ls backtest_results/daily_values_*_d1_20200101_20241231.csv | \
  sed 's/.*daily_values_//;s/_d1.*//' | sort
```

### 检查最近一次walk_forward的结果

```bash
# 查看组合稳健性排名
cat backtest_results/walk_forward/portfolio_robustness_d1.csv | head -15

# 查看窗口汇总
cat backtest_results/walk_forward/window_summary_d1.csv

# 查看详细的过拟合指数分布
awk -F, 'NR>1 {print $NF}' backtest_results/walk_forward/portfolio_robustness_d1.csv | sort -n
```

---

## 总结

**你目前的状态：**
✓ 有198个基础策略的完整daily_values文件
✓ walk_forward_validator.py已创建并支持并行化
✓ 可以直接运行验证

**立即可运行的命令：**
```bash
python walk_forward_validator.py \
  --timeframe d1 \
  --train-months 12 \
  --test-months 6 \
  --step-months 6
```

**如果遇到"窗口内无数据"错误：**
1. 检查缺失的策略daily_values文件
2. 对缺失的策略运行 `bt_main.py` 回测
3. 重新运行walk_forward_validator

**预期结果：**
- `backtest_results/walk_forward/portfolio_robustness_d1.csv` - 组合稳健性排名
- `backtest_results/walk_forward/window_summary_d1.csv` - 窗口级统计
- `backtest_results/walk_forward/walk_forward_details_d1.csv` - 详细结果
