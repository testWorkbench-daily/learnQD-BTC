"""
快速修复数据 - 创建可用的连续合约数据
处理负价格
支持 NQ 和 BTC 期货数据处理
"""

import pandas as pd
import numpy as np

def clean_btc_data(input_path, output_path):
    """
    清理 BTC 期货数据
    
    步骤：
    1. 筛选 BTC 期货合约（不含价差合约）
    2. 同一分钟内多条数据，保留成交量最大的
    3. 移除负价格和异常低价
    4. 检测并移除孤立尖峰
    5. 保存清理后的 OHLCV 数据
    """
    print("="*60)
    print("BTC 期货数据清理工具")
    print("="*60)
    
    print("\n步骤1: 加载数据...")
    df = pd.read_csv(input_path, parse_dates=['ts_event'])
    print(f"原始数据: {len(df):,} 行")
    print(f"时间范围: {df['ts_event'].min()} 到 {df['ts_event'].max()}")
    print(f"所有唯一symbol数量: {df['symbol'].nunique()}")
    
    # 步骤2: 筛选 BTC 期货合约（不含价差）
    print("\n步骤2: 筛选 BTC 期货合约（不含价差）...")
    before = len(df)
    
    # 筛选条件：symbol包含'BTC'且不含'-'（排除价差合约）
    df = df[
        df['symbol'].str.contains('BTC', na=False) & 
        ~df['symbol'].str.contains('-', na=False)
    ].copy()
    
    btc_symbols = sorted(df['symbol'].unique())
    print(f"✓ 找到 {len(btc_symbols)} 个 BTC 期货合约")
    print(f"合约列表: {', '.join(btc_symbols[:10])}{'...' if len(btc_symbols) > 10 else ''}")
    print(f"保留数据: {len(df):,} 行 (占比 {len(df)/before*100:.2f}%)")
    
    if len(df) == 0:
        print("\n❌ 错误: 未找到任何 BTC 期货数据！")
        return None
    
    # 步骤3: 同一分钟内多条数据，保留成交量最大的
    print("\n步骤3: 处理同一分钟内的重复数据...")
    before_dedup = len(df)
    
    # 将时间截断到分钟级别
    df['minute'] = df['ts_event'].dt.floor('min')
    
    # 检查重复情况
    duplicates_per_minute = df.groupby('minute').size()
    duplicate_minutes = duplicates_per_minute[duplicates_per_minute > 1]
    
    if len(duplicate_minutes) > 0:
        print(f"发现 {len(duplicate_minutes):,} 个分钟存在多条数据")
        print(f"最多的分钟有 {duplicate_minutes.max()} 条数据")
        
        # 显示一些重复的例子
        print("\n重复数据示例（前3个分钟）：")
        print("-" * 100)
        sample_minutes = duplicate_minutes.head(3).index.tolist()
        for minute in sample_minutes:
            minute_data = df[df['minute'] == minute][['ts_event', 'symbol', 'close', 'volume']].copy()
            print(f"\n{minute}:")
            for _, row in minute_data.iterrows():
                print(f"  时间: {row['ts_event']}, 合约: {row['symbol']}, 收盘价: {row['close']:.2f}, 成交量: {row['volume']:,}")
        print("-" * 100)
        
        # 对每个分钟，保留成交量最大的那条记录
        idx_max_volume = df.groupby('minute')['volume'].idxmax()
        df = df.loc[idx_max_volume].copy()
        
        removed_dups = before_dedup - len(df)
        print(f"\n✓ 已为每个分钟保留成交量最大的记录")
        print(f"移除了 {removed_dups:,} 行重复数据 ({removed_dups/before_dedup*100:.2f}%)")
    else:
        print("✓ 未发现同一分钟内的重复数据")
        removed_dups = 0
    
    # 删除临时的minute列
    df = df.drop('minute', axis=1)
    
    # 按时间排序
    df = df.sort_values('ts_event').reset_index(drop=True)
    
    # 步骤4: 移除明显错误的数据
    print("\n步骤4: 移除数据错误...")
    print(f"当前价格范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
    
    before = len(df)
    # BTC期货价格通常在几百到几万美元，设置最低价格为10
    df = df[
        (df['open'] > 10) & 
        (df['high'] > 10) & 
        (df['low'] > 10) & 
        (df['close'] > 10)
    ].copy()
    removed_errors = before - len(df)
    print(f"移除了 {removed_errors:,} 行错误数据 ({removed_errors/before*100:.2f}%)")
    
    # 步骤5: 检测并移除孤立尖峰（V型异常）
    print("\n步骤5: 检测并移除孤立尖峰（V型异常）...")
    print("说明: 检测价格大幅跳变后又立刻回归的异常数据（保留真实的持续性市场变化）")
    
    before = len(df)
    # 计算价格变化
    df['price_change'] = df['close'].pct_change()
    df['next_price_change'] = df['price_change'].shift(-1)
    df['prev_close'] = df['close'].shift(1)
    df['next_close'] = df['close'].shift(-1)
    
    # 孤立尖峰检测逻辑：
    # 1. 当前变化超过5%（上涨或下跌）
    # 2. 下一个tick的变化方向相反且也较大（至少3%）
    # 3. 最终价格回到原来的±2%范围内
    abnormal_spikes = df[
        (df['price_change'].abs() > 0.05) &  # 大幅变化
        (df['next_price_change'].abs() > 0.03) &  # 下一个也有较大变化
        (df['price_change'] * df['next_price_change'] < 0) &  # 方向相反
        ((df['next_close'] - df['prev_close']).abs() / df['prev_close'] < 0.02)  # 回到原位±2%
    ].copy()
    
    if len(abnormal_spikes) > 0:
        print(f"\n发现 {len(abnormal_spikes):,} 条孤立尖峰记录（价格跳变后又快速回归）：")
        print("-" * 120)
        print(f"{'时间':<25} {'前一价':>10} {'尖峰价':>10} {'后一价':>10} {'第一跳':>10} {'回归跳':>10} {'类型':<12}")
        print("-" * 120)
        
        for idx, row in abnormal_spikes.head(10).iterrows():
            time_str = str(row['ts_event'])[:19]
            prev_close = row['prev_close']
            curr_close = row['close']
            next_close = row['next_close']
            jump1 = row['price_change'] * 100
            jump2 = row['next_price_change'] * 100
            spike_type = "向上尖峰↑" if jump1 > 0 else "向下尖峰↓"
            print(f"{time_str:<25} {prev_close:>10.2f} {curr_close:>10.2f} {next_close:>10.2f} {jump1:>9.2f}% {jump2:>9.2f}% {spike_type:<12}")
        
        if len(abnormal_spikes) > 10:
            print(f"... （还有 {len(abnormal_spikes)-10} 条未显示）")
        print("-" * 120)
        print("✓ 这些是典型的V型异常（可能是脏数据或瞬间错误报价），将被移除")
    else:
        print("✓ 未发现孤立尖峰异常")
    
    # 移除孤立尖峰
    df = df[~df.index.isin(abnormal_spikes.index)].copy()
    
    # 删除临时列
    df = df.drop(['price_change', 'next_price_change', 'prev_close', 'next_close'], axis=1)
    
    removed_spikes = before - len(df)
    print(f"\n移除了 {removed_spikes:,} 行孤立尖峰数据 ({removed_spikes/before*100:.2f}%)")
    print(f"剩余数据: {len(df):,} 行")
    
    # 检查是否还有大幅变化（可能是真实的市场事件）
    df_temp = df.copy()
    df_temp['pct_change'] = df_temp['close'].pct_change()
    large_changes = df_temp[df_temp['pct_change'].abs() > 0.05]
    if len(large_changes) > 0:
        print(f"\n⚠️  数据中仍有 {len(large_changes)} 条大于5%的变化（已保留，可能是真实市场事件）")
    
    # 步骤6: 保存清理后的数据
    print("\n步骤6: 保存清理后的数据...")
    
    output_df = df[['ts_event', 'open', 'high', 'low', 'close', 'volume']].copy()
    output_df.to_csv(output_path, index=False)
    
    print(f"\n清理后数据已保存到: {output_path}")
    print(f"最终数据: {len(output_df):,} 行")
    print(f"时间范围: {output_df['ts_event'].min()} 到 {output_df['ts_event'].max()}")
    print(f"价格范围: {output_df['close'].min():.2f} 到 {output_df['close'].max():.2f}")
    
    # 生成统计报告
    print("\n" + "="*60)
    print("数据质量报告")
    print("="*60)
    original_count = pd.read_csv(input_path).shape[0]
    print(f"数据完整性: {len(output_df)/original_count*100:.2f}%")
    print(f"筛选BTC期货: 保留 {len(output_df):,} 行")
    print(f"移除重复数据（同一分钟保留最大成交量）: {removed_dups:,} 行")
    print(f"移除错误数据: {removed_errors:,} 行")
    print(f"移除孤立尖峰数据: {removed_spikes:,} 行")

    returns_stats = output_df['close'].pct_change()
    print(f"\n收益率统计:")
    print(f"  均值: {returns_stats.mean()*100:.4f}%")
    print(f"  标准差: {returns_stats.std()*100:.4f}%")
    print(f"  最大: {returns_stats.max()*100:.2f}%")
    print(f"  最小: {returns_stats.min()*100:.2f}%")
    print(f"  95分位: {returns_stats.quantile(0.95)*100:.2f}%")
    print(f"  5分位: {returns_stats.quantile(0.05)*100:.2f}%")
    
    print("\n" + "="*60)
    print("✅ BTC 数据清理完成！")
    print("="*60)
    print(f"\n现在可以使用 {output_path} 进行回测")
    
    return output_path


def quick_clean_data(input_path, output_path):
    """
    快速清理数据，移除明显错误并创建连续合约
    
    步骤：
    1. 移除负价格和异常低价
    2. 同一分钟内多条数据，保留成交量最大的
    3. 检测并调整大幅跳空
    4. 保存清理后的数据
    """
    print("="*60)
    print("快速数据清理工具")
    print("="*60)
    
    print("\n步骤1: 加载数据...")
    df = pd.read_csv(input_path, parse_dates=['ts_event'])
    print(f"原始数据: {len(df):,} 行")
    print(f"时间范围: {df['ts_event'].min()} 到 {df['ts_event'].max()}")
    print(f"价格范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
    
    # 步骤2: 移除明显错误的数据
    print("\n步骤2: 移除数据错误...")
    
    # 移除负价格和极端低价（NQ正常价格应该在几百到几万）
    before = len(df)
    df = df[
        (df['open'] > 100) & 
        (df['high'] > 100) & 
        (df['low'] > 100) & 
        (df['close'] > 100)
    ].copy()
    removed = before - len(df)
    print(f"移除了 {removed:,} 行错误数据 ({removed/before*100:.2f}%)")
    
    # 步骤2.5: 同一分钟内多条数据，保留成交量最大的
    print("\n步骤2.5: 处理同一分钟内的重复数据...")
    before_dedup = len(df)
    removed_dups = 0  # 初始化去重移除数
    
    # 将时间截断到分钟级别
    df['minute'] = df['ts_event'].dt.floor('min')
    
    # 检查重复情况
    duplicates_per_minute = df.groupby('minute').size()
    duplicate_minutes = duplicates_per_minute[duplicates_per_minute > 1]
    
    if len(duplicate_minutes) > 0:
        print(f"发现 {len(duplicate_minutes):,} 个分钟存在多条数据")
        print(f"最多的分钟有 {duplicate_minutes.max()} 条数据")
        
        # 显示一些重复的例子
        print("\n重复数据示例（前5个分钟）：")
        print("-" * 100)
        sample_minutes = duplicate_minutes.head(5).index.tolist()
        for minute in sample_minutes:
            minute_data = df[df['minute'] == minute][['ts_event', 'close', 'volume']].copy()
            print(f"\n{minute}:")
            for _, row in minute_data.iterrows():
                print(f"  时间: {row['ts_event']}, 收盘价: {row['close']:.2f}, 成交量: {row['volume']:,}")
        print("-" * 100)
        
        # 对每个分钟，保留成交量最大的那条记录
        # 使用 idxmax 找到每个分钟成交量最大的行索引
        idx_max_volume = df.groupby('minute')['volume'].idxmax()
        df = df.loc[idx_max_volume].copy()
        
        removed_dups = before_dedup - len(df)
        print(f"\n✓ 已为每个分钟保留成交量最大的记录")
        print(f"移除了 {removed_dups:,} 行重复数据 ({removed_dups/before_dedup*100:.2f}%)")
    else:
        print("✓ 未发现同一分钟内的重复数据")
    
    # 删除临时的minute列
    df = df.drop('minute', axis=1)
    
    # 按时间排序
    df = df.sort_values('ts_event').reset_index(drop=True)
    
    # 步骤3: 检测并移除孤立尖峰（V型异常）
    print("\n步骤3: 检测并移除孤立尖峰（V型异常）...")
    print("说明: 检测价格大幅跳变后又立刻回归的异常数据（保留真实的持续性市场变化）")
    
    before = len(df)
    # 计算价格变化
    df['price_change'] = df['close'].pct_change()
    df['next_price_change'] = df['price_change'].shift(-1)
    df['prev_close'] = df['close'].shift(1)
    df['next_close'] = df['close'].shift(-1)
    
    # 孤立尖峰检测逻辑：
    # 1. 当前变化超过5%（上涨或下跌）
    # 2. 下一个tick的变化方向相反且也较大（至少3%）
    # 3. 最终价格回到原来的±2%范围内
    abnormal_spikes = df[
        (df['price_change'].abs() > 0.05) &  # 大幅变化
        (df['next_price_change'].abs() > 0.03) &  # 下一个也有较大变化
        (df['price_change'] * df['next_price_change'] < 0) &  # 方向相反
        ((df['next_close'] - df['prev_close']).abs() / df['prev_close'] < 0.02)  # 回到原位±2%
    ].copy()
    
    if len(abnormal_spikes) > 0:
        print(f"\n发现 {len(abnormal_spikes):,} 条孤立尖峰记录（价格跳变后又快速回归）：")
        print("-" * 120)
        print(f"{'时间':<25} {'前一价':>10} {'尖峰价':>10} {'后一价':>10} {'第一跳':>10} {'回归跳':>10} {'类型':<12}")
        print("-" * 120)
        
        for idx, row in abnormal_spikes.iterrows():
            time_str = str(row['ts_event'])[:19]
            prev_close = row['prev_close']
            curr_close = row['close']
            next_close = row['next_close']
            jump1 = row['price_change'] * 100
            jump2 = row['next_price_change'] * 100
            spike_type = "向上尖峰↑" if jump1 > 0 else "向下尖峰↓"
            print(f"{time_str:<25} {prev_close:>10.2f} {curr_close:>10.2f} {next_close:>10.2f} {jump1:>9.2f}% {jump2:>9.2f}% {spike_type:<12}")
        
        print("-" * 120)
        print("✓ 这些是典型的V型异常（可能是脏数据或瞬间错误报价），将被移除")
    else:
        print("✓ 未发现孤立尖峰异常")
    
    # 移除孤立尖峰
    df = df[~df.index.isin(abnormal_spikes.index)].copy()
    
    # 删除临时列
    df = df.drop(['price_change', 'next_price_change', 'prev_close', 'next_close'], axis=1)
    
    removed_spikes = before - len(df)
    print(f"\n移除了 {removed_spikes:,} 行孤立尖峰数据 ({removed_spikes/before*100:.2f}%)")
    print(f"剩余数据: {len(df):,} 行")
    
    # 检查是否还有大幅变化（可能是真实的市场事件）
    df_temp = df.copy()
    df_temp['pct_change'] = df_temp['close'].pct_change()
    large_changes = df_temp[df_temp['pct_change'].abs() > 0.05]
    if len(large_changes) > 0:
        print(f"\n⚠️  数据中仍有 {len(large_changes)} 条大于5%的变化（已保留，可能是真实市场事件）")
    
    # 步骤4: 保存清理后的数据
    print("\n步骤4: 保存清理后的数据...")
    
    output_df = df[['ts_event', 'open', 'high', 'low', 'close', 'volume']].copy()
    output_df.to_csv(output_path, index=False)
    
    print(f"\n清理后数据已保存到: {output_path}")
    print(f"最终数据: {len(output_df):,} 行")
    print(f"时间范围: {output_df['ts_event'].min()} 到 {output_df['ts_event'].max()}")
    print(f"价格范围: {output_df['close'].min():.2f} 到 {output_df['close'].max():.2f}")
    
    # 生成统计报告
    print("\n" + "="*60)
    print("数据质量报告")
    print("="*60)
    print(f"数据完整性: {len(output_df)/len(pd.read_csv(input_path))*100:.2f}%")
    print(f"移除错误数据: {removed:,} 行")
    print(f"移除重复数据（同一分钟保留最大成交量）: {removed_dups:,} 行")
    print(f"移除孤立尖峰数据: {removed_spikes:,} 行")

    returns_stats = output_df['close'].pct_change()
    print(f"\n收益率统计:")
    print(f"  均值: {returns_stats.mean()*100:.4f}%")
    print(f"  标准差: {returns_stats.std()*100:.4f}%")
    print(f"  最大: {returns_stats.max()*100:.2f}%")
    print(f"  最小: {returns_stats.min()*100:.2f}%")
    print(f"  95分位: {returns_stats.quantile(0.95)*100:.2f}%")
    print(f"  5分位: {returns_stats.quantile(0.05)*100:.2f}%")
    
    print("\n" + "="*60)
    print("✅ 数据清理完成！")
    print("="*60)
    print(f"\n现在可以使用 {output_path} 进行回测")
    
    return output_path


if __name__ == '__main__':
    import sys
    
    # 检查命令行参数，判断处理 NQ 还是 BTC
    if len(sys.argv) > 1 and sys.argv[1] == '--btc':
        # 处理 BTC 数据
        input_path = './btc_m1_all.csv'
        output_path = './btc_m1_cleaned.csv'
        
        # 可以从命令行指定输出路径
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
        
        print("\nBTC 期货数据清理工具")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print("\n开始处理...\n")
        
        clean_btc_data(input_path, output_path)
    else:
        # 处理 NQ 数据（默认）
        input_path = './nq_m1_all_backtrader.csv'
        output_path = './nq_m1_cleaned.csv'
        
        # 可以从命令行指定输出路径
        if len(sys.argv) > 1:
            output_path = sys.argv[1]
        
        print("\nNQ 数据清理工具")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print("\n开始处理...\n")
        
        quick_clean_data(input_path, output_path)

