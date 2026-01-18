"""
时区修复工具 - 去除CSV中的时区信息以兼容 backtrader
用于修复已有的带时区信息的CSV文件

用法:
    python3 fix_timezone.py input.csv output.csv
    python3 fix_timezone.py btc_m1_cleaned.csv btc_m1_cleaned_fixed.csv
"""

import pandas as pd
import sys
import os


def fix_timezone(input_path, output_path=None):
    """
    去除CSV文件中的时区信息
    
    Args:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径，如果为None则覆盖原文件
    """
    print("=" * 60)
    print("时区修复工具")
    print("=" * 60)
    
    # 如果未指定输出路径，则覆盖原文件
    if output_path is None:
        output_path = input_path
        print(f"\n⚠️  警告: 将覆盖原文件 {input_path}")
    else:
        print(f"\n输入文件: {input_path}")
        print(f"输出文件: {output_path}")
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"\n❌ 错误: 文件不存在 {input_path}")
        return False
    
    try:
        # 读取CSV
        print("\n步骤1: 读取CSV文件...")
        df = pd.read_csv(input_path)
        print(f"✓ 读取成功，共 {len(df):,} 行")
        
        # 检查是否有时间列
        time_columns = ['ts_event', 'datetime', 'date', 'time', 'timestamp']
        found_time_col = None
        
        for col in time_columns:
            if col in df.columns:
                found_time_col = col
                break
        
        if found_time_col is None:
            # 假设第一列是时间列
            found_time_col = df.columns[0]
            print(f"⚠️  未找到标准时间列名，使用第一列: {found_time_col}")
        
        print(f"\n步骤2: 处理时间列 '{found_time_col}'...")
        
        # 显示原始格式示例
        print(f"原始格式示例（前3行）:")
        for i in range(min(3, len(df))):
            print(f"  {df[found_time_col].iloc[i]}")
        
        # 转换为datetime并去除时区
        df[found_time_col] = pd.to_datetime(df[found_time_col])
        
        # 检查是否有时区信息
        if df[found_time_col].dt.tz is not None:
            print(f"✓ 检测到时区信息: {df[found_time_col].dt.tz}")
            df[found_time_col] = df[found_time_col].dt.tz_localize(None)
            print("✓ 已移除时区信息")
        else:
            print("✓ 未检测到时区信息（可能已经是正确格式）")
        
        # 显示修复后的格式
        print(f"\n修复后格式示例（前3行）:")
        for i in range(min(3, len(df))):
            print(f"  {df[found_time_col].iloc[i]}")
        
        # 保存
        print(f"\n步骤3: 保存到 {output_path}...")
        df.to_csv(output_path, index=False)
        print("✓ 保存成功")
        
        # 验证
        print("\n步骤4: 验证修复结果...")
        verify_df = pd.read_csv(output_path, nrows=5)
        print(f"验证读取（前5行的时间列）:")
        for i in range(min(5, len(verify_df))):
            print(f"  {verify_df[found_time_col].iloc[i]}")
        
        # 检查格式
        sample_time = str(verify_df[found_time_col].iloc[0])
        if '+' in sample_time or 'UTC' in sample_time.upper():
            print("\n⚠️  警告: 文件中仍包含时区信息")
            return False
        else:
            print("\n✅ 验证通过，时区信息已完全移除")
            return True
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_fix(directory, pattern="*_cleaned.csv"):
    """
    批量修复目录下的所有CSV文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
    """
    import glob
    
    print("=" * 60)
    print("批量时区修复工具")
    print("=" * 60)
    print(f"\n目录: {directory}")
    print(f"匹配模式: {pattern}")
    
    # 查找所有匹配的文件
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"\n未找到匹配的文件")
        return
    
    print(f"\n找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    print("\n开始批量处理...")
    success_count = 0
    fail_count = 0
    
    for file_path in files:
        print(f"\n{'='*60}")
        print(f"处理: {os.path.basename(file_path)}")
        print('='*60)
        
        # 创建备份
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"✓ 已创建备份: {os.path.basename(backup_path)}")
        
        # 修复文件（覆盖原文件）
        if fix_timezone(file_path, file_path):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("批量处理完成")
    print("=" * 60)
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  单个文件修复:")
        print("    python3 fix_timezone.py <input.csv> [output.csv]")
        print("  ")
        print("  批量修复（覆盖原文件并创建.backup备份）:")
        print("    python3 fix_timezone.py --batch [directory] [pattern]")
        print("")
        print("示例:")
        print("  python3 fix_timezone.py btc_m1_cleaned.csv btc_m1_cleaned_fixed.csv")
        print("  python3 fix_timezone.py btc_m1_cleaned.csv  # 覆盖原文件")
        print("  python3 fix_timezone.py --batch ./  # 批量修复当前目录所有 *_cleaned.csv")
        print("  python3 fix_timezone.py --batch ./ '*.csv'  # 批量修复所有csv")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        # 批量模式
        directory = sys.argv[2] if len(sys.argv) > 2 else './'
        pattern = sys.argv[3] if len(sys.argv) > 3 else '*_cleaned.csv'
        batch_fix(directory, pattern)
    else:
        # 单文件模式
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        success = fix_timezone(input_file, output_file)
        sys.exit(0 if success else 1)
