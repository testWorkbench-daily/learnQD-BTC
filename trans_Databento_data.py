import databento as db
import pandas as pd

# 2. 读取下载的文件
data = db.DBNStore.from_file('data/BTC-GLBX-20260113-AAKWJ5U7NC/glbx-mdp3-20160112-20260111.ohlcv-1m.dbn.zst')

# 3. 查看数据结构
df = data.to_df()
print(df.head())
print(df.columns)

# 4. 存成 CSV
df.to_csv('./data/btc_m1_all.csv')

# 5. 处理成 Backtrader 格式
df.index = pd.to_datetime(df.index).tz_localize(None)
df = df[['open', 'high', 'low', 'close', 'volume']]
df.to_csv('./data/btc_m1_all_backtrader.csv')