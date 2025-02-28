source venv/bin/activateimport yfinance as yf

# 获取比特币数据
btc_data = yf.download('BTC-USD', start='2013-01-01', end='2025-01-01')

# 查看前几行数据
print(btc_data.head())

#测试一下