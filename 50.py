from datetime import timedelta

# 计算九神指数拟合价格，返回特定日期的拟合价格
def calculate_fitted_price_on_date(date):
    days_since_genesis = (date - pd.Timestamp('2009-01-03')).days
    fitted_price = 10 ** (5.84 * np.log10(days_since_genesis) - 17.01)
    return fitted_price * 7  # 转换为人民币价格（根据需要可以移除）

# 计算2009年到未来50年的每年1月3日的拟合价格
def calculate_fitted_prices_50_years():
    start_date = pd.Timestamp('2009-01-03')
    prices = {}
    
    for year in range(50):
        current_date = start_date + pd.DateOffset(years=year)
        prices[current_date.year] = calculate_fitted_price_on_date(current_date)
    
    return prices

# 现在调用这个函数可以得到每年1月3日的拟合价格：
fitted_prices_50_years = calculate_fitted_prices_50_years()

# 打印结果
for year, price in fitted_prices_50_years.items():
    print(f"Year: {year}, Fitted Price: {price:.2f} CNY")