from flask import Flask, render_template, request, redirect
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 初始化 Flask 应用
app = Flask(__name__)

# 定义每次显示的行数（1年约365天）
ROWS_PER_PAGE = 365

# 设置全局浮点数显示格式，保留整数并使用千分号
pd.options.display.float_format = '{:,.0f}'.format

# 获取比特币历史价格数据（Yahoo Finance）
def fetch_btc_data():
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        btc_data = yf.download('BTC-CNY', start='2013-01-01', end=end_date, interval='1d', auto_adjust=True)
        if btc_data.empty:
            return None, "无法获取比特币数据，请检查网络或 Yahoo Finance API。"
        btc_data = btc_data[['Close']]
        btc_data.dropna(inplace=True)
        return btc_data, None
    except Exception as e:
        return None, f"获取数据失败：{str(e)}"

# 计算200日定投成本
def calculate_investment_cost_200(prices):
    rolling_mean = prices['Close'].rolling(window=200).mean()
    return rolling_mean

# 计算九神指数拟合价格
def calculate_fitted_price(prices):
    days_since_genesis = (pd.to_datetime(prices.index) - pd.Timestamp('2009-01-03')).days
    days_series = pd.Series(days_since_genesis, index=prices.index)
    days_series = days_series[days_series > 0]
    if days_series.empty:
        raise ValueError("没有有效的日期数据可计算拟合价格")
    fitted_price = 10 ** (5.84 * np.log10(days_series) - 17.01)
    return pd.Series(fitted_price * 7, index=days_series.index)

# 生成未来一年的九神拟合价格数据
def generate_future_fitted_price(days_in_future=365):
    today = pd.Timestamp.today()
    future_dates = pd.date_range(start=today + timedelta(days=1), periods=days_in_future)
    days_since_genesis_future = (future_dates - pd.Timestamp('2009-01-03')).days
    fitted_price_future = 10 ** (5.84 * np.log10(days_since_genesis_future) - 17.01) * 7
    return pd.Series(fitted_price_future, index=future_dates)

def calculate_ahr999(prices, investment_cost_200, fitted_price):
    # 对齐索引
    investment_cost_200 = investment_cost_200.reindex(prices.index).fillna(0)
    fitted_price = fitted_price.reindex(prices.index).fillna(0)
    
    # 调试信息
    #print(type(fitted_price))
    #print(type(investment_cost_200))
    #print(len(fitted_price), len(investment_cost_200))
    #print("investment_cost_200 shape:", investment_cost_200.shape)
    #print("fitted_price shape:", fitted_price.shape)
    
    # 计算 denominator，确保一维
    denominator = investment_cost_200 * fitted_price
    #print("denominator shape:", denominator.shape)
    
    # 获取一维数组，强制展平
    close_vals = prices['Close'].values.flatten()  # 将 (3814, 1) 强制展平为 (3814,)
    denominator_vals = denominator.values
    
    # 计算中间结果
    close_squared = close_vals ** 2
    condition = denominator_vals != 0
    x = close_squared / denominator_vals
    
    # 创建一维 NaN 数组
    y = np.full(close_vals.shape, np.nan)
    
    # 调试形状
    #print("close_vals shape:", close_vals.shape)
    #print("close_squared shape:", close_squared.shape)
    ##print("denominator_vals shape:", denominator_vals.shape)
    #print("condition shape:", condition.shape)
    #print("x shape:", x.shape)
    #print("y shape:", y.shape)
    
    # 计算 AHR999
    ahr999 = np.where(condition, x, y)
    #print("ahr999 shape:", ahr999.shape)
    
    # 返回 Series
    return pd.Series(ahr999, index=prices.index).round(2)

# 创建 Plotly 图表
def create_plotly_figure(prices, future_fitted_price):
    # 合并历史和未来日期
    all_dates = prices.index.union(future_fitted_price.index)
    # 扩展 prices 到未来日期
    prices = prices.reindex(all_dates)
    # 合并历史和未来的 FittedPrice
    prices['FittedPrice'] = prices['FittedPrice'].combine_first(future_fitted_price)
    # 创建图表，设置secondary_y=True以启用右侧的第二个Y轴
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 添加AHR999指数线（使用右侧的Y轴）
    fig.add_trace(go.Scatter(x=prices.index, y=prices['AHR999'], mode='lines', name='AHR999 指数', line=dict(color='violet')), secondary_y=True)

    # 添加比特币价格线（使用左侧的Y轴）
    fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='比特币价格($)', line=dict(color='blue'), hovertemplate='%{y:,.0f}'), secondary_y=False)

    # 添加200日定投成本线（使用左侧的Y轴）
    fig.add_trace(go.Scatter(x=prices.index, y=prices['InvestmentCost200'], mode='lines', name='200日定投成本线', line=dict(color='purple'), hovertemplate='%{y:,.0f}'), secondary_y=False)

    # 添加九神指数拟合价格线（使用左侧的Y轴）
    fig.add_trace(go.Scatter(x=prices.index, y=prices['FittedPrice'], mode='lines', name='九神指数预测价格', line=dict(color='cyan'), hovertemplate='%{y:,.0f}'), secondary_y=False)

    # 添加定投和抄底价格线（使用左侧的Y轴）
    fig.add_trace(go.Scatter(x=prices.index, y=prices['1.2 Invest Price'], mode='lines', name='1.2定投价格', line=dict(color='green'), hovertemplate='%{y:,.0f}'), secondary_y=False)
    fig.add_trace(go.Scatter(x=prices.index, y=prices['0.45 Bottom Price'], mode='lines', name='0.45抄底价格', line=dict(color='red'), hovertemplate='%{y:,.0f}'), secondary_y=False)
    fig.add_trace(go.Scatter(x=prices.index, y=prices['6.66 Runaway Price'], mode='lines', name='6.66跑路价格', line=dict(color='orange'), hovertemplate='%{y:,.0f}'), secondary_y=False)

    # 设置显示范围
    today = pd.Timestamp.today()
    start_date = today - timedelta(days=365)
    end_date = today + timedelta(days=365)

    fig.update_layout(
        autosize=True,
        height=600,
        width=1300,
        xaxis=dict(
            title='日期',
            tickformat='%Y-%m-%d',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="2y", step="year", stepmode="backward"),
                    dict(count=4, label="4y", step="year", stepmode="backward"),
                    dict(count=6, label="6y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True,
                thickness=0.15,
                bgcolor='white',
            ),
            range=[start_date, end_date],
        ),
        yaxis=dict(
            side='left',
            tickformat=',',
            autorange=True
            ),
        yaxis2=dict(
            side='right',
            autorange=True,
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing='constant'
        ),
        margin=dict(l=20, r=20, t=50, b=30)
    )

    return fig

# 主路由
@app.route('/')
def index():
    # 获取当前页面的偏移量和显示数量
    offset = int(request.args.get('offset', 0))
    limit = ROWS_PER_PAGE  # 每次加载365行数据

    # 获取数据
    prices, error = fetch_btc_data()
    if prices is None:  # 如果价格数据为 None，则返回错误页面
        return render_template('error.html', message=error)

    # 后续的数据处理和图表生成逻辑
    # 计算200日定投成本
    prices['InvestmentCost200'] = calculate_investment_cost_200(prices)

    # 计算九神指数拟合价格
    prices['FittedPrice'] = calculate_fitted_price(prices)

    # 计算 AHR999 指数
    prices['AHR999'] = calculate_ahr999(prices, prices['InvestmentCost200'], prices['FittedPrice'])

    # 计算1.2定投价格和0.45抄底价格以及6.66跑路价格
    prices['1.2 Invest Price'] = np.sqrt(1.2 * prices['InvestmentCost200'] * prices['FittedPrice'])
    prices['0.45 Bottom Price'] = np.sqrt(0.45 * prices['InvestmentCost200'] * prices['FittedPrice'])
    prices['6.66 Runaway Price'] = np.sqrt(6.66 * prices['InvestmentCost200'] * prices['FittedPrice'])

    # 计算操作建议
    conditions = [
        (prices['AHR999'] < 0.45),
        (prices['AHR999'] >= 0.45) & (prices['AHR999'] <= 1.2),
        (prices['AHR999'] > 1.2) & (prices['AHR999'] <= 5),
        (prices['AHR999'] > 5) & (prices['AHR999'] <= 6.66),
        (prices['AHR999'] > 6.66)
    ]
    choices = ['抄底', '定投', '等待起飞', '准备抛货', '快跑']
    prices['Operation'] = np.select(conditions, choices, default='无')

    # 生成未来一年的九神拟合价格数据
    future_fitted_price = generate_future_fitted_price(days_in_future=365)

    # 创建图表   先隐藏了，现在不完美
    #fig = create_plotly_figure(prices, future_fitted_price)
    #plot_html = fig.to_html(full_html=False, config={'displayModeBar': False, 'responsive': True})

    # 筛选并按日期倒序排列每日数据
    prices_sorted = prices[['AHR999', 'Close', 'InvestmentCost200', 'FittedPrice', '1.2 Invest Price', '0.45 Bottom Price', '6.66 Runaway Price', 'Operation']].sort_index(ascending=False)
    prices_sorted.reset_index(inplace=True)
    prices_sorted.columns = ['时间', 'Ahr999指数', '比特币价格', '200日定投成本', '九神指数预测价格', '1.2定投价格', '0.45抄底价格', '6.66跑路价格', '适合做啥']

    # 将 AHR999 指数保留两位小数
    prices_sorted['Ahr999指数'] = prices_sorted['Ahr999指数'].map('{:.2f}'.format)
    
    # 实现分页：取出部分数据
    total_rows = len(prices_sorted)
    prices_sorted_paginated = prices_sorted[offset:offset + limit]

    # 计算当前页的结束行数
    end_row = min(offset + limit, total_rows)
    # 计算前一页的 offset
    prev_offset = max(offset - limit, 0)
    next_offset = offset + limit if offset + limit < total_rows else None

    # 将数据转换为HTML表格
    table_html = prices_sorted_paginated.to_html(index=False, classes='data-table')

    # 渲染模板并传递分页信息
    return render_template(
        'index.html', 
        #plot_html=plot_html,  #先隐藏图标
        tables=table_html, 
        offset=offset, 
        limit=limit, 
        total_rows=total_rows, 
        end_row=end_row, 
        prev_offset=prev_offset, 
        next_offset=next_offset
    )

# 运行应用
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)