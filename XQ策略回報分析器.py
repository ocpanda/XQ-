import calendar
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns
import traceback
import math


# 設定 matplotlib 使用微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== Configuration Module ===================== #

class StrategyConfig:
    def __init__(self):
        # 預設值
        self.strategy_type = 'short'  # 多方或空方
        self.is_margin = True # 是否當沖
        self.use_lot = True  # True表示用張數計算,False表示用零股計算
        self.fixed_order_amount = None  # 固定下單金額,None表示使用實際計算值
        self.fee_discount = 0.28  # 手續費折數
    
    def set_is_margin(self, is_margin):
        """設定是否當沖
        Args:
            is_margin (bool): True表示當沖,False表示非當沖
        """
        self.is_margin = is_margin

    def set_strategy_type(self, strategy_type):
        """設定策略類型
        Args:
            strategy_type (str): 'long'表示多方,'short'表示空方
        """
        self.strategy_type = strategy_type

    def set_use_lot(self, use_lot):
        """設定是否使用張數計算
        Args:
            use_lot (bool): True表示用張數計算,False表示用零股計算
        """
        self.use_lot = use_lot

    def set_fee_discount(self, fee_discount):
        """設定手續費折數
        Args:
            fee_discount (float): 手續費折數
        """
        self.fee_discount = fee_discount

    def set_fixed_order_amount(self, amount):
        """設定固定下單金額
        Args:
            amount (float): 固定下單金額,None表示使用實際計算值
        """
        self.fixed_order_amount = amount

    def calculate_order_amount(self, entry_price, quantity):
        """計算下單金額
        Args:
            entry_price (float): 進場價格
            quantity (int): 交易數量
        Returns:
            float: 計算後的下單金額
        """
        if self.fixed_order_amount is not None:
            if self.use_lot:
                return entry_price * 1000 * math.ceil(self.fixed_order_amount / (entry_price * 1000))
            else:
                return entry_price * math.ceil(self.fixed_order_amount / entry_price)
        else:
            if self.use_lot:
                return entry_price * 1000 * quantity
            else:
                return entry_price * quantity
    
    def calculate_fee(self, order_amount, is_buy=True):
        """計算手續費
        Args:
            order_amount (float): 下單金額
            is_buy (bool): 是否為買進,True表示買進,False表示賣出
        Returns:
            float: 計算後的手續費
        """
        if is_buy:
            # 買進手續費
            return order_amount * 0.001425 * self.fee_discount
        else:
            # 賣出手續費 + 證交稅
            if self.is_margin:
                return (order_amount * 0.001425 * self.fee_discount) + (order_amount * 0.0015)
            else:
                return (order_amount * 0.001425 * self.fee_discount) + (order_amount * 0.003)

    def calculate_profit(self, entry_price, exit_price, quantity):
        """計算獲利金額
        Args:
            entry_price (float): 進場價格
            exit_price (float): 出場價格
            quantity (int): 交易數量
        Returns:
            float: 計算後的獲利金額
        """
        # 計算下單金額
        entry_amount = self.calculate_order_amount(entry_price, quantity)
        exit_amount = self.calculate_order_amount(exit_price, quantity)
        
        # 計算手續費
        entry_fee = self.calculate_fee(entry_amount, is_buy=True)
        exit_fee = self.calculate_fee(exit_amount, is_buy=False)
        
        # 計算毛利
        if self.use_lot:
            if self.strategy_type == 'long':
                gross_profit = (exit_price - entry_price) * quantity * 1000
            else:
                gross_profit = (entry_price - exit_price) * quantity * 1000
        else:
            if self.strategy_type == 'long':
                gross_profit = (exit_price - entry_price) * math.ceil(self.fixed_order_amount / entry_price)
            else:
                gross_profit = (entry_price - exit_price) * math.ceil(self.fixed_order_amount / entry_price)
                
        # 扣除手續費後的淨利
        net_profit = gross_profit - entry_fee - exit_fee
        
        return net_profit

class ConfigManager:
    def __init__(self):
        self.strategy_configs = {}  # 儲存每個策略的設定
        
    def add_strategy(self, strategy_name):
        """新增策略設定
        Args:
            strategy_name (str): 策略名稱
        """
        if strategy_name not in self.strategy_configs:
            self.strategy_configs[strategy_name] = StrategyConfig()

    def get_strategy_config(self, strategy_name):
        """取得策略設定
        Args:
            strategy_name (str): 策略名稱
        Returns:
            StrategyConfig: 策略設定物件
        """
        if strategy_name not in self.strategy_configs:
            self.add_strategy(strategy_name)
        return self.strategy_configs[strategy_name]

    def set_strategy_use_lot(self, strategy_name, use_lot):
        """設定策略是否使用張數計算
        Args:
            strategy_name (str): 策略名稱
            use_lot (bool): True表示用張數計算,False表示用零股計算
        """
        config = self.get_strategy_config(strategy_name)
        config.set_use_lot(use_lot)

    def set_strategy_fixed_amount(self, strategy_name, amount):
        """設定策略固定下單金額
        Args:
            strategy_name (str): 策略名稱 
            amount (float): 固定下單金額,None表示使用實際計算值
        """
        config = self.get_strategy_config(strategy_name)
        config.set_fixed_order_amount(amount)

    def set_fee_discount(self, strategy_name, fee_discount):
        """設定手續費折數
        Args:
            strategy_name (str): 策略名稱
            fee_discount (float): 手續費折數
        """
        config = self.get_strategy_config(strategy_name)
        config.set_fee_discount(fee_discount)
    
    def set_is_margin(self, strategy_name, is_margin):
        """設定是否當沖
        Args:
            strategy_name (str): 策略名稱
            is_margin (bool): True表示當沖,False表示非當沖
        """
        config = self.get_strategy_config(strategy_name)
        config.set_is_margin(is_margin)

    def set_strategy_type(self, strategy_name, strategy_type):
        """設定策略類型
        Args:
            strategy_name (str): 策略名稱
            strategy_type (str): 'long'表示多方,'short'表示空方
        """
        config = self.get_strategy_config(strategy_name)
        config.set_strategy_type(strategy_type)
    
    def clear_all_configs(self):
        """清除所有策略設定"""
        self.strategy_configs.clear()


# ===================== Data Processing Module ===================== #

class DataProcessor:
    @staticmethod
    def process_file(file_path, strategy_name, config_manager):
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_path, encoding='ansi')
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_path, sheet_name='交易分析')
            df['獲利金額'] = df['獲利金額'].str.replace(',', '').astype(float)
        else:
            raise ValueError("不支持的文件格式。請使用 CSV 或 XLSX 文件。")

        df['進場時間'] = pd.to_datetime(df['進場時間'], format='%Y/%m/%d %H:%M')
        df['出場時間'] = pd.to_datetime(df['出場時間'], format='%Y/%m/%d %H:%M')
        df.sort_values(by='進場時間', inplace=True)
        df['日期'] = df['進場時間'].dt.normalize()

        # 取得該策略的設定
        config = config_manager.get_strategy_config(strategy_name)

        # 根據設定計算下單金額和獲利金額
        df['下單金額'] = df.apply(lambda row: config.calculate_order_amount(row['進場價格'], row['交易數量']), axis=1)
        df['獲利金額'] = df.apply(lambda row: config.calculate_profit(row['進場價格'], row['出場價格'], row['交易數量']), axis=1)

        daily_returns = df.groupby('日期')['獲利金額'].sum()
        daily_returns.index = pd.to_datetime(daily_returns.index)

        return df, daily_returns

    @staticmethod
    def calculate_stats(df):
        grouped_trades = df.groupby(['日期', '商品代碼']).size()
        total_trades = len(grouped_trades)

        # 計算獲利交易次數
        win_trades = len(df.groupby(['日期', '商品代碼'])['獲利金額'].sum()[lambda x: x > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        average_profit = df['獲利金額'].mean() if total_trades > 0 else 0

        daily_order_amount = df.groupby('日期')['下單金額'].sum()
        maximum_order_amount = daily_order_amount.max() if total_trades > 0 else 0
        max_order_date = daily_order_amount.idxmax() if total_trades > 0 else None

        return {
            'total_trades': total_trades,
            'win_trades': win_trades,
            'win_rate': win_rate,
            'average_profit': average_profit,
            'maximum_order_amount': maximum_order_amount
        }

    @staticmethod
    def calculate_drawdown(returns):
        cumulative_returns = returns.cumsum()
        peak = cumulative_returns.cummax()
        drawdown = peak - cumulative_returns
        max_drawdown = drawdown.cummax()
        return cumulative_returns, peak, drawdown, max_drawdown

    @staticmethod
    def filter_overlapping_trades(selected_df, other_df):
        """
        過濾掉與選定策略有重疊商品的交易
        """
        filtered_df = other_df.copy()
        selected_trades = selected_df.groupby(['日期', '商品代碼']).size()

        # 找出需要排除的交易
        for (date, symbol) in selected_trades.index:
            mask = (filtered_df['日期'] == date) & (filtered_df['商品代碼'] == symbol)
            filtered_df = filtered_df[~mask]

        return filtered_df

# ===================== Chart Plotting Module ===================== #

class ChartPlotter:
    def __init__(self):
        self.vertical_line1 = None
        self.horizontal_line1 = None
        self.vertical_line2 = None
        self.horizontal_line2 = None
        self.stats_text = None
        self.current_month = datetime.now()
        self.current_year = datetime.now().year

        self.strategy_colors = {}  # 用於存儲策略的顏色映射
        self.color_list = ['blue', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.combined_color = 'green'

    def plot_correlation(self, all_returns):
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = all_returns.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title('策略相關性分析')
        return fig

    def plot_monthly_returns(self, trading_data):
        self._assign_colors(trading_data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.2, 0.8])
        plt.subplots_adjust(hspace=0.3)

        year_start = pd.Timestamp(self.current_year, 1, 1)
        year_end = pd.Timestamp(self.current_year, 12, 31)

        monthly_data = self._get_monthly_data(trading_data)
        months = pd.date_range(start=year_start, end=year_end, freq='M')
        bar_width = 0.8 / (len(trading_data) + 1)
        bars = []

        for i, (name, returns) in enumerate(monthly_data.items()):
            monthly_values = returns.reindex(months, fill_value=0)
            color = self.strategy_colors[name]
            bar_container = ax1.bar(np.arange(12) + i * bar_width,
                                    monthly_values.values,
                                    bar_width,
                                    label=name,
                                    alpha=0.7,
                                    picker=True,
                                    color=color)
            for rect in bar_container:
                rect.custom_label = name
            bars.extend(bar_container)

            yearly_returns = returns.resample('Y').sum()
            ax2.plot(yearly_returns.index.year, yearly_returns.values,
                     marker='o', label=name, linewidth=2, color=color)

        combined_monthly = pd.DataFrame(monthly_data).fillna(0).sum(axis=1)
        bar_container = ax1.bar(np.arange(12) + len(trading_data) * bar_width,
                                combined_monthly.reindex(months, fill_value=0).values,
                                bar_width,
                                label='策略合併執行',
                                alpha=0.7,
                                picker=True,
                                color=self.combined_color)
        for rect in bar_container:
            rect.custom_label = '策略合併執行'
        bars.extend(bar_container)

        combined_yearly = combined_monthly.resample('Y').sum()
        ax2.plot(combined_yearly.index.year, combined_yearly.values,
                 marker='o', label='策略合併執行', linewidth=2, linestyle='--', color=self.combined_color)

        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        self._set_monthly_chart_properties(ax1, ax2, combined_monthly)
        self._add_monthly_hover_effect(fig, ax1, bars)

        return fig

    def plot_daily_returns(self, trading_data):
        self._assign_colors(trading_data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.2, 0.8])
        plt.subplots_adjust(hspace=0.3)

        # 获取当前月份的起始和结束日期
        month_start, month_end = self._get_current_month_range()
        all_dates = pd.date_range(start=month_start, end=month_end)
        
        # 创建一个DataFrame来存储所有策略的每日收益
        all_returns_df = pd.DataFrame(index=all_dates)
        for name, returns in trading_data.items():
            returns.index = pd.to_datetime(returns.index)
            all_returns_df[name] = returns.reindex(all_dates, fill_value=0)
            
        # 计算每日的合并收益
        combined_returns = all_returns_df.sum(axis=1)
        bar_width = 0.8 / (len(trading_data) + 1)
        bars = []

        # 用于存储月度收益数据
        monthly_data = {}

        for i, (name, returns) in enumerate(trading_data.items()):
            returns.index = pd.to_datetime(returns.index)
            daily_data = returns.reindex(all_dates, fill_value=0)
            color = self.strategy_colors[name]
            bar_container = ax1.bar(np.arange(len(all_dates)) + i * bar_width,
                                    daily_data.values,
                                    bar_width,
                                    label=name,
                                    alpha=0.7,
                                    picker=True,
                                    color=color)
            for rect in bar_container:
                rect.custom_label = name
            bars.extend(bar_container)

            # 计算月度收益并存储
            monthly_returns = returns.resample('M').sum()
            monthly_data[name] = monthly_returns

            # 在第二个子图上绘制月度收益趋势
            ax2.plot(monthly_returns.index, monthly_returns.values,
                    label=name, marker='o', color=color)

        # 绘制合并策略的柱状图
        bar_container = ax1.bar(np.arange(len(all_dates)) + len(trading_data) * bar_width,
                                combined_returns.values,
                                bar_width,
                                label='策略合併執行',
                                alpha=0.7,
                                picker=True,
                                color=self.combined_color)
        for rect in bar_container:
            rect.custom_label = '策略合併執行'
        bars.extend(bar_container)

        # 获取所有策略的最早和最晚日期
        all_start_dates = []
        all_end_dates = []
        for returns in trading_data.values():
            returns.index = pd.to_datetime(returns.index)
            all_start_dates.append(returns.index.min())
            all_end_dates.append(returns.index.max())
        
        overall_start = min(all_start_dates)
        overall_end = max(all_end_dates)
        
        # 创建完整时间范围的DataFrame
        full_date_range = pd.date_range(start=overall_start, end=overall_end, freq='D')
        full_returns_df = pd.DataFrame(index=full_date_range)
        
        # 填充所有策略的数据
        for name, returns in trading_data.items():
            full_returns_df[name] = returns.reindex(full_date_range, fill_value=0)
        
        # 计算合并策略的月度收益趋势
        combined_monthly = full_returns_df.sum(axis=1).resample('M').sum()
        ax2.plot(combined_monthly.index, combined_monthly.values,
                label='策略合併執行', marker='o', linewidth=2, linestyle='--', color=self.combined_color)

        # 设置图表属性
        self._set_daily_chart_properties(ax1, ax2, combined_returns, all_dates)
        self._add_daily_hover_effect(fig, ax1, bars, all_dates)

        return fig


    def plot_returns_and_drawdown(self, trading_data, strategy_stats):
        self._assign_colors(trading_data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
        fig.subplots_adjust(right=0.75)

        all_dates = self._get_all_dates(trading_data)
        combined_returns_df = pd.DataFrame(trading_data).fillna(0)
        combined_returns = combined_returns_df.sum(axis=1)
        combined_returns = combined_returns.reindex(all_dates, fill_value=0)
        combined_cum_returns, _, _, combined_max_dd = DataProcessor.calculate_drawdown(combined_returns)

        ax1.plot(combined_cum_returns.index, combined_cum_returns.values, label='策略合併執行',
                 linewidth=2, linestyle='--', color=self.combined_color)
        ax2.plot(combined_max_dd.index, combined_max_dd.values, label='策略合併執行',
                 linewidth=2, linestyle='--', color=self.combined_color)

        for name, returns in trading_data.items():
            returns = returns.reindex(all_dates, fill_value=0)
            cum_returns, _, _, max_dd = DataProcessor.calculate_drawdown(returns)
            color = self.strategy_colors[name]

            ax1.plot(cum_returns.index, cum_returns.values, label=name, color=color)
            ax2.plot(max_dd.index, max_dd.values, label=name, color=color)

        self._set_returns_chart_properties(ax1, ax2)
        stats_text = self.generate_stats_text(trading_data, strategy_stats, combined_max_dd)
        self.stats_text = fig.text(0.77, 0.5, stats_text, va='center', fontsize=10)
        self._add_returns_hover_effect(fig, ax1, ax2)

        return fig

    # ========== Helper Methods ========== #

    def _assign_colors(self, trading_data):
        strategy_names = list(trading_data.keys())
        for i, name in enumerate(strategy_names):
            if name not in self.strategy_colors:
                self.strategy_colors[name] = self.color_list[i % len(self.color_list)]

    def _get_monthly_data(self, trading_data):
        monthly_data = {}
        for name, returns in trading_data.items():
            returns.index = pd.to_datetime(returns.index)
            monthly_returns = returns.resample('M').sum()
            monthly_data[name] = monthly_returns
        return monthly_data

    def _get_current_month_range(self):
        month_start = pd.Timestamp(self.current_month.year, self.current_month.month, 1)
        if self.current_month.month == 12:
            month_end = pd.Timestamp(self.current_month.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = pd.Timestamp(self.current_month.year, self.current_month.month + 1, 1) - timedelta(days=1)
        return month_start, month_end

    def _set_monthly_chart_properties(self, ax1, ax2, combined_monthly):
        year_mask = combined_monthly.index.year == self.current_year
        total_profit = combined_monthly[year_mask].sum()
        ax1.set_title(f'{self.current_year}年 月獲利分析 (年度總獲利: {total_profit:,.0f})')
        ax1.set_xlabel('月份')
        ax1.set_ylabel('獲利金額')
        ax1.legend(loc='upper left')

        ax2.set_title('年度獲利趨勢')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('年度獲利金額')
        ax2.legend(loc='upper left')

        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                        '7月', '8月', '9月', '10月', '11月', '12月']
        ax1.set_xticks(np.arange(12) + (len(self.strategy_colors) * (0.8 / (len(self.strategy_colors) + 1))) / 2)
        ax1.set_xticklabels(month_labels)

        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    def _add_monthly_hover_effect(self, fig, ax1, bars):
        def hover(event):
            if event.inaxes != ax1:
                return

            x_pos = event.xdata
            month_index = int(x_pos + 0.5)

            for text in ax1.texts:
                text.remove()

            total_strategies = len(self.strategy_colors) + 1

            for bar_index, bar in enumerate(bars):
                bar_x = bar.get_x()
                bar_month = int(bar_x + 0.5)

                if bar_month == month_index:
                    if not hasattr(bar, 'original_height'):
                        bar.original_height = bar.get_height()
                    bar.set_height(bar.original_height * 1.1)
                    label = bar.custom_label
                    value = bar.original_height

                    vertical_offset = 0.98 - (0.05 * total_strategies)
                    current_index = len(ax1.texts)
                    text_position = vertical_offset + (0.05 * current_index)

                    ax1.text(0.98, text_position,
                             f'{label}: {value:,.0f}',
                             transform=ax1.transAxes,
                             ha='right',
                             va='top')
                else:
                    if hasattr(bar, 'original_height'):
                        bar.set_height(bar.original_height)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

    def _set_daily_chart_properties(self, ax1, ax2, combined_returns, all_dates):
        total_profit = combined_returns.sum()
        ax1.set_title(f'{self.current_month.year}年{self.current_month.month}月 日獲利分析 (總獲利: {total_profit:,.0f})')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('獲利金額')
        ax1.legend(loc='upper left')

        # 设置 x 轴刻度
        ax1.set_xticks(np.arange(len(all_dates)) + (len(self.strategy_colors) * (0.8 / (len(self.strategy_colors) + 1))) / 2)
        ax1.set_xticklabels([d.strftime('%m/%d') for d in all_dates], rotation=45)

        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

        # 设置第二个子图的属性
        ax2.set_title('月度獲利趨勢')
        ax2.set_xlabel('月份')
        ax2.set_ylabel('月獲利金額')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()


    def _add_daily_hover_effect(self, fig, ax1, bars, all_dates):
        def hover(event):
            if event.inaxes != ax1:
                return

            x_pos = event.xdata
            day_index = int(x_pos + 0.5)

            for text in ax1.texts:
                text.remove()

            total_strategies = len(self.strategy_colors) + 1

            for bar_index, bar in enumerate(bars):
                bar_x = bar.get_x()
                bar_day = int(bar_x + 0.5)

                if bar_day == day_index:
                    if not hasattr(bar, 'original_height'):
                        bar.original_height = bar.get_height()
                    bar.set_height(bar.original_height * 1.1)
                    label = bar.custom_label
                    value = bar.original_height

                    vertical_offset = 0.98 - (0.05 * total_strategies)
                    current_index = len(ax1.texts)
                    text_position = vertical_offset + (0.05 * current_index)

                    ax1.text(0.98, text_position,
                             f'{label}: {value:,.0f}',
                             transform=ax1.transAxes,
                             ha='right',
                             va='top')
                else:
                    if hasattr(bar, 'original_height'):
                        bar.set_height(bar.original_height)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

    def _get_all_dates(self, trading_data):
        all_dates = pd.DatetimeIndex([])
        for returns in trading_data.values():
            returns.index = pd.to_datetime(returns.index)
            all_dates = all_dates.union(returns.index)
        all_dates = all_dates.sort_values()
        return all_dates

    def _set_returns_chart_properties(self, ax1, ax2):
        ax1.set_title('累積報酬')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('累積報酬')
        ax1.legend()

        ax2.set_title('最大回撤 (MDD)')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('回撤')
        ax2.legend()

        ax1.get_yaxis().get_major_formatter().set_scientific(False)
        ax2.get_yaxis().get_major_formatter().set_scientific(False)

    def _add_returns_hover_effect(self, fig, ax1, ax2):
        self.vertical_line1 = None
        self.horizontal_line1 = None
        self.vertical_line2 = None
        self.horizontal_line2 = None

        def hover(event):
            if event.inaxes not in [ax1, ax2]:
                return

            for line in [self.vertical_line1, self.horizontal_line1,
                         self.vertical_line2, self.horizontal_line2]:
                if line is not None:
                    try:
                        line.remove()
                    except ValueError:
                        pass

            if event.inaxes == ax1:
                self.vertical_line1 = ax1.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.5)
                self.horizontal_line1 = ax1.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.5)
            elif event.inaxes == ax2:
                self.vertical_line2 = ax2.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.5)
                self.horizontal_line2 = ax2.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.5)

            date_str = matplotlib.dates.num2date(event.xdata).strftime('%Y-%m-%d')
            title_text = f"滑鼠位置日期：{date_str}\n"
            x_data = ax1.lines[0].get_xdata()
            x_data_num = matplotlib.dates.date2num(x_data)
            x_idx = np.argmin(abs(x_data_num - event.xdata))

            for line in ax1.lines:
                label = line.get_label()
                y_data = line.get_ydata()
                if x_idx < len(y_data):
                    y_value = y_data[x_idx]

                    for mdd_line in ax2.lines:
                        if mdd_line.get_label() == label:
                            mdd_data = mdd_line.get_ydata()
                            if x_idx < len(mdd_data):
                                mdd_value = mdd_data[x_idx]
                                title_text += f"{label} - 累積獲利: {y_value:,.0f}, MDD: {-mdd_value:,.0f}\n"
                            break

            ax1.set_title(title_text)
            ax2.set_title('最大回撤 (MDD)')

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)

    def generate_stats_text(self, trading_data, strategy_stats, combined_max_dd):
        stats_text_list = []

        combined_df = pd.concat([stats['df'] for stats in strategy_stats.values()], ignore_index=True)
        combined_stats = DataProcessor.calculate_stats(combined_df)

        stats_text_list.extend([
            "策略合併執行:",
            f"總交易次數: {combined_stats['total_trades']}",
            f"勝率: {combined_stats['win_rate']:.2%}",
            f"平均獲利: {combined_stats['average_profit']:.2f}",
            f"最大下單金額: {combined_stats['maximum_order_amount']:.0f}",
            f"MDD: {-combined_max_dd.iloc[-1]:.0f}",
            ""
        ])

        for name, stats in strategy_stats.items():
            stats_text_list.extend([
                f"{name}:",
                f"總交易次數: {stats['stats']['total_trades']}",
                f"勝率: {stats['stats']['win_rate']:.2%}",
                f"平均獲利: {stats['stats']['average_profit']:.2f}",
                f"最大下單金額: {stats['stats']['maximum_order_amount']:.0f}",
                f"MDD: {-stats['max_dd'].iloc[-1]:.0f}",
                ""
            ])

        return '\n'.join(stats_text_list)

    # ========== Navigation Methods ========== #

    def next_month(self):
        if self.current_month.month == 12:
            self.current_month = self.current_month.replace(year=self.current_month.year + 1, month=1)
        else:
            self.current_month = self.current_month.replace(month=self.current_month.month + 1)

    def prev_month(self):
        if self.current_month.month == 1:
            self.current_month = self.current_month.replace(year=self.current_month.year - 1, month=12)
        else:
            self.current_month = self.current_month.replace(month=self.current_month.month - 1)

    def set_month(self, year, month):
        self.current_month = datetime(year, month, 1)

    def next_year(self):
        self.current_year += 1

    def prev_year(self):
        self.current_year -= 1

    def set_year(self, year):
        self.current_year = year

# ===================== Application Module ===================== #

class StrategyAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XQ 策略回測報表分析器")

        # 數據存儲
        self.trading_data = {}  # 存儲每日獲利
        self.strategy_stats = {}  # 存儲策略統計信息
        self.original_data = {}  # 存儲原始數據

        # 配置管理
        self.config_manager = ConfigManager()

        # 圖表相關
        self.canvas = None
        self.chart_plotter = ChartPlotter()

        # Changelog 相關
        self.changelog_text = None
        self.changelog_content = """
            Powered by ocpanda at https://github.com/ocpanda
            如果對此軟體有任何建議或錯誤回報，歡迎到 GitHub 上提出 issue。
            喜歡的話，可以給顆星星，謝謝！

            版本更新記錄：

            v0.4.0 (2024-11-03)
            Features:
            - 新增選擇重複商品欲優先執行策略功能
            - 統一了策略在圖表上的顏色
            - 新增策略參數設定功能
                - 支援手續費設定
                - 支援策略類型設定
                - 支援是否當沖設定
                - 支援使用張數計算或零股計算
                - 支援固定下單金額設定
            Bug Fixes:
            - 修正勝率計算，改為根據日期和商品代碼分組計算，也就是同一天同一個商品算一次

            v0.3.0 (2024-10-31)
            Features:
            - 新增日獲利圖表
            - 新增月獲利圖表

            v0.2.0 (2024-10-31)
            Features:
            - 新增版本更新記錄頁面
            - 支援 XQ 完整交易回報 Excel 檔案匯入
            - 調整滑鼠十字線顯示報酬及 MDD 數值
            Bug Fixes:
            - 修復日期格式解析問題
            - 優化圖表顯示效果
            - 修正最大下單金額顯示錯誤

            v0.1.0 (2024-10-30)
            - 初始版本發布
            - 支援 XQ 簡易交易回報 CSV 檔案匯入
            - 提供累積報酬和最大回撤圖表
            - 支援策略相關性分析
        """

        # 日期選擇按鈕
        self.date_frame = None
        self.year_var = tk.StringVar(value=str(datetime.now().year))
        self.month_var = tk.StringVar(value=str(datetime.now().month))

        self.setup_ui()

    # ========== UI Setup ========== #

    def setup_ui(self):
        self._setup_left_panel()
        self._setup_dropdown_menu()

    def _setup_left_panel(self):
        self.button_frame = tk.Frame(self.root, height=self.root.winfo_screenheight())
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 添加按鈕
        tk.Button(self.button_frame, text="顯示累積報酬",
                  command=self.show_cumulative_returns).pack(side=tk.TOP, pady=5, anchor='nw')
        tk.Button(self.button_frame, text="分析策略相關性",
                  command=self.analyze_correlation).pack(side=tk.TOP, pady=5, anchor='nw')
        tk.Button(self.button_frame, text="顯示日獲利",
                  command=self.show_daily_returns).pack(side=tk.TOP, pady=5, anchor='nw')
        tk.Button(self.button_frame, text="顯示月獲利",
                  command=self.show_monthly_returns).pack(side=tk.TOP, pady=5, anchor='nw')

        tk.Button(self.button_frame, text="匯入報表",
                  command=self.import_reports).pack(side=tk.BOTTOM, ipady=15, anchor='sw')
        tk.Button(self.button_frame, text="關閉報表",
                  command=self.close_reports).pack(side=tk.BOTTOM, pady=10, ipady=10, anchor='sw')
        tk.Button(self.button_frame, text="設定策略參數",
                  command=self.show_strategy_config).pack(side=tk.BOTTOM, pady=5, anchor='sw')
        tk.Button(self.button_frame, text="版本更新記錄",
                  command=self.show_changelog).pack(side=tk.BOTTOM, pady=5, anchor='sw')

    def _setup_dropdown_menu(self):
        self.dropdown_frame = tk.Frame(self.button_frame)
        self.dropdown_frame.pack(side=tk.TOP, pady=5, anchor='nw')

        tk.Label(self.dropdown_frame, text="選擇重複商品欲優先執行策略:").pack(side=tk.TOP, pady=2)

        self.report_var = tk.StringVar()
        self.report_dropdown = tk.OptionMenu(self.dropdown_frame, self.report_var, "")
        self.report_dropdown.pack(side=tk.LEFT)

        def on_selection_change(*args):
            selected = self.report_var.get()
            if selected:
                self._reset_data()
                self.filter_overlapping_trades(selected)

        self.report_var.trace('w', on_selection_change)

        tk.Button(self.dropdown_frame, text="清除選擇",
                  command=self._clear_selection).pack(side=tk.RIGHT, pady=2)

    # ========== Data Management ========== #

    def import_reports(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        for file_path in file_paths:
            name = file_path.split('/')[-1]
            # 添加策略到配置管理器
            self.config_manager.add_strategy(name)
            # 使用配置管理器处理文件
            df, daily_returns = DataProcessor.process_file(file_path, name, self.config_manager)
            stats = DataProcessor.calculate_stats(df)
            _, _, _, max_dd = DataProcessor.calculate_drawdown(daily_returns)

            self.trading_data[name] = daily_returns
            self.strategy_stats[name] = {
                'df': df,
                'stats': stats,
                'max_dd': max_dd
            }

            # 保存原始數據
            self.original_data[name] = {
                'df': df.copy(),
                'daily_returns': daily_returns.copy(),
                'stats': stats.copy(),
                'max_dd': max_dd.copy()
            }

        self._update_dropdown_menu()
        print("已匯入報表：", list(self.trading_data.keys()))
        self.show_cumulative_returns()

    def filter_overlapping_trades(self, selected_report):
        if selected_report and selected_report in self.strategy_stats:
            selected_df = self.strategy_stats[selected_report]['df']

            for name in self.strategy_stats:
                if name != selected_report:
                    df = self.original_data[name]['df'].copy()
                    filtered_df = DataProcessor.filter_overlapping_trades(selected_df, df)

                    daily_returns = filtered_df.groupby('日期')['獲利金額'].sum()
                    stats = DataProcessor.calculate_stats(filtered_df)
                    _, _, _, max_dd = DataProcessor.calculate_drawdown(daily_returns)

                    self.trading_data[name] = daily_returns
                    self.strategy_stats[name] = {
                        'df': filtered_df,
                        'stats': stats,
                        'max_dd': max_dd
                    }
                else:
                    self.trading_data[name] = self.original_data[name]['daily_returns']
                    self.strategy_stats[name] = {
                        'df': self.original_data[name]['df'],
                        'stats': self.original_data[name]['stats'],
                        'max_dd': self.original_data[name]['max_dd']
                    }
            self.show_cumulative_returns()
        else:
            print("未選擇有效的報表。")

    def _reset_data(self):
        self.trading_data.update({name: self.original_data[name]['daily_returns'] for name in self.original_data})
        self.strategy_stats.update({name: {
            'df': self.original_data[name]['df'],
            'stats': self.original_data[name]['stats'],
            'max_dd': self.original_data[name]['max_dd']
        } for name in self.original_data})

    def _clear_selection(self):
        self.report_var.set('')
        self._reset_data()
        self.show_cumulative_returns()

    def _update_dropdown_menu(self):
        menu = self.report_dropdown["menu"]
        menu.delete(0, "end")
        for name in self.trading_data.keys():
            menu.add_command(label=name, command=lambda n=name: self.report_var.set(n))

    # ========== Chart Display Methods ========== #

    def show_strategy_config(self):
        self.strategy_config_window = tk.Toplevel(self.root)
        self.strategy_config_window.title("策略參數設定")
        self.strategy_config_window.geometry("400x300")

        # 建立設定框架
        config_frame = tk.Frame(self.strategy_config_window)
        config_frame.pack(padx=10, pady=10)

        # 手續費折數
        fee_discount_var = tk.DoubleVar(value=0.28)
        tk.Label(config_frame, text="手續費折數:").grid(row=0, column=0, sticky='w', pady=5)
        fee_discount_entry = tk.Entry(config_frame, textvariable=fee_discount_var)
        fee_discount_entry.grid(row=0, column=1, sticky='w', pady=5)

        # 策略選擇下拉選單
        tk.Label(config_frame, text="選擇策略:").grid(row=1, column=0, sticky='w', pady=5)
        strategy_var = tk.StringVar()
        strategy_dropdown = tk.OptionMenu(config_frame, strategy_var, *self.trading_data.keys())
        strategy_dropdown.grid(row=1, column=1, sticky='w', pady=5)

        # 策略類型選擇
        strategy_type_var = tk.StringVar(value='short')
        tk.Radiobutton(config_frame, text="多方", variable=strategy_type_var, value='long').grid(row=2, column=0, sticky='w', pady=5)
        tk.Radiobutton(config_frame, text="空方", variable=strategy_type_var, value='short').grid(row=2, column=1, sticky='w', pady=5)

        # 是否當沖選擇
        tk.Label(config_frame, text="是否為當沖:").grid(row=3, column=0, sticky='w', pady=5)
        is_margin_var = tk.BooleanVar(value=True)
        tk.Checkbutton(config_frame, text="是否為當沖", variable=is_margin_var).grid(row=3, column=1, sticky='w', pady=5)

        # 張數/零股設定
        use_lot_var = tk.BooleanVar(value=True)
        tk.Checkbutton(config_frame, text="使用張數計算(取消勾選為零股)", variable=use_lot_var).grid(row=4, column=0, columnspan=2, sticky='w', pady=5)

        # 固定下單金額設定
        tk.Label(config_frame, text="固定下單金額 (選填):").grid(row=5, column=0, sticky='w', pady=5)
        amount_entry = tk.Entry(config_frame)
        amount_entry.grid(row=5, column=1, sticky='w', pady=5)

        # 當選擇策略時更新設定值
        def update_config(*args):
            strategy = strategy_var.get()
            if strategy:
                config = self.config_manager.get_strategy_config(strategy)
                fee_discount_var.set(config.fee_discount)
                strategy_type_var.set(config.strategy_type)
                is_margin_var.set(config.is_margin)
                use_lot_var.set(config.use_lot)
                if config.fixed_order_amount is not None:
                    amount_entry.delete(0, tk.END)
                    amount_entry.insert(0, str(config.fixed_order_amount))
                else:
                    amount_entry.delete(0, tk.END)

        strategy_var.trace('w', update_config)

        # 儲存按鈕
        def save_config():
            strategy = strategy_var.get()
            if strategy:
                config = self.config_manager.get_strategy_config(strategy)
                config.set_fee_discount(fee_discount_var.get())
                config.set_use_lot(use_lot_var.get())
                config.set_strategy_type(strategy_type_var.get())
                config.set_is_margin(is_margin_var.get())
                try:
                    amount_text = amount_entry.get()
                    amount = float(amount_text) if amount_text else None
                    config.set_fixed_order_amount(amount)
                except ValueError:
                    messagebox.showerror("錯誤", "請輸入有效的金額數字")
                    return
                messagebox.showinfo("成功", "策略設定已儲存，圖表將更新")
                # 更新数据
                df = self.original_data[strategy]['df']
                # 重新計算下單金額和獲利金額
                df['下單金額'] = df.apply(lambda row: config.calculate_order_amount(row['進場價格'], row['交易數量']), axis=1)
                df['獲利金額'] = df.apply(lambda row: config.calculate_profit(row['進場價格'], row['出場價格'], row['交易數量']), axis=1)
                # 更新每日收益和統計數據
                daily_returns = df.groupby('日期')['獲利金額'].sum()
                stats = DataProcessor.calculate_stats(df)
                _, _, _, max_dd = DataProcessor.calculate_drawdown(daily_returns)
                # 更新 trading_data 和 strategy_stats
                self.trading_data[strategy] = daily_returns
                self.strategy_stats[strategy] = {
                    'df': df,
                    'stats': stats,
                    'max_dd': max_dd
                }
                self.show_cumulative_returns()
                self.strategy_config_window.destroy()
            else:
                messagebox.showerror("錯誤", "請選擇策略")

        tk.Button(config_frame, text="儲存設定", command=save_config).grid(row=6, column=0, columnspan=2, pady=20)

        

    def show_cumulative_returns(self):
        self._clear_canvas()
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return

        self._clear_date_frame()
        fig = self.chart_plotter.plot_returns_and_drawdown(self.trading_data, self.strategy_stats)
        self._update_canvas(fig)

    def analyze_correlation(self):
        self._clear_canvas()
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return

        self._clear_date_frame()
        all_returns = pd.DataFrame(self.trading_data).fillna(0)
        fig = self.chart_plotter.plot_correlation(all_returns)
        self._update_canvas(fig)

    def show_daily_returns(self):
        self._clear_canvas()
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return

        self._setup_date_frame('month')
        fig = self.chart_plotter.plot_daily_returns(self.trading_data)
        self._update_canvas(fig)

    def show_monthly_returns(self):
        self._clear_canvas()
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return

        self._setup_date_frame('year')
        fig = self.chart_plotter.plot_monthly_returns(self.trading_data)
        self._update_canvas(fig)

    # ========== UI Helper Methods ========== #

    def _clear_canvas(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.changelog_text:
            self.changelog_text.destroy()
            self.changelog_text = None

    def _clear_date_frame(self):
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None

    def _setup_date_frame(self, mode):
        self._clear_date_frame()
        self.date_frame = tk.Frame(self.root)

        tk.Label(self.date_frame, text="年:").pack(side=tk.LEFT)
        year_entry = tk.Entry(self.date_frame, textvariable=self.year_var, width=6)
        year_entry.pack(side=tk.LEFT, padx=2)

        if mode == 'month':
            tk.Label(self.date_frame, text="月:").pack(side=tk.LEFT)
            month_entry = tk.Entry(self.date_frame, textvariable=self.month_var, width=4)
            month_entry.pack(side=tk.LEFT, padx=2)

            tk.Button(self.date_frame, text="跳轉",
                      command=self.jump_to_month).pack(side=tk.LEFT, padx=5)
            tk.Button(self.date_frame, text="上個月",
                      command=self.prev_month).pack(side=tk.LEFT, padx=2)
            tk.Button(self.date_frame, text="下個月",
                      command=self.next_month).pack(side=tk.LEFT, padx=2)
        elif mode == 'year':
            tk.Button(self.date_frame, text="跳轉",
                      command=self.jump_to_year).pack(side=tk.LEFT, padx=5)
            tk.Button(self.date_frame, text="上一年",
                      command=self.prev_year).pack(side=tk.LEFT, padx=2)
            tk.Button(self.date_frame, text="下一年",
                      command=self.next_year).pack(side=tk.LEFT, padx=2)

        self.date_frame.pack(side=tk.TOP, pady=5)

    def _update_canvas(self, fig):
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().config(width=800, height=600)

    # ========== Navigation Methods ========== #

    def jump_to_year(self):
        try:
            year = int(self.year_var.get())
            self.chart_plotter.set_year(year)
            self.show_monthly_returns()
        except ValueError:
            print("請輸入有效的年份")

    def prev_year(self):
        self.chart_plotter.prev_year()
        self.year_var.set(str(self.chart_plotter.current_year))
        self.show_monthly_returns()

    def next_year(self):
        self.chart_plotter.next_year()
        self.year_var.set(str(self.chart_plotter.current_year))
        self.show_monthly_returns()

    def jump_to_month(self):
        try:
            year = int(self.year_var.get())
            month = int(self.month_var.get())
            if 1 <= month <= 12:
                self.chart_plotter.set_month(year, month)
                self.show_daily_returns()
            else:
                print("月份必須在1-12之間")
        except ValueError:
            print("請輸入有效的年月")

    def prev_month(self):
        self.chart_plotter.prev_month()
        self.year_var.set(str(self.chart_plotter.current_month.year))
        self.month_var.set(str(self.chart_plotter.current_month.month))
        self.show_daily_returns()

    def next_month(self):
        self.chart_plotter.next_month()
        self.year_var.set(str(self.chart_plotter.current_month.year))
        self.month_var.set(str(self.chart_plotter.current_month.month))
        self.show_daily_returns()

    # ========== Other Methods ========== #

    def show_changelog(self):
        self._clear_canvas()
        self.changelog_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=30)
        self.changelog_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.changelog_text.insert(tk.END, self.changelog_content)
        self.changelog_text.config(state=tk.DISABLED)

    def close_reports(self):
        self.trading_data.clear()
        self.strategy_stats.clear()
        self.config_manager.clear_all_configs()  # 清除所有策略設定
        print("已關閉所有報表。")
        self._clear_canvas()
        self._clear_date_frame()

# ===================== Main Execution ===================== #

def on_closing():
    root.destroy()
    root.quit()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.geometry("1400x900")
        root.protocol("WM_DELETE_WINDOW", on_closing)
        app = StrategyAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        with open("error_log.txt", "w") as log_file:
            log_file.write("An error occurred:\n")
            log_file.write(str(e) + "\n")
            log_file.write(traceback.format_exc())
        raise
