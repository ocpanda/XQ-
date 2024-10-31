import calendar
import tkinter as tk
from tkinter import filedialog, scrolledtext
import matplotlib
from matplotlib import ticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta

# 設定matplotlib使用微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    @staticmethod
    def process_file(file_path):
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_path, encoding='ansi')
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_path, sheet_name='交易分析')
            df['獲利金額'] = df['獲利金額'].str.replace(',', '').astype(float)
        else:
            raise ValueError("不支持的文件格式。請使用CSV或XLSX文件。")
            
        df['進場時間'] = pd.to_datetime(df['進場時間'], format='%Y/%m/%d %H:%M')
        df.sort_values(by='進場時間', inplace=True)
        df['日期'] = df['進場時間'].dt.date
        daily_returns = pd.Series(df.groupby('日期')['獲利金額'].sum(), index=pd.DatetimeIndex(df.groupby('日期').groups.keys()))
        
        return df, daily_returns
        
    @staticmethod
    def calculate_stats(df):
        total_trades = len(df)
        win_trades = len(df[df['獲利金額'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        average_profit = df['獲利金額'].mean() if total_trades > 0 else 0
        df['下單金額'] = df['進場價格'] * 1000 * df['交易數量']
        daily_order_amount = df.groupby('日期')['下單金額'].sum()
        maximum_order_amount = daily_order_amount.max() if total_trades > 0 else 0
        max_order_date = daily_order_amount.idxmax() if total_trades > 0 else None
        print(f"下單金額最大的日期是: {max_order_date}")
        print(f"當日下單金額為: {maximum_order_amount:,.0f}")
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

class ChartPlotter:
    def __init__(self):
        self.vertical_line1 = None
        self.horizontal_line1 = None
        self.vertical_line2 = None
        self.horizontal_line2 = None
        self.stats_text = None
        self.current_month = datetime.now()
        self.current_year = datetime.now().year
        
    def plot_correlation(self, all_returns):
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = all_returns.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title('策略相關性分析')
        return fig

    def plot_monthly_returns(self, trading_data):
        # 創建兩個子圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.2, 0.8])

        # 調整子圖之間的間距
        plt.subplots_adjust(hspace=0.3)  # 調整間距參數 hspace
        
        # 獲取當前年份的所有月份
        year_start = pd.Timestamp(self.current_year, 1, 1)
        year_end = pd.Timestamp(self.current_year, 12, 31)
        
        # 合併所有策略的月份索引
        monthly_data = {}
        for name, returns in trading_data.items():
            # 將日資料轉換為月資料
            monthly_returns = returns.resample('M').sum()
            monthly_data[name] = monthly_returns
        
        # 繪製個別策略柱狀圖
        months = pd.date_range(start=year_start, end=year_end, freq='M')
        bar_width = 0.8 / (len(trading_data) + 1)  # +1 是為了留出合併策略的空間
        bars = []
        
        for i, (name, returns) in enumerate(monthly_data.items()):
            monthly_values = returns.reindex(months, fill_value=0)
            bar_container = ax1.bar(np.arange(12) + i * bar_width,
                                 monthly_values.values,
                                 bar_width,
                                 label=name,
                                 alpha=0.7,
                                 picker=True)
            for rect in bar_container:
                rect.custom_label = name
            bars.extend(bar_container)
            
            # 在第二個子圖繪製年度累積折線圖
            yearly_returns = returns.resample('Y').sum()
            ax2.plot(yearly_returns.index.year, yearly_returns.values,
                    marker='o', label=name, linewidth=2)
            
        # 計算並繪製合併策略
        combined_monthly = pd.DataFrame(monthly_data).fillna(0).sum(axis=1)
        bar_container = ax1.bar(np.arange(12) + len(trading_data) * bar_width,
                             combined_monthly.reindex(months, fill_value=0).values,
                             bar_width,
                             label='策略合併執行',
                             alpha=0.7,
                             picker=True)
        for rect in bar_container:
            rect.custom_label = '策略合併執行'
        bars.extend(bar_container)
        
        # 在第二個子圖繪製合併策略的年度累積折線
        combined_yearly = combined_monthly.resample('Y').sum()
        ax2.plot(combined_yearly.index.year, combined_yearly.values,
                marker='o', label='策略合併執行', linewidth=2, linestyle='--')
        
        # 在ax2添加0軸黑色虛線
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 設置圖表屬性
        # 只計算當年度的總獲利
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
        
        # 設置x軸刻度
        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月', 
                       '7月', '8月', '9月', '10月', '11月', '12月']
        ax1.set_xticks(np.arange(12) + (len(trading_data) * bar_width) / 2)
        ax1.set_xticklabels(month_labels)
        
        # 添加網格線
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # 設置ax2的y軸格式,避免使用科學符號
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # 添加滑鼠互動效果
        def hover(event):
            if event.inaxes != ax1:
                return
            
            # 找出滑鼠所在的月份索引
            x_pos = event.xdata
            month_index = int(x_pos + 0.5)
            
            # 清除所有文字
            for text in ax1.texts:
                text.remove()
                
            # 找出該月所有柱狀圖的值
            month_values = []
            for bar in bars:
                bar_x = bar.get_x()
                bar_month = int(bar_x + 0.5)
                if bar_month == month_index:
                    if not hasattr(bar, 'original_height'):
                        bar.original_height = bar.get_height()
                    month_values.append((bar, bar.original_height))
            
            # 計算要顯示的值的數量
            total_strategies = len(trading_data) + 1
            
            # 遍歷所有柱狀圖
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
        
        return fig

    def plot_daily_returns(self, trading_data):
        # 創建兩個子圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1.2, 0.8])

        # 調整子圖之間的間距
        plt.subplots_adjust(hspace=0.3)  # 調整間距參數 hspace
        
        # 獲取當前月份的起始和結束日期
        month_start = pd.Timestamp(self.current_month.year, self.current_month.month, 1)
        if self.current_month.month == 12:
            month_end = pd.Timestamp(self.current_month.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = pd.Timestamp(self.current_month.year, self.current_month.month + 1, 1) - timedelta(days=1)

        # 合併所有策略的日期索引
        all_dates = pd.date_range(start=month_start, end=month_end)
        combined_returns = pd.Series(0, index=all_dates)
        
        # 繪製個別策略柱狀圖
        bar_width = 0.8 / (len(trading_data) + 1)  # +1 是為了留出合併策略的空間
        bars = []  # 儲存所有柱狀圖對象
        for i, (name, returns) in enumerate(trading_data.items()):
            daily_data = returns.reindex(all_dates, fill_value=0)
            bar_container = ax1.bar(np.arange(len(all_dates)) + i * bar_width, 
                                 daily_data.values,
                                 bar_width,
                                 label=name,
                                 alpha=0.7,
                                 picker=True)
            for rect in bar_container:
                rect.custom_label = name
            bars.extend(bar_container)
            combined_returns += daily_data

        # 繪製合併策略柱狀圖
        bar_container = ax1.bar(np.arange(len(all_dates)) + len(trading_data) * bar_width,
                             combined_returns.values,
                             bar_width,
                             label='策略合併執行',
                             alpha=0.7,
                             picker=True)
        for rect in bar_container:
            rect.custom_label = '策略合併執行'
        bars.extend(bar_container)

        # 設置第一個子圖屬性
        total_profit = combined_returns.sum()
        ax1.set_title(f'{self.current_month.year}年{self.current_month.month}月 日獲利分析 (總獲利: {total_profit:,.0f})')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('獲利金額')
        ax1.legend(loc='upper left')

        # 設置x軸刻度
        ax1.set_xticks(np.arange(len(all_dates)) + (len(trading_data) * bar_width) / 2)
        ax1.set_xticklabels([d.strftime('%m/%d') for d in all_dates], rotation=45)

        # 添加網格線
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

        # 繪製月度獲利趨勢圖
        for name, returns in trading_data.items():
            # 計算月度獲利
            monthly_returns = returns.resample('M').sum()
            ax2.plot(monthly_returns.index, monthly_returns.values, label=name, marker='o')

        # 計算並繪製合併策略的月度獲利
        combined_monthly = pd.DataFrame(trading_data).fillna(0).sum(axis=1).resample('M').sum()
        ax2.plot(combined_monthly.index, combined_monthly.values, 
                label='策略合併執行', marker='o', linewidth=2, linestyle='--')
        
        # 在ax2添加0軸黑色虛線
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 設置第二個子圖屬性
        ax2.set_title('月度獲利趨勢')
        ax2.set_xlabel('月份')
        ax2.set_ylabel('月獲利金額')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 調整子圖間距
        plt.tight_layout()

        # 添加滑鼠互動效果
        def hover(event):
            if event.inaxes != ax1:
                return
            
            # 找出滑鼠所在的日期索引
            x_pos = event.xdata
            day_index = int(x_pos + 0.5) # 加0.5進行四捨五入修正位置偏移
            
            # 清除所有文字
            for text in ax1.texts:
                text.remove()
                
            # 先找出該天所有柱狀圖的值
            day_values = []
            for bar in bars:
                bar_x = bar.get_x()
                bar_day = int(bar_x + 0.5)
                if bar_day == day_index:
                    if not hasattr(bar, 'original_height'):
                        bar.original_height = bar.get_height()
                    day_values.append((bar, bar.original_height))
            
            # 計算要顯示的值的數量 (策略數量 + 1)
            total_strategies = len(trading_data) + 1
            
            # 遍歷所有柱狀圖
            for bar_index, bar in enumerate(bars):
                bar_x = bar.get_x()
                bar_day = int(bar_x + 0.5)
                
                # 如果柱狀圖屬於同一天
                if bar_day == day_index:
                    # 保存原始高度
                    if not hasattr(bar, 'original_height'):
                        bar.original_height = bar.get_height()
                    
                    # 放大柱狀圖
                    bar.set_height(bar.original_height * 1.1)
                    
                    # 在右上角顯示數值
                    # 從bar的label屬性中取得label
                    label = bar.custom_label
                    value = bar.original_height
                    
                    # 根據總策略數計算垂直位移
                    vertical_offset = 0.98 - (0.05 * total_strategies)
                    current_index = len(ax1.texts)
                    text_position = vertical_offset + (0.05 * current_index)
                    
                    ax1.text(0.98, text_position,
                           f'{label}: {value:,.0f}',
                           transform=ax1.transAxes,
                           ha='right',
                           va='top')
                else:
                    # 恢復原始高度
                    if hasattr(bar, 'original_height'):
                        bar.set_height(bar.original_height)
                        
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', hover)
        
        return fig
        
    def plot_returns_and_drawdown(self, trading_data, strategy_stats):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))
        fig.subplots_adjust(right=0.75)
        
        # 合併所有策略的日期索引
        all_dates = pd.Index([])
        for returns in trading_data.values():
            all_dates = all_dates.union(returns.index)
        all_dates = all_dates.sort_values()
        
        # 計算合併策略數據
        combined_returns_df = pd.DataFrame(trading_data).fillna(0)
        combined_returns = combined_returns_df.sum(axis=1)
        combined_returns = combined_returns.reindex(all_dates, fill_value=0)
        combined_cum_returns, combined_peak, combined_dd, combined_max_dd = DataProcessor.calculate_drawdown(combined_returns)
        
        # 繪製合併策略曲線
        ax1.plot(combined_cum_returns.index, combined_cum_returns.values, label='策略合併執行', linewidth=2, linestyle='--')
        ax2.plot(combined_max_dd.index, combined_max_dd.values, label='策略合併執行', linewidth=2, linestyle='--')
        
        # 繪製個別策略曲線
        for name, returns in trading_data.items():
            returns = returns.reindex(all_dates, fill_value=0)
            cum_returns, peak, dd, max_dd = DataProcessor.calculate_drawdown(returns)
            
            ax1.plot(cum_returns.index, cum_returns.values, label=name)
            ax2.plot(max_dd.index, max_dd.values, label=name)
        
        # 設置圖表屬性
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
        
        # 添加統計資訊
        stats_text = self.generate_stats_text(trading_data, strategy_stats, combined_max_dd)
        self.stats_text = fig.text(0.77, 0.5, stats_text, va='center', fontsize=10)
        
        # 添加互動功能
        self.add_hover_effect(fig, ax1, ax2)
        
        return fig
        
    def generate_stats_text(self, trading_data, strategy_stats, combined_max_dd):
        stats_text_list = []
        
        # 合併策略統計
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
        
        # 個別策略統計
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
        
    def add_hover_effect(self, fig, ax1, ax2):
        # 初始化十字線
        self.vertical_line1 = None
        self.horizontal_line1 = None
        self.vertical_line2 = None 
        self.horizontal_line2 = None

        def hover(event):
            if event.inaxes not in [ax1, ax2]:
                return
                
            # 清除舊的十字線
            for line in [self.vertical_line1, self.horizontal_line1, 
                        self.vertical_line2, self.horizontal_line2]:
                if line is not None:
                    try:
                        line.remove()
                    except ValueError:
                        pass
                    
            # 繪製新的十字線
            if event.inaxes == ax1:
                self.vertical_line1 = ax1.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.5)
                self.horizontal_line1 = ax1.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.5)
            elif event.inaxes == ax2:
                self.vertical_line2 = ax2.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.5)
                self.horizontal_line2 = ax2.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.5)
                
            # 更新標題
            date_str = matplotlib.dates.num2date(event.xdata).strftime('%Y-%m-%d')
            
            # 取得目前滑鼠位置的所有曲線數值
            title_text = f"滑鼠位置日期：{date_str}\n"
            # 找到最接近的數據點
            x_data = ax1.lines[0].get_xdata()
            x_data_num = matplotlib.dates.date2num(x_data)
            x_idx = np.argmin(abs(x_data_num - event.xdata))
            
            # 取得每條曲線在該位置的值
            for line in ax1.lines:
                label = line.get_label()
                y_data = line.get_ydata()
                if x_idx < len(y_data):  # 確保索引不超出範圍
                    y_value = y_data[x_idx]
                    
                    # 找到對應的MDD值
                    for mdd_line in ax2.lines:
                        if mdd_line.get_label() == label:
                            mdd_data = mdd_line.get_ydata()
                            if x_idx < len(mdd_data):  # 確保索引不超出範圍
                                mdd_value = mdd_data[x_idx]
                                title_text += f"{label} - 累積獲利: {y_value:,.0f}, MDD: {-mdd_value:,.0f}\n"
                            break
            
            ax1.set_title(title_text)
            ax2.set_title('最大回撤 (MDD)')
            
            fig.canvas.draw_idle()
            
        fig.canvas.mpl_connect('motion_notify_event', hover)

    def next_month(self):
        if self.current_month.month == 12:
            self.current_month = self.current_month.replace(year=self.current_month.year + 1, month=1)
        else:
            self.current_month = self.current_month.replace(month=self.current_month.month + 1)

    def prev_month(self):
        if self.current_month.month == 1:
            self.current_month = self.current_month.replace(year=self.current_month.year - 1, month=12)
        else:
            try:
                self.current_month = self.current_month.replace(month=self.current_month.month - 1)
            except ValueError:
                # 如果日期超出範圍,則設為上個月的最後一天
                if self.current_month.month == 1:
                    new_year = self.current_month.year - 1
                    new_month = 12
                else:
                    new_year = self.current_month.year
                    new_month = self.current_month.month - 1
                last_day = calendar.monthrange(new_year, new_month)[1]
                self.current_month = self.current_month.replace(year=new_year, month=new_month, day=last_day)
                
    def set_month(self, year, month):
        self.current_month = datetime(year, month, 1)
        
    def next_year(self):
        self.current_year += 1
        
    def prev_year(self):
        self.current_year -= 1
        
    def set_year(self, year):
        self.current_year = year

class StrategyAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XQ 策略回測報表分析器")
        
        self.setup_ui()
        
        # 數據存儲
        self.trading_data = {}  # 存儲每日獲利
        self.strategy_stats = {}  # 存儲策略統計信息
        
        # 圖表相關
        self.canvas = None
        self.chart_plotter = ChartPlotter()
        
        # Changelog相關
        self.changelog_text = None
        self.changelog_content = """
            Powered by ocpanda at https://github.com/ocpanda

            版本更新記錄：

            v0.1.0 (2024-10-30)
            - 初始版本發布
            - 支援XQ簡易交易回報CSV檔案匯入
            - 提供累積報酬和最大回撤圖表
            - 支援策略相關性分析

            v0.2.0 (2024-10-31)
            Features:
            - 新增版本更新記錄頁面
            - 支援XQ完整交易回報Excel檔案匯入
            - 調整滑鼠十字線顯示報酬及MDD數值
            Bug Fixes:
            - 修復日期格式解析問題
            - 優化圖表顯示效果
            - 修正最大下單金額顯示錯誤

            v0.3.0 (2024-10-31)
            Features:
            - 新增日獲利圖表
            - 新增月獲利圖表
        """
        
        # 日期選擇按鈕
        self.date_frame = None
        self.year_var = tk.StringVar(value=str(datetime.now().year))
        self.month_var = tk.StringVar(value=str(datetime.now().month))
        
    def setup_ui(self):
        # 創建左側按鈕框架
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
        tk.Button(self.button_frame, text="版本更新記錄", 
                 command=self.show_changelog).pack(side=tk.BOTTOM, pady=5, anchor='sw')
                 
    def show_monthly_returns(self):
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return
            
        # 清除現有的日期選擇框架
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None
            
        # 創建或顯示日期選擇框架
        self.date_frame = tk.Frame(self.root)
            
        # 年份選擇
        tk.Label(self.date_frame, text="年:").pack(side=tk.LEFT)
        year_entry = tk.Entry(self.date_frame, textvariable=self.year_var, width=6)
        year_entry.pack(side=tk.LEFT, padx=2)
            
        # 跳轉按鈕
        tk.Button(self.date_frame, text="跳轉", 
                 command=self.jump_to_year).pack(side=tk.LEFT, padx=5)
            
        # 上/下一年按鈕
        tk.Button(self.date_frame, text="上一年", 
                 command=self.prev_year).pack(side=tk.LEFT, padx=2)
        tk.Button(self.date_frame, text="下一年", 
                 command=self.next_year).pack(side=tk.LEFT, padx=2)
        
        self.date_frame.pack(side=tk.TOP, pady=5)
            
        fig = self.chart_plotter.plot_monthly_returns(self.trading_data)
        self.update_canvas(fig)
        
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
        
    def import_reports(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        for file_path in file_paths:
            name = file_path.split('/')[-1]
            df, daily_returns = DataProcessor.process_file(file_path)
            stats = DataProcessor.calculate_stats(df)
            _, _, _, max_dd = DataProcessor.calculate_drawdown(daily_returns)
            
            self.trading_data[name] = daily_returns
            self.strategy_stats[name] = {
                'df': df,
                'stats': stats,
                'max_dd': max_dd
            }
            
        print("已匯入報表：", list(self.trading_data.keys()))
        self.show_cumulative_returns()
        
    def analyze_correlation(self):
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return
            
        # 清除現有的日期選擇框架
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None
            
        all_returns = pd.DataFrame(self.trading_data).fillna(0)
        fig = self.chart_plotter.plot_correlation(all_returns)
        self.update_canvas(fig)
        
    def show_cumulative_returns(self):
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return
            
        # 清除現有的日期選擇框架
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None
            
        fig = self.chart_plotter.plot_returns_and_drawdown(self.trading_data, self.strategy_stats)
        self.update_canvas(fig)

    def show_daily_returns(self):
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return
            
        # 清除現有的日期選擇框架
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None
            
        # 創建或顯示日期選擇框架
        self.date_frame = tk.Frame(self.root)
            
        # 年份選擇
        tk.Label(self.date_frame, text="年:").pack(side=tk.LEFT)
        year_entry = tk.Entry(self.date_frame, textvariable=self.year_var, width=6)
        year_entry.pack(side=tk.LEFT, padx=2)
            
        # 月份選擇
        tk.Label(self.date_frame, text="月:").pack(side=tk.LEFT)
        month_entry = tk.Entry(self.date_frame, textvariable=self.month_var, width=4)
        month_entry.pack(side=tk.LEFT, padx=2)
            
        # 跳轉按鈕
        tk.Button(self.date_frame, text="跳轉", 
                 command=self.jump_to_month).pack(side=tk.LEFT, padx=5)
            
        # 上/下個月按鈕
        tk.Button(self.date_frame, text="上個月", 
                 command=self.prev_month).pack(side=tk.LEFT, padx=2)
        tk.Button(self.date_frame, text="下個月", 
                 command=self.next_month).pack(side=tk.LEFT, padx=2)
        
        self.date_frame.pack(side=tk.TOP, pady=5)
            
        fig = self.chart_plotter.plot_daily_returns(self.trading_data)
        self.update_canvas(fig)

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
        
    def show_changelog(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        if self.changelog_text:
            self.changelog_text.destroy()
            
        # 清除現有的日期選擇框架
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None
            
        self.changelog_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=30)
        self.changelog_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.changelog_text.insert(tk.END, self.changelog_content)
        self.changelog_text.config(state=tk.DISABLED)
        
    def close_reports(self):
        self.trading_data.clear()
        self.strategy_stats.clear()
        print("已關閉所有報表。")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.changelog_text:
            self.changelog_text.destroy()
            self.changelog_text = None
        if self.date_frame:
            self.date_frame.pack_forget()
            self.date_frame = None
            
    def update_canvas(self, fig):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        if self.changelog_text:
            self.changelog_text.destroy()
            self.changelog_text = None
            
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().config(width=800, height=600)

def on_closing():
    root.destroy()
    root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x900")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = StrategyAnalyzerApp(root)
    root.mainloop()
