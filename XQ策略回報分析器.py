import tkinter as tk
from tkinter import filedialog, scrolledtext
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        daily_returns = df.groupby('日期')['獲利金額'].sum()
        
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
        
    def plot_correlation(self, all_returns):
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = all_returns.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title('策略相關性分析')
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
        def hover(event):
            if event.inaxes not in [ax1, ax2]:
                return
                
            # 清除舊的十字線
            for line in [self.vertical_line1, self.horizontal_line1, 
                        self.vertical_line2, self.horizontal_line2]:
                if line is not None:
                    line.remove()
                    
            # 繪製新的十字線
            if event.inaxes == ax1:
                self.vertical_line1 = ax1.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.5)
                self.horizontal_line1 = ax1.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.5)
            elif event.inaxes == ax2:
                self.vertical_line2 = ax2.axvline(x=event.xdata, color='gray', linestyle='--', alpha=0.5)
                self.horizontal_line2 = ax2.axhline(y=event.ydata, color='gray', linestyle='--', alpha=0.5)
                
            # 更新標題
            date_str = matplotlib.dates.num2date(event.xdata).strftime('%Y-%m-%d')
            ax1.set_title(f"滑鼠位置日期：{date_str}")
            ax2.set_title('最大回撤 (MDD)')
            
            fig.canvas.draw_idle()
            
        fig.canvas.mpl_connect('motion_notify_event', hover)

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
            版本更新記錄：

            v0.1.0 (2024-10-30)
            - 初始版本發布
            - 支援XQ簡易交易回報CSV檔案匯入
            - 提供累積報酬和最大回撤圖表
            - 支援策略相關性分析

            v0.1.0 (2024-10-31)
            - 修復日期格式解析問題
            - 優化圖表顯示效果
            - 修正最大下單金額顯示錯誤
            - 支援XQ完整交易回報Excel檔案匯入
            - 新增版本更新記錄頁面
        """
        
    def setup_ui(self):
        # 創建左側按鈕框架
        self.button_frame = tk.Frame(self.root, height=self.root.winfo_screenheight())
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 添加按鈕
        tk.Button(self.button_frame, text="顯示累積報酬", 
                 command=self.show_cumulative_returns).pack(side=tk.TOP, pady=5, anchor='nw')
        tk.Button(self.button_frame, text="分析策略相關性", 
                 command=self.analyze_correlation).pack(side=tk.TOP, pady=5, anchor='nw')
        tk.Button(self.button_frame, text="版本更新記錄", 
                 command=self.show_changelog).pack(side=tk.TOP, pady=5, anchor='nw')
        tk.Button(self.button_frame, text="匯入報表", 
                 command=self.import_reports).pack(side=tk.BOTTOM, ipady=15, anchor='sw')
        tk.Button(self.button_frame, text="關閉報表", 
                 command=self.close_reports).pack(side=tk.BOTTOM, pady=10, ipady=10, anchor='sw')
        
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
            
        all_returns = pd.DataFrame(self.trading_data).fillna(0)
        fig = self.chart_plotter.plot_correlation(all_returns)
        self.update_canvas(fig)
        
    def show_cumulative_returns(self):
        if not self.trading_data:
            print("尚未匯入任何報表。")
            return
            
        fig = self.chart_plotter.plot_returns_and_drawdown(self.trading_data, self.strategy_stats)
        self.update_canvas(fig)
        
    def show_changelog(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        if self.changelog_text:
            self.changelog_text.destroy()
            
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
    root.geometry("800x600")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = StrategyAnalyzerApp(root)
    root.mainloop()
