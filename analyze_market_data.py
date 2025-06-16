#!/usr/bin/env python3
"""
ESP32 Kraken Market Data Analysis Tool

This script analyzes the CSV data collected from your ESP32 Kraken Futures
market data collector. It provides comprehensive analysis of:
- Order flow patterns
- Price level dynamics  
- Market microstructure
- Liquidity analysis
- Visualization for market making strategy development

Usage:
    python analyze_market_data.py kraken_market_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import sys
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MarketDataAnalyzer:
    def __init__(self, csv_file):
        """Initialize analyzer with CSV data from ESP32."""
        self.csv_file = csv_file
        self.data = None
        self.symbol = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess CSV data."""
        try:
            # Load CSV data
            self.data = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.data)} market data points")
            
            # Convert timestamp to datetime
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp_ms'], unit='ms')
            
            # Get symbol
            self.symbol = self.data['symbol'].iloc[0] if len(self.data) > 0 else "Unknown"
            
            # Sort by timestamp
            self.data = self.data.sort_values('timestamp')
            
            # Add derived columns
            self.data['price_change'] = self.data['price'].diff()
            self.data['size_change'] = self.data['size'].diff()
            
            print(f"📊 Symbol: {self.symbol}")
            print(f"⏰ Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            print(f"💰 Price range: ${self.data['price'].min():.2f} - ${self.data['price'].max():.2f}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def basic_statistics(self):
        """Calculate basic market statistics."""
        print("\n" + "="*60)
        print("📈 BASIC MARKET STATISTICS")
        print("="*60)
        
        # Separate bid and ask data
        bids = self.data[self.data['side'] == 'buy']
        asks = self.data[self.data['side'] == 'sell']
        
        print(f"📊 Total Updates: {len(self.data):,}")
        print(f"📈 Bid Updates: {len(bids):,} ({len(bids)/len(self.data)*100:.1f}%)")
        print(f"📉 Ask Updates: {len(asks):,} ({len(asks)/len(self.data)*100:.1f}%)")
        
        # Price statistics
        print(f"\n💰 Price Statistics:")
        print(f"   Min Price: ${self.data['price'].min():.2f}")
        print(f"   Max Price: ${self.data['price'].max():.2f}")
        print(f"   Avg Price: ${self.data['price'].mean():.2f}")
        print(f"   Price Range: ${self.data['price'].max() - self.data['price'].min():.2f}")
        
        # Size statistics
        print(f"\n📦 Size Statistics:")
        print(f"   Min Size: {self.data['size'].min():.2f}")
        print(f"   Max Size: {self.data['size'].max():.2f}")
        print(f"   Avg Size: {self.data['size'].mean():.2f}")
        print(f"   Total Volume: {self.data['size'].sum():.2f}")
        
        # Update frequency
        duration = (self.data['timestamp'].max() - self.data['timestamp'].min()).total_seconds()
        frequency = len(self.data) / duration if duration > 0 else 0
        print(f"\n⚡ Update Frequency: {frequency:.2f} updates/second")
        
        return {
            'total_updates': len(self.data),
            'bid_updates': len(bids),
            'ask_updates': len(asks),
            'price_range': self.data['price'].max() - self.data['price'].min(),
            'avg_size': self.data['size'].mean(),
            'total_volume': self.data['size'].sum(),
            'update_frequency': frequency
        }
    
    def analyze_order_flow(self):
        """Analyze order flow patterns."""
        print("\n" + "="*60)
        print("🌊 ORDER FLOW ANALYSIS")
        print("="*60)
        
        # Order additions vs removals
        additions = self.data[self.data['size'] > 0]
        removals = self.data[self.data['size'] == 0]
        
        print(f"➕ Order Additions: {len(additions):,} ({len(additions)/len(self.data)*100:.1f}%)")
        print(f"➖ Order Removals: {len(removals):,} ({len(removals)/len(self.data)*100:.1f}%)")
        
        # Bid vs Ask flow
        bid_additions = len(additions[additions['side'] == 'buy'])
        ask_additions = len(additions[additions['side'] == 'sell'])
        
        print(f"\n📊 Addition Flow:")
        print(f"   Bid Additions: {bid_additions:,}")
        print(f"   Ask Additions: {ask_additions:,}")
        
        # Large orders
        large_threshold = self.data['size'].quantile(0.95)
        large_orders = self.data[self.data['size'] >= large_threshold]
        
        print(f"\n🐋 Large Orders (>={large_threshold:.0f} size):")
        print(f"   Count: {len(large_orders):,}")
        print(f"   Avg Size: {large_orders['size'].mean():.2f}")
        print(f"   Bid/Ask Split: {len(large_orders[large_orders['side']=='buy']):,} / {len(large_orders[large_orders['side']=='sell']):,}")
        
        return {
            'additions': len(additions),
            'removals': len(removals),
            'bid_additions': bid_additions,
            'ask_additions': ask_additions,
            'large_orders': len(large_orders)
        }
    
    def analyze_price_levels(self):
        """Analyze price level dynamics."""
        print("\n" + "="*60)
        print("📊 PRICE LEVEL ANALYSIS")
        print("="*60)
        
        # Most active price levels
        price_activity = self.data.groupby('price').size().sort_values(ascending=False)
        
        print("🔥 Most Active Price Levels:")
        for i, (price, count) in enumerate(price_activity.head(10).items()):
            print(f"   {i+1:2d}. ${price:8.1f} - {count:,} updates")
        
        # Price level statistics
        unique_prices = len(self.data['price'].unique())
        price_density = unique_prices / (self.data['price'].max() - self.data['price'].min())
        
        print(f"\n📈 Price Level Stats:")
        print(f"   Unique Price Levels: {unique_prices:,}")
        print(f"   Price Density: {price_density:.2f} levels per $")
        
        # Bid/Ask spread analysis
        bids = self.data[self.data['side'] == 'buy']['price']
        asks = self.data[self.data['side'] == 'sell']['price']
        
        if len(bids) > 0 and len(asks) > 0:
            best_bid = bids.max()
            best_ask = asks.min()
            spread = best_ask - best_bid
            spread_bps = (spread / ((best_bid + best_ask) / 2)) * 10000
            
            print(f"\n💰 Best Bid/Ask:")
            print(f"   Best Bid: ${best_bid:.2f}")
            print(f"   Best Ask: ${best_ask:.2f}")
            print(f"   Spread: ${spread:.2f} ({spread_bps:.1f} bps)")
        
        return {
            'unique_prices': unique_prices,
            'most_active_price': price_activity.index[0],
            'most_active_count': price_activity.iloc[0]
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Market Data Analysis - {self.symbol}', fontsize=16, fontweight='bold')
        
        # 1. Price over time
        bid_data = self.data[self.data['side'] == 'buy']
        ask_data = self.data[self.data['side'] == 'sell']
        
        if len(bid_data) > 0:
            ax1.scatter(bid_data['timestamp'], bid_data['price'], c='green', alpha=0.6, s=1, label='Bids')
        if len(ask_data) > 0:
            ax1.scatter(ask_data['timestamp'], ask_data['price'], c='red', alpha=0.6, s=1, label='Asks')
        
        ax1.set_title('Price Levels Over Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Order size distribution
        self.data[self.data['size'] > 0]['size'].hist(bins=50, ax=ax2, alpha=0.7, edgecolor='black')
        ax2.set_title('Order Size Distribution')
        ax2.set_xlabel('Order Size')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Price level activity heatmap
        price_bins = pd.cut(self.data['price'], bins=20)
        time_bins = pd.cut(self.data['timestamp'], bins=20)
        heatmap_data = self.data.groupby([time_bins, price_bins]).size().unstack(fill_value=0)
        
        sns.heatmap(heatmap_data, ax=ax3, cmap='YlOrRd', cbar_kws={'label': 'Activity'})
        ax3.set_title('Price Level Activity Heatmap')
        ax3.set_xlabel('Price Bins')
        ax3.set_ylabel('Time Bins')
        
        # 4. Bid vs Ask volume
        side_volume = self.data.groupby('side')['size'].sum()
        side_volume.plot(kind='bar', ax=ax4, color=['green', 'red'], alpha=0.7)
        ax4.set_title('Total Volume by Side')
        ax4.set_ylabel('Total Size')
        ax4.set_xticklabels(['Buy', 'Sell'], rotation=0)
        ax4.grid(True, alpha=0.3)
        
        # 5. Update frequency over time
        self.data.set_index('timestamp').resample('30s').size().plot(ax=ax5, color='blue', alpha=0.7)
        ax5.set_title('Update Frequency (30s windows)')
        ax5.set_ylabel('Updates per 30s')
        ax5.grid(True, alpha=0.3)
        
        # 6. Price change distribution
        price_changes = self.data['price_change'].dropna()
        if len(price_changes) > 0:
            price_changes.hist(bins=50, ax=ax6, alpha=0.7, edgecolor='black')
            ax6.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax6.set_title('Price Change Distribution')
            ax6.set_xlabel('Price Change ($)')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'market_analysis_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Saved visualization: {plot_filename}")
        
        plt.show()
        
        return plot_filename
    
    def export_summary_report(self):
        """Export comprehensive analysis report."""
        report_filename = f'market_report_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_filename, 'w') as f:
            f.write(f"ESP32 Kraken Market Data Analysis Report\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.csv_file}\n\n")
            
            # Basic stats
            stats = self.basic_statistics()
            f.write(f"Basic Statistics:\n")
            f.write(f"- Total Updates: {stats['total_updates']:,}\n")
            f.write(f"- Update Frequency: {stats['update_frequency']:.2f}/sec\n")
            f.write(f"- Price Range: ${stats['price_range']:.2f}\n")
            f.write(f"- Average Size: {stats['avg_size']:.2f}\n")
            f.write(f"- Total Volume: {stats['total_volume']:.2f}\n\n")
            
            # Order flow
            flow = self.analyze_order_flow()
            f.write(f"Order Flow Analysis:\n")
            f.write(f"- Order Additions: {flow['additions']:,}\n")
            f.write(f"- Order Removals: {flow['removals']:,}\n")
            f.write(f"- Bid/Ask Flow: {flow['bid_additions']:,} / {flow['ask_additions']:,}\n")
            f.write(f"- Large Orders: {flow['large_orders']:,}\n\n")
            
            # Price levels
            levels = self.analyze_price_levels()
            f.write(f"Price Level Analysis:\n")
            f.write(f"- Unique Price Levels: {levels['unique_prices']:,}\n")
            f.write(f"- Most Active Price: ${levels['most_active_price']:.2f}\n")
            f.write(f"- Most Active Count: {levels['most_active_count']:,}\n\n")
        
        print(f"📄 Saved report: {report_filename}")
        return report_filename
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("🚀 Starting comprehensive market data analysis...")
        
        # Run all analyses
        basic_stats = self.basic_statistics()
        order_flow = self.analyze_order_flow()
        price_levels = self.analyze_price_levels()
        
        # Create visualizations
        plot_file = self.create_visualizations()
        
        # Export report
        report_file = self.export_summary_report()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Visualization: {plot_file}")
        print(f"📄 Report: {report_file}")
        print(f"📈 Total data points analyzed: {len(self.data):,}")
        print("\n🎯 Use these insights for market making strategy development!")

def main():
    parser = argparse.ArgumentParser(description='Analyze ESP32 Kraken market data')
    parser.add_argument('csv_file', help='CSV file from ESP32 data collection')
    parser.add_argument('--quick', action='store_true', help='Quick analysis without visualizations')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File '{args.csv_file}' not found")
        sys.exit(1)
    
    # Create analyzer
    analyzer = MarketDataAnalyzer(args.csv_file)
    
    if args.quick:
        # Quick analysis
        analyzer.basic_statistics()
        analyzer.analyze_order_flow()
        analyzer.analyze_price_levels()
    else:
        # Full analysis with visualizations
        analyzer.run_full_analysis()

if __name__ == "__main__":
    main()