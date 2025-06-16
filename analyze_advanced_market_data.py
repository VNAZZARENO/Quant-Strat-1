#!/usr/bin/env python3
"""
ESP32 Kraken Market Data Analysis Tool - Enhanced for 0+ Strategy

Advanced market microstructure analysis including:
- Full orderbook reconstruction and L2 book dynamics
- Price impact modeling and liquidity depletion analysis
- Queue dynamics for 0+ strategy implementation
- Advanced statistical modeling and machine learning features
- Comprehensive visualizations for HFT strategy development

Usage:
    python analyze_advanced_market_data.py kraken_market_data.csv [--export-models]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import sys
from pathlib import Path
from collections import defaultdict, deque
from scipy import stats, optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OrderBookReconstructor:
    """Reconstructs full L2 orderbook from tick data."""
    
    def __init__(self):
        self.bids = defaultdict(float)  # price -> total_size
        self.asks = defaultdict(float)  # price -> total_size
        self.book_snapshots = []
        self.price_levels = []
        
    def process_update(self, timestamp, side, price, size):
        """Process individual orderbook update."""
        book = self.bids if side == 'buy' else self.asks
        
        if size == 0:
            # Remove price level
            if price in book:
                del book[price]
        else:
            # Add/update price level
            book[price] = size
            
        # Store snapshot
        self.book_snapshots.append({
            'timestamp': timestamp,
            'best_bid': max(self.bids.keys()) if self.bids else np.nan,
            'best_ask': min(self.asks.keys()) if self.asks else np.nan,
            'bid_size': self.bids.get(max(self.bids.keys()), 0) if self.bids else 0,
            'ask_size': self.asks.get(min(self.asks.keys()), 0) if self.asks else 0,
            'spread': (min(self.asks.keys()) - max(self.bids.keys())) if (self.bids and self.asks) else np.nan,
            'mid_price': (max(self.bids.keys()) + min(self.asks.keys())) / 2 if (self.bids and self.asks) else np.nan,
            'total_bid_levels': len(self.bids),
            'total_ask_levels': len(self.asks),
            'total_bid_volume': sum(self.bids.values()),
            'total_ask_volume': sum(self.asks.values())
        })
        
    def get_book_dataframe(self):
        """Convert book snapshots to DataFrame."""
        return pd.DataFrame(self.book_snapshots)
        
    def calculate_depth_metrics(self, depth_levels=5):
        """Calculate market depth metrics."""
        depth_data = []
        
        for snapshot in self.book_snapshots:
            if not pd.isna(snapshot['best_bid']) and not pd.isna(snapshot['best_ask']):
                # Calculate cumulative depth
                bid_depth = 0
                ask_depth = 0
                
                # Get top N levels
                sorted_bids = sorted([p for p in self.bids.keys() if p <= snapshot['best_bid']], reverse=True)[:depth_levels]
                sorted_asks = sorted([p for p in self.asks.keys() if p >= snapshot['best_ask']])[:depth_levels]
                
                for price in sorted_bids:
                    bid_depth += self.bids[price]
                    
                for price in sorted_asks:
                    ask_depth += self.asks[price]
                    
                depth_data.append({
                    'timestamp': snapshot['timestamp'],
                    'bid_depth': bid_depth,
                    'ask_depth': ask_depth,
                    'total_depth': bid_depth + ask_depth,
                    'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
                })
                
        return pd.DataFrame(depth_data)

class QueueDynamicsAnalyzer:
    """Analyzes queue dynamics for 0+ strategy insights."""
    
    def __init__(self):
        self.queue_events = []
        self.price_queues = defaultdict(list)  # price -> [(timestamp, event_type, size)]
        
    def process_event(self, timestamp, side, price, size, prev_size=0):
        """Process queue event for 0+ analysis."""
        event_type = 'add' if size > prev_size else 'reduce' if size < prev_size else 'cancel' if size == 0 else 'unknown'
        
        self.queue_events.append({
            'timestamp': timestamp,
            'side': side,
            'price': price,
            'size': size,
            'prev_size': prev_size,
            'event_type': event_type,
            'size_change': size - prev_size
        })
        
        self.price_queues[price].append((timestamp, event_type, size))
        
    def calculate_queue_metrics(self):
        """Calculate queue strength and dynamics metrics."""
        df = pd.DataFrame(self.queue_events)
        if df.empty:
            return pd.DataFrame()
            
        # Group by price level to analyze queue behavior
        queue_metrics = []
        
        for price in df['price'].unique():
            price_data = df[df['price'] == price].copy()
            price_data = price_data.sort_values('timestamp')
            
            # Calculate queue strength over time
            price_data['queue_strength'] = price_data['size'].fillna(0)
            price_data['queue_velocity'] = price_data['size_change'].rolling(window=5).mean()
            price_data['queue_stability'] = price_data['size'].rolling(window=10).std()
            
            # Add to metrics
            for _, row in price_data.iterrows():
                queue_metrics.append({
                    'timestamp': row['timestamp'],
                    'price': price,
                    'side': row['side'],
                    'queue_strength': row['queue_strength'],
                    'queue_velocity': row['queue_velocity'],
                    'queue_stability': row['queue_stability'],
                    'event_type': row['event_type']
                })
                
        return pd.DataFrame(queue_metrics)
        
    def identify_queue_opportunities(self, min_strength=100, max_position=3):
        """Identify 0+ strategy opportunities."""
        metrics_df = self.calculate_queue_metrics()
        if metrics_df.empty:
            return pd.DataFrame()
            
        # Find strong queues with low volatility
        opportunities = metrics_df[
            (metrics_df['queue_strength'] >= min_strength) &
            (metrics_df['queue_stability'] <= metrics_df['queue_stability'].quantile(0.3))
        ].copy()
        
        # Estimate queue position and scratching probability
        opportunities['estimated_position'] = np.random.randint(1, max_position + 1, len(opportunities))
        opportunities['scratch_probability'] = (opportunities['queue_strength'] - opportunities['estimated_position'] * 100) / opportunities['queue_strength']
        opportunities['scratch_probability'] = opportunities['scratch_probability'].clip(0, 1)
        
        # Calculate theoretical 0+ edge
        tick_size = 0.5  # Adjust based on instrument
        opportunities['theoretical_edge'] = (opportunities['scratch_probability'] * tick_size) - (2 * 0.1)  # Assuming 0.1 transaction cost
        
        return opportunities[opportunities['theoretical_edge'] > 0]

class AdvancedMarketAnalyzer:
    def __init__(self, csv_file, advanced_mode=True):
        """Initialize enhanced analyzer with CSV data from ESP32."""
        self.csv_file = csv_file
        self.data = None
        self.symbol = None
        self.advanced_mode = advanced_mode
        
        # Advanced analysis containers
        self.orderbook = OrderBookReconstructor()
        self.price_impact_model = None
        self.queue_dynamics = QueueDynamicsAnalyzer()
        self.ml_features = None
        
        self.load_data()

    def load_data(self):
        """Load and preprocess CSV data with advanced features."""
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
            
            if self.advanced_mode:
                print("🔬 Initializing advanced analysis components...")
                self._build_advanced_features()
                self._reconstruct_orderbook()
                self._calculate_ml_features()
            
            print(f"📊 Symbol: {self.symbol}")
            print(f"⏰ Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            print(f"💰 Price range: ${self.data['price'].min():.2f} - ${self.data['price'].max():.2f}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
            
    def _build_advanced_features(self):
        """Build advanced market microstructure features."""
        # Time-based features
        self.data['time_since_start'] = (self.data['timestamp'] - self.data['timestamp'].min()).dt.total_seconds()
        self.data['time_delta'] = self.data['timestamp'].diff().dt.total_seconds()
        
        # Price movement features
        self.data['price_returns'] = self.data['price'].pct_change()
        self.data['price_volatility'] = self.data['price_returns'].rolling(window=20).std()
        self.data['price_momentum'] = self.data['price'].rolling(window=10).mean() - self.data['price'].rolling(window=30).mean()
        
        # Volume features
        self.data['volume_ma'] = self.data['size'].rolling(window=20).mean()
        self.data['volume_std'] = self.data['size'].rolling(window=20).std()
        self.data['volume_ratio'] = self.data['size'] / self.data['volume_ma']
        
        # Market microstructure features
        self.data['is_aggressive'] = (self.data['size_change'] > 0) & (self.data['size'] > 0)
        self.data['is_cancellation'] = self.data['size'] == 0
        self.data['order_flow_imbalance'] = self._calculate_order_flow_imbalance()
        
    def _calculate_order_flow_imbalance(self, window=50):
        """Calculate order flow imbalance for market impact prediction."""
        buy_volume = self.data[self.data['side'] == 'buy']['size'].rolling(window=window).sum()
        sell_volume = self.data[self.data['side'] == 'sell']['size'].rolling(window=window).sum()
        
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / total_volume
        
        return imbalance.reindex(self.data.index).fillna(0)
        
    def _reconstruct_orderbook(self):
        """Reconstruct orderbook from tick data."""
        print("📚 Reconstructing orderbook...")
        
        # Process each update through orderbook reconstructor
        prev_sizes = {}  # (side, price) -> prev_size
        
        for _, row in self.data.iterrows():
            key = (row['side'], row['price'])
            prev_size = prev_sizes.get(key, 0)
            
            # Update orderbook
            self.orderbook.process_update(row['timestamp'], row['side'], row['price'], row['size'])
            
            # Update queue dynamics
            self.queue_dynamics.process_event(row['timestamp'], row['side'], row['price'], row['size'], prev_size)
            
            prev_sizes[key] = row['size']
            
        print(f"✅ Processed {len(self.orderbook.book_snapshots)} book snapshots")
        
    def _calculate_ml_features(self):
        """Calculate machine learning features for predictive modeling."""
        print("🤖 Calculating ML features...")
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            self.data[f'price_mean_{window}'] = self.data['price'].rolling(window=window).mean()
            self.data[f'price_std_{window}'] = self.data['price'].rolling(window=window).std()
            self.data[f'size_mean_{window}'] = self.data['size'].rolling(window=window).mean()
            self.data[f'size_std_{window}'] = self.data['size'].rolling(window=window).std()
            
        # Technical indicators
        self.data['rsi'] = self._calculate_rsi(self.data['price'], window=14)
        self.data['bollinger_upper'], self.data['bollinger_lower'] = self._calculate_bollinger_bands(self.data['price'])
        
        # Rolling highs and lows with variation analysis
        for window in [10, 20, 50, 100]:
            self.data[f'rolling_high_{window}'] = self.data['price'].rolling(window=window).max()
            self.data[f'rolling_low_{window}'] = self.data['price'].rolling(window=window).min()
            self.data[f'price_range_{window}'] = self.data[f'rolling_high_{window}'] - self.data[f'rolling_low_{window}']
            self.data[f'price_position_{window}'] = (self.data['price'] - self.data[f'rolling_low_{window}']) / (self.data[f'price_range_{window}'] + 1e-6)
            
            # Size variation analysis
            self.data[f'size_high_{window}'] = self.data['size'].rolling(window=window).max()
            self.data[f'size_low_{window}'] = self.data['size'].rolling(window=window).min()
            self.data[f'size_variation_{window}'] = (self.data[f'size_high_{window}'] - self.data[f'size_low_{window}']) / (self.data[f'size_high_{window}'] + 1e-6)
        
        # Microstructure features
        self.data['tick_rule'] = np.sign(self.data['price_change'])
        self.data['trade_intensity'] = 1 / (self.data['time_delta'] + 1e-6)  # Trades per second
        self.data['size_intensity'] = self.data['size'] * self.data['trade_intensity']
        
        self.ml_features = [col for col in self.data.columns if any(x in col for x in ['_mean_', '_std_', 'rsi', 'bollinger', 'tick_rule', 'intensity', 'imbalance', 'volatility', 'momentum', 'rolling_high_', 'rolling_low_', 'price_range_', 'price_position_', 'size_high_', 'size_low_', 'size_variation_'])]
        print(f"✅ Generated {len(self.ml_features)} ML features")
        
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower

    def analyze_orderbook_dynamics(self):
        """Analyze orderbook reconstruction and dynamics."""
        print("\n" + "="*60)
        print("📚 ORDERBOOK DYNAMICS ANALYSIS")
        print("="*60)
        
        book_df = self.orderbook.get_book_dataframe()
        depth_df = self.orderbook.calculate_depth_metrics()
        
        if book_df.empty:
            print("❌ No orderbook data available")
            return {}
            
        # Basic orderbook statistics
        print(f"📊 Orderbook Snapshots: {len(book_df):,}")
        print(f"💰 Average Spread: ${book_df['spread'].mean():.4f}")
        print(f"📈 Spread Volatility: ${book_df['spread'].std():.4f}")
        print(f"🎯 Mid Price Range: ${book_df['mid_price'].min():.2f} - ${book_df['mid_price'].max():.2f}")
        
        # Market depth analysis
        if not depth_df.empty:
            print(f"\n📊 Market Depth Analysis:")
            print(f"   Average Bid Depth: {depth_df['bid_depth'].mean():.1f}")
            print(f"   Average Ask Depth: {depth_df['ask_depth'].mean():.1f}")
            print(f"   Average Depth Imbalance: {depth_df['depth_imbalance'].mean():.3f}")
            print(f"   Depth Imbalance Std: {depth_df['depth_imbalance'].std():.3f}")
        
        # Liquidity metrics
        book_df['liquidity_score'] = book_df['total_bid_volume'] + book_df['total_ask_volume']
        book_df['book_imbalance'] = (book_df['total_bid_volume'] - book_df['total_ask_volume']) / book_df['liquidity_score']
        
        print(f"\n💧 Liquidity Metrics:")
        print(f"   Average Total Liquidity: {book_df['liquidity_score'].mean():.1f}")
        print(f"   Liquidity Volatility: {book_df['liquidity_score'].std():.1f}")
        print(f"   Average Book Imbalance: {book_df['book_imbalance'].mean():.3f}")
        
        return {
            'book_snapshots': len(book_df),
            'avg_spread': book_df['spread'].mean(),
            'spread_volatility': book_df['spread'].std(),
            'avg_liquidity': book_df['liquidity_score'].mean(),
            'book_imbalance': book_df['book_imbalance'].mean()
        }

    def analyze_queue_dynamics_for_0plus(self):
        """Analyze queue dynamics for 0+ strategy implementation."""
        print("\n" + "="*60)
        print("🎯 QUEUE DYNAMICS FOR 0+ STRATEGY")
        print("="*60)
        
        queue_metrics = self.queue_dynamics.calculate_queue_metrics()
        opportunities = self.queue_dynamics.identify_queue_opportunities()
        
        if queue_metrics.empty:
            print("❌ No queue metrics available")
            return {}
            
        # Queue strength analysis
        print(f"📊 Queue Events Analyzed: {len(queue_metrics):,}")
        print(f"🏃 Average Queue Strength: {queue_metrics['queue_strength'].mean():.1f}")
        print(f"⚡ Queue Velocity (avg): {queue_metrics['queue_velocity'].mean():.2f}")
        print(f"📊 Queue Stability (avg): {queue_metrics['queue_stability'].mean():.2f}")
        
        # Event type distribution
        event_counts = queue_metrics['event_type'].value_counts()
        print(f"\n📈 Queue Event Distribution:")
        for event, count in event_counts.items():
            pct = count / len(queue_metrics) * 100
            print(f"   {event.title()}: {count:,} ({pct:.1f}%)")
            
        # 0+ Opportunities
        if not opportunities.empty:
            print(f"\n🎯 0+ Strategy Opportunities Found: {len(opportunities):,}")
            print(f"💰 Average Theoretical Edge: ${opportunities['theoretical_edge'].mean():.4f}")
            print(f"🎲 Average Scratch Probability: {opportunities['scratch_probability'].mean():.3f}")
            print(f"📊 Best Theoretical Edge: ${opportunities['theoretical_edge'].max():.4f}")
            
            # Best opportunities by side
            bid_opps = opportunities[opportunities['side'] == 'buy']
            ask_opps = opportunities[opportunities['side'] == 'sell']
            
            print(f"\n📊 Opportunities by Side:")
            print(f"   Buy Side: {len(bid_opps):,} opportunities")
            print(f"   Sell Side: {len(ask_opps):,} opportunities")
            
        else:
            print("\n❌ No profitable 0+ opportunities identified")
            print("   Consider adjusting parameters (transaction costs, queue strength thresholds)")
            
        return {
            'total_queue_events': len(queue_metrics),
            'avg_queue_strength': queue_metrics['queue_strength'].mean(),
            'opportunities_found': len(opportunities),
            'avg_theoretical_edge': opportunities['theoretical_edge'].mean() if not opportunities.empty else 0
        }

    def build_price_impact_model(self):
        """Build ML model to predict price impact from order flow."""
        print("\n" + "="*60)
        print("🤖 PRICE IMPACT MODELING")
        print("="*60)
        
        if not self.ml_features:
            print("❌ ML features not available")
            return None
            
        # Prepare features for modeling
        feature_cols = [col for col in self.ml_features if col in self.data.columns]
        feature_data = self.data[feature_cols].dropna()
        
        if len(feature_data) < 100:
            print("❌ Insufficient data for modeling")
            return None
            
        # Create target variable (future price change)
        self.data['future_price_change'] = self.data['price'].shift(-5) - self.data['price']  # 5-step ahead prediction
        
        # Prepare training data
        valid_indices = feature_data.index
        X = feature_data.loc[valid_indices]
        y = self.data.loc[valid_indices, 'future_price_change'].dropna()
        
        # Align X and y
        common_indices = X.index.intersection(y.index)
        X = X.loc[common_indices]
        y = y.loc[common_indices]
        
        if len(X) < 50:
            print("❌ Insufficient aligned data for modeling")
            return None
            
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"🎯 Model Performance:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MSE: {mse:.6f}")
        print(f"   RMSE: {np.sqrt(mse):.6f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
        self.price_impact_model = {
            'model': model,
            'scaler': scaler,
            'features': feature_cols,
            'performance': {'r2': r2, 'mse': mse, 'rmse': np.sqrt(mse)},
            'feature_importance': feature_importance
        }
        
        return self.price_impact_model

    def analyze_rolling_patterns(self):
        """Analyze rolling high/low patterns and variations."""
        print("\n" + "="*60)
        print("📈 ROLLING HIGH/LOW PATTERN ANALYSIS")
        print("="*60)
        
        # Check if rolling features exist
        rolling_features = [col for col in self.data.columns if 'rolling_high_' in col or 'rolling_low_' in col]
        if not rolling_features:
            print("❌ No rolling high/low features available")
            return {}
            
        results = {}
        
        # Analyze different time windows
        for window in [10, 20, 50, 100]:
            if f'price_range_{window}' in self.data.columns:
                price_range = self.data[f'price_range_{window}'].dropna()
                price_position = self.data[f'price_position_{window}'].dropna()
                
                print(f"\n📊 {window}-Period Analysis:")
                print(f"   Average Range: ${price_range.mean():.2f}")
                print(f"   Range Volatility: ${price_range.std():.2f}")
                print(f"   Max Range: ${price_range.max():.2f}")
                print(f"   Min Range: ${price_range.min():.2f}")
                
                # Price position analysis
                high_position = (price_position > 0.8).sum()
                low_position = (price_position < 0.2).sum()
                mid_position = ((price_position >= 0.4) & (price_position <= 0.6)).sum()
                
                print(f"   High Position (>80%): {high_position:,} ({high_position/len(price_position)*100:.1f}%)")
                print(f"   Low Position (<20%): {low_position:,} ({low_position/len(price_position)*100:.1f}%)")
                print(f"   Mid Position (40-60%): {mid_position:,} ({mid_position/len(price_position)*100:.1f}%)")
                
                results[f'range_{window}'] = {
                    'avg_range': price_range.mean(),
                    'range_volatility': price_range.std(),
                    'high_position_pct': high_position/len(price_position)*100,
                    'low_position_pct': low_position/len(price_position)*100
                }
        
        # Size variation analysis
        print(f"\n📦 SIZE VARIATION ANALYSIS:")
        for window in [10, 20, 50]:
            if f'size_variation_{window}' in self.data.columns:
                size_var = self.data[f'size_variation_{window}'].dropna()
                print(f"   {window}-period Size Variation: {size_var.mean():.3f} (±{size_var.std():.3f})")
                
                results[f'size_var_{window}'] = {
                    'mean': size_var.mean(),
                    'std': size_var.std()
                }
        
        # Pattern detection for 0+ strategy
        if 'price_position_50' in self.data.columns and 'size_variation_20' in self.data.columns:
            print(f"\n🎯 0+ STRATEGY PATTERN INSIGHTS:")
            
            # Find periods of low volatility + extreme positions
            stable_periods = self.data[
                (self.data['size_variation_20'] < self.data['size_variation_20'].quantile(0.3)) &
                ((self.data['price_position_50'] > 0.8) | (self.data['price_position_50'] < 0.2))
            ]
            
            print(f"   Stable + Extreme Position Periods: {len(stable_periods):,}")
            if len(stable_periods) > 0:
                print(f"   These represent potential mean reversion opportunities")
                print(f"   Average queue strength during these periods: {stable_periods['size'].mean():.1f}")
                
                results['mean_reversion_opportunities'] = len(stable_periods)
        
        return results

    def create_advanced_visualizations(self):
        """Create comprehensive advanced visualizations."""
        print("\n" + "="*60)
        print("📊 CREATING ADVANCED VISUALIZATIONS")
        print("="*60)
        
        # Create large subplot grid
        fig = plt.figure(figsize=(24, 28))
        gs = fig.add_gridspec(7, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Advanced Market Microstructure Analysis - {self.symbol}', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Orderbook reconstruction
        ax1 = fig.add_subplot(gs[0, 0])
        book_df = self.orderbook.get_book_dataframe()
        if not book_df.empty:
            ax1.plot(book_df.index, book_df['spread'], alpha=0.7, color='red', label='Spread')
            ax1.set_title('Bid-Ask Spread Over Time')
            ax1.set_ylabel('Spread ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Market depth visualization
        ax2 = fig.add_subplot(gs[0, 1])
        depth_df = self.orderbook.calculate_depth_metrics()
        if not depth_df.empty:
            ax2.plot(depth_df.index, depth_df['bid_depth'], alpha=0.7, color='green', label='Bid Depth')
            ax2.plot(depth_df.index, depth_df['ask_depth'], alpha=0.7, color='red', label='Ask Depth')
            ax2.set_title('Market Depth Over Time')
            ax2.set_ylabel('Depth (Size)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Order flow imbalance
        ax3 = fig.add_subplot(gs[0, 2])
        if 'order_flow_imbalance' in self.data.columns:
            ax3.plot(self.data.index, self.data['order_flow_imbalance'], alpha=0.7, color='purple')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Order Flow Imbalance')
            ax3.set_ylabel('Imbalance')
            ax3.grid(True, alpha=0.3)
        
        # 4. Price impact scatter
        ax4 = fig.add_subplot(gs[1, 0])
        if 'future_price_change' in self.data.columns and 'order_flow_imbalance' in self.data.columns:
            scatter_data = self.data[['order_flow_imbalance', 'future_price_change']].dropna()
            if len(scatter_data) > 0:
                ax4.scatter(scatter_data['order_flow_imbalance'], scatter_data['future_price_change'], 
                           alpha=0.5, s=1)
                ax4.set_xlabel('Order Flow Imbalance')
                ax4.set_ylabel('Future Price Change')
                ax4.set_title('Price Impact vs Order Flow')
                ax4.grid(True, alpha=0.3)
        
        # 5. Queue dynamics heatmap
        ax5 = fig.add_subplot(gs[1, 1])
        queue_metrics = self.queue_dynamics.calculate_queue_metrics()
        if not queue_metrics.empty and len(queue_metrics) > 50:
            try:
                # Create simplified heatmap using quantile-based binning
                queue_metrics_sample = queue_metrics.sample(min(1000, len(queue_metrics)))  # Sample for performance
                
                # Create price and time bins manually
                price_bins = pd.qcut(queue_metrics_sample['price'], q=10, duplicates='drop')
                time_bins = pd.qcut(range(len(queue_metrics_sample)), q=min(15, len(queue_metrics_sample)//10), duplicates='drop')
                
                queue_metrics_sample['price_bin'] = price_bins
                queue_metrics_sample['time_bin'] = time_bins
                
                pivot_data = queue_metrics_sample.pivot_table(
                    values='queue_strength', 
                    index='price_bin',
                    columns='time_bin',
                    aggfunc='mean'
                )
                
                if not pivot_data.empty:
                    sns.heatmap(pivot_data, ax=ax5, cmap='YlOrRd', cbar_kws={'label': 'Queue Strength'})
                    ax5.set_title('Queue Strength Heatmap')
                    ax5.set_xlabel('Time Period')
                    ax5.set_ylabel('Price Level')
                    ax5.tick_params(axis='both', which='major', labelsize=6)
                else:
                    ax5.text(0.5, 0.5, 'Insufficient data\nfor heatmap', ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title('Queue Strength Heatmap')
            except Exception as e:
                ax5.text(0.5, 0.5, f'Heatmap generation\nfailed', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Queue Strength Heatmap')
        else:
            ax5.text(0.5, 0.5, 'Insufficient queue data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Queue Strength Heatmap')
        
        # 6. Volatility analysis
        ax6 = fig.add_subplot(gs[1, 2])
        if 'price_volatility' in self.data.columns:
            ax6.plot(self.data.index, self.data['price_volatility'], alpha=0.7, color='orange')
            ax6.set_title('Price Volatility Over Time')
            ax6.set_ylabel('Volatility')
            ax6.grid(True, alpha=0.3)
        
        # 7. Volume intensity analysis
        ax7 = fig.add_subplot(gs[2, 0])
        if 'size_intensity' in self.data.columns:
            ax7.plot(self.data.index, self.data['size_intensity'], alpha=0.7, color='blue')
            ax7.set_title('Size Intensity (Size × Frequency)')
            ax7.set_ylabel('Intensity')
            ax7.grid(True, alpha=0.3)
        
        # 8. RSI indicator
        ax8 = fig.add_subplot(gs[2, 1])
        if 'rsi' in self.data.columns:
            ax8.plot(self.data.index, self.data['rsi'], alpha=0.7, color='purple')
            ax8.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            ax8.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            ax8.set_title('RSI Indicator')
            ax8.set_ylabel('RSI')
            ax8.set_ylim(0, 100)
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Bollinger Bands
        ax9 = fig.add_subplot(gs[2, 2])
        if all(col in self.data.columns for col in ['bollinger_upper', 'bollinger_lower']):
            ax9.plot(self.data.index, self.data['price'], alpha=0.7, color='black', label='Price')
            ax9.plot(self.data.index, self.data['bollinger_upper'], alpha=0.5, color='red', label='Upper Band')
            ax9.plot(self.data.index, self.data['bollinger_lower'], alpha=0.5, color='green', label='Lower Band')
            ax9.fill_between(self.data.index, self.data['bollinger_upper'], self.data['bollinger_lower'], 
                            alpha=0.1, color='gray')
            ax9.set_title('Bollinger Bands')
            ax9.set_ylabel('Price ($)')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        # 10. 0+ Opportunities visualization
        ax10 = fig.add_subplot(gs[3, 0])
        opportunities = self.queue_dynamics.identify_queue_opportunities()
        if not opportunities.empty:
            ax10.scatter(opportunities['queue_strength'], opportunities['theoretical_edge'], 
                        c=opportunities['scratch_probability'], cmap='viridis', alpha=0.7)
            ax10.set_xlabel('Queue Strength')
            ax10.set_ylabel('Theoretical Edge ($)')
            ax10.set_title('0+ Strategy Opportunities')
            cbar = plt.colorbar(ax10.collections[0], ax=ax10)
            cbar.set_label('Scratch Probability')
            ax10.grid(True, alpha=0.3)
        
        # 11. Price level activity
        ax11 = fig.add_subplot(gs[3, 1])
        price_activity = self.data.groupby('price').size().sort_values(ascending=False).head(15)
        price_activity.plot(kind='bar', ax=ax11, color='skyblue', alpha=0.7)
        ax11.set_title('Most Active Price Levels')
        ax11.set_xlabel('Price Level')
        ax11.set_ylabel('Activity Count')
        ax11.tick_params(axis='x', rotation=45)
        ax11.grid(True, alpha=0.3)
        
        # 12. Event type distribution
        ax12 = fig.add_subplot(gs[3, 2])
        if not queue_metrics.empty:
            event_counts = queue_metrics['event_type'].value_counts()
            ax12.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', startangle=90)
            ax12.set_title('Queue Event Distribution')
        
        # 13. Rolling highs and lows analysis
        ax13 = fig.add_subplot(gs[4, 0])
        if 'rolling_high_50' in self.data.columns and 'rolling_low_50' in self.data.columns:
            ax13.plot(self.data.index, self.data['price'], alpha=0.7, color='black', label='Price', linewidth=1)
            ax13.plot(self.data.index, self.data['rolling_high_50'], alpha=0.6, color='green', label='50-period High')
            ax13.plot(self.data.index, self.data['rolling_low_50'], alpha=0.6, color='red', label='50-period Low')
            ax13.fill_between(self.data.index, self.data['rolling_high_50'], self.data['rolling_low_50'], 
                            alpha=0.1, color='gray')
            ax13.set_title('Rolling High/Low Analysis (50-period)')
            ax13.set_ylabel('Price ($)')
            ax13.legend()
            ax13.grid(True, alpha=0.3)
        
        # 14. Price position in range
        ax14 = fig.add_subplot(gs[4, 1])
        if 'price_position_50' in self.data.columns:
            ax14.plot(self.data.index, self.data['price_position_50'], alpha=0.7, color='purple')
            ax14.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Midpoint')
            ax14.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High (80%)')
            ax14.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Low (20%)')
            ax14.set_title('Price Position in 50-period Range')
            ax14.set_ylabel('Position (0=Low, 1=High)')
            ax14.set_ylim(0, 1)
            ax14.legend()
            ax14.grid(True, alpha=0.3)
        
        # 15. Size variation analysis
        ax15 = fig.add_subplot(gs[4, 2])
        if 'size_variation_20' in self.data.columns:
            ax15.plot(self.data.index, self.data['size_variation_20'], alpha=0.7, color='orange')
            ax15.set_title('Size Variation (20-period)')
            ax15.set_ylabel('Variation Ratio')
            ax15.grid(True, alpha=0.3)
        
        # 16. Feature importance (if model exists)
        if self.price_impact_model:
            ax16 = fig.add_subplot(gs[5, :])
            importance_df = self.price_impact_model['feature_importance'].head(15)
            bars = ax16.barh(range(len(importance_df)), importance_df['importance'].values, color='lightcoral', alpha=0.7)
            ax16.set_yticks(range(len(importance_df)))
            ax16.set_yticklabels(importance_df['feature'].values)
            ax16.set_xlabel('Feature Importance')
            ax16.set_title('ML Model Feature Importance (Top 15)')
            ax16.grid(True, alpha=0.3)
        
        # 17-19. Model performance visualization
        if self.price_impact_model and 'future_price_change' in self.data.columns:
            # Prediction vs actual scatter
            ax17 = fig.add_subplot(gs[6, 0])
            feature_cols = self.price_impact_model['features']
            feature_data = self.data[feature_cols].dropna()
            
            if len(feature_data) > 0:
                valid_indices = feature_data.index
                X = feature_data.loc[valid_indices]
                y_actual = self.data.loc[valid_indices, 'future_price_change'].dropna()
                
                common_indices = X.index.intersection(y_actual.index)
                if len(common_indices) > 10:
                    X_plot = X.loc[common_indices]
                    y_plot = y_actual.loc[common_indices]
                    
                    X_scaled = self.price_impact_model['scaler'].transform(X_plot)
                    y_pred = self.price_impact_model['model'].predict(X_scaled)
                    
                    ax17.scatter(y_plot, y_pred, alpha=0.5, s=1)
                    ax17.plot([y_plot.min(), y_plot.max()], [y_plot.min(), y_plot.max()], 'r--', alpha=0.8)
                    ax17.set_xlabel('Actual Price Change')
                    ax17.set_ylabel('Predicted Price Change')
                    ax17.set_title('Model Prediction vs Actual')
                    ax17.grid(True, alpha=0.3)
            
            # Residuals plot
            ax18 = fig.add_subplot(gs[6, 1])
            if 'y_pred' in locals() and 'y_plot' in locals():
                residuals = y_plot - y_pred
                ax18.scatter(y_pred, residuals, alpha=0.5, s=1)
                ax18.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                ax18.set_xlabel('Predicted Values')
                ax18.set_ylabel('Residuals')
                ax18.set_title('Residual Plot')
                ax18.grid(True, alpha=0.3)
            
            # Model performance metrics
            ax19 = fig.add_subplot(gs[6, 2])
            performance = self.price_impact_model['performance']
            metrics = ['R²', 'MSE', 'RMSE']
            values = [performance['r2'], performance['mse'], performance['rmse']]
            
            bars = ax19.bar(metrics, values, color=['green', 'orange', 'red'], alpha=0.7)
            ax19.set_title('Model Performance Metrics')
            ax19.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax19.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                         f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'advanced_market_analysis_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Saved advanced visualization: {plot_filename}")
        
        plt.show()
        
        return plot_filename

    def export_comprehensive_report(self):
        """Export comprehensive analysis report with all insights."""
        report_filename = f'comprehensive_market_report_{self.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"COMPREHENSIVE MARKET MICROSTRUCTURE ANALYSIS REPORT\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.csv_file}\n")
            f.write(f"Total Data Points: {len(self.data):,}\n\n")
            
            # Basic statistics
            f.write(f"BASIC MARKET STATISTICS\n")
            f.write(f"-" * 30 + "\n")
            bids = self.data[self.data['side'] == 'buy']
            asks = self.data[self.data['side'] == 'sell']
            f.write(f"Bid Updates: {len(bids):,}\n")
            f.write(f"Ask Updates: {len(asks):,}\n")
            f.write(f"Price Range: ${self.data['price'].min():.2f} - ${self.data['price'].max():.2f}\n")
            f.write(f"Average Size: {self.data['size'].mean():.2f}\n")
            f.write(f"Total Volume: {self.data['size'].sum():.2f}\n\n")
            
            # Orderbook analysis
            book_metrics = self.analyze_orderbook_dynamics()
            f.write(f"ORDERBOOK DYNAMICS\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Book Snapshots: {book_metrics.get('book_snapshots', 0):,}\n")
            f.write(f"Average Spread: ${book_metrics.get('avg_spread', 0):.4f}\n")
            f.write(f"Spread Volatility: ${book_metrics.get('spread_volatility', 0):.4f}\n")
            f.write(f"Average Liquidity: {book_metrics.get('avg_liquidity', 0):.1f}\n")
            f.write(f"Book Imbalance: {book_metrics.get('book_imbalance', 0):.3f}\n\n")
            
            # Queue dynamics
            queue_metrics = self.analyze_queue_dynamics_for_0plus()
            f.write(f"QUEUE DYNAMICS FOR 0+ STRATEGY\n")
            f.write(f"-" * 35 + "\n")
            f.write(f"Total Queue Events: {queue_metrics.get('total_queue_events', 0):,}\n")
            f.write(f"Average Queue Strength: {queue_metrics.get('avg_queue_strength', 0):.1f}\n")
            f.write(f"Opportunities Found: {queue_metrics.get('opportunities_found', 0):,}\n")
            f.write(f"Average Theoretical Edge: ${queue_metrics.get('avg_theoretical_edge', 0):.4f}\n\n")
            
            # Rolling pattern analysis
            rolling_patterns = self.analyze_rolling_patterns()
            f.write(f"ROLLING HIGH/LOW PATTERN ANALYSIS\n")
            f.write(f"-" * 35 + "\n")
            for window in [20, 50, 100]:
                if f'range_{window}' in rolling_patterns:
                    pattern = rolling_patterns[f'range_{window}']
                    f.write(f"{window}-Period Analysis:\n")
                    f.write(f"  Average Range: ${pattern['avg_range']:.2f}\n")
                    f.write(f"  Range Volatility: ${pattern['range_volatility']:.2f}\n")
                    f.write(f"  High Position %: {pattern['high_position_pct']:.1f}%\n")
                    f.write(f"  Low Position %: {pattern['low_position_pct']:.1f}%\n\n")
            
            if 'mean_reversion_opportunities' in rolling_patterns:
                f.write(f"Mean Reversion Opportunities: {rolling_patterns['mean_reversion_opportunities']:,}\n\n")
            
            # ML model performance
            if self.price_impact_model:
                f.write(f"PRICE IMPACT MODEL PERFORMANCE\n")
                f.write(f"-" * 35 + "\n")
                perf = self.price_impact_model['performance']
                f.write(f"R² Score: {perf['r2']:.4f}\n")
                f.write(f"RMSE: {perf['rmse']:.6f}\n")
                f.write(f"Features Used: {len(self.price_impact_model['features'])}\n\n")
                
                # Top features
                f.write(f"TOP 10 PREDICTIVE FEATURES:\n")
                importance_df = self.price_impact_model['feature_importance']
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    f.write(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}\n")
                f.write("\n")
            
            # Trading recommendations
            f.write(f"0+ STRATEGY IMPLEMENTATION RECOMMENDATIONS\n")
            f.write(f"-" * 45 + "\n")
            
            opportunities = self.queue_dynamics.identify_queue_opportunities()
            if not opportunities.empty:
                avg_edge = opportunities['theoretical_edge'].mean()
                max_edge = opportunities['theoretical_edge'].max()
                avg_scratch_prob = opportunities['scratch_probability'].mean()
                
                f.write(f"Strategy appears viable with {len(opportunities):,} opportunities\n")
                f.write(f"Expected average edge: ${avg_edge:.4f} per trade\n")
                f.write(f"Maximum theoretical edge: ${max_edge:.4f}\n")
                f.write(f"Average scratch probability: {avg_scratch_prob:.3f}\n")
                f.write(f"\nRECOMMENDATIONS:\n")
                f.write(f"- Focus on queues with strength > 100\n")
                f.write(f"- Target scratch probability > 0.5\n")
                f.write(f"- Monitor order flow imbalance for timing\n")
                f.write(f"- Implement latency optimization (target <50μs)\n")
                
                # Add rolling pattern insights
                if 'mean_reversion_opportunities' in rolling_patterns and rolling_patterns['mean_reversion_opportunities'] > 0:
                    f.write(f"- Consider mean reversion signals during extreme price positions\n")
                    f.write(f"- {rolling_patterns['mean_reversion_opportunities']:,} mean reversion setups identified\n")
                
            else:
                f.write(f"No profitable opportunities identified\n")
                f.write(f"RECOMMENDATIONS:\n")
                f.write(f"- Reduce transaction costs\n")
                f.write(f"- Consider different symbols/markets\n")
                f.write(f"- Adjust queue position thresholds\n")
                f.write(f"- Implement more sophisticated scratching logic\n")
        
        print(f"📄 Saved comprehensive report: {report_filename}")
        return report_filename

    def run_full_advanced_analysis(self):
        """Run complete advanced analysis pipeline."""
        print("🚀 Starting comprehensive advanced market analysis...")
        
        # Run all analyses
        orderbook_metrics = self.analyze_orderbook_dynamics()
        queue_metrics = self.analyze_queue_dynamics_for_0plus()
        rolling_patterns = self.analyze_rolling_patterns()
        price_model = self.build_price_impact_model()
        
        # Create visualizations
        plot_file = self.create_advanced_visualizations()
        
        # Export comprehensive report
        report_file = self.export_comprehensive_report()
        
        print("\n" + "="*60)
        print("✅ ADVANCED ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Advanced Visualization: {plot_file}")
        print(f"📄 Comprehensive Report: {report_file}")
        print(f"📈 Total data points analyzed: {len(self.data):,}")
        print(f"🔬 Advanced features generated: {len(self.ml_features) if self.ml_features else 0}")
        
        if price_model:
            print(f"🤖 ML Model R² Score: {price_model['performance']['r2']:.4f}")
            
        opportunities = self.queue_dynamics.identify_queue_opportunities()
        if not opportunities.empty:
            print(f"🎯 0+ Opportunities: {len(opportunities):,} (avg edge: ${opportunities['theoretical_edge'].mean():.4f})")
        else:
            print(f"⚠️  No 0+ opportunities found - consider parameter tuning")
            
        print("\n🎯 Ready for advanced HFT strategy implementation!")

def main():
    parser = argparse.ArgumentParser(description='Advanced ESP32 Kraken market data analysis')
    parser.add_argument('csv_file', help='CSV file from ESP32 data collection')
    parser.add_argument('--basic', action='store_true', help='Run basic analysis only (faster)')
    parser.add_argument('--export-models', action='store_true', help='Export trained models')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"❌ Error: File '{args.csv_file}' not found")
        sys.exit(1)
    
    # Create analyzer
    analyzer = AdvancedMarketAnalyzer(args.csv_file, advanced_mode=not args.basic)
    
    if args.basic:
        # Basic analysis
        analyzer.analyze_orderbook_dynamics()
        analyzer.analyze_queue_dynamics_for_0plus()
    else:
        # Full advanced analysis
        analyzer.run_full_advanced_analysis()
        
        # Export models if requested
        if args.export_models and analyzer.price_impact_model:
            import joblib
            model_filename = f'price_impact_model_{analyzer.symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(analyzer.price_impact_model, model_filename)
            print(f"🤖 Exported ML model: {model_filename}")

if __name__ == "__main__":
    main()