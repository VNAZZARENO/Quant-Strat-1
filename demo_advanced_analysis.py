#!/usr/bin/env python3
"""
Demo script for Advanced Market Data Analysis

This script demonstrates how to use the enhanced analysis tools for 0+ strategy development.
It shows the complete workflow from data loading to strategy insights.

Usage:
    python demo_advanced_analysis.py [csv_file]
"""

import sys
from pathlib import Path
import glob

def find_latest_csv():
    """Find the latest CSV file in the current directory."""
    csv_files = glob.glob("kraken_market_data_*.csv")
    if not csv_files:
        return None
    return max(csv_files, key=lambda x: Path(x).stat().st_mtime)

def main():
    print("🚀 ESP32 Market Data Advanced Analysis Demo")
    print("=" * 50)
    
    # Find CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = find_latest_csv()
        
    if not csv_file or not Path(csv_file).exists():
        print("❌ No CSV file found. Please:")
        print("1. Run the ESP32 data collection first")
        print("2. Download the data using: python download_data.py <ESP32_IP>")
        print("3. Or specify a CSV file: python demo_advanced_analysis.py <file.csv>")
        sys.exit(1)
    
    print(f"📊 Using data file: {csv_file}")
    print(f"📦 File size: {Path(csv_file).stat().st_size:,} bytes")
    
    # Import and run analysis
    try:
        from analyze_advanced_market_data import AdvancedMarketAnalyzer
        
        print("\n🔬 Initializing Advanced Market Analyzer...")
        analyzer = AdvancedMarketAnalyzer(csv_file, advanced_mode=True)
        
        print("\n📈 Running comprehensive analysis...")
        analyzer.run_full_advanced_analysis()
        
        print("\n" + "=" * 50)
        print("✅ DEMO COMPLETE!")
        print("=" * 50)
        print("\n📋 Analysis Results:")
        print("- Check the generated PNG file for visualizations")
        print("- Review the comprehensive report TXT file")
        print("- Use insights for 0+ strategy implementation")
        
        # Show key insights
        opportunities = analyzer.queue_dynamics.identify_queue_opportunities()
        if not opportunities.empty:
            print(f"\n🎯 Key 0+ Strategy Insights:")
            print(f"   • {len(opportunities):,} profitable opportunities identified")
            print(f"   • Average theoretical edge: ${opportunities['theoretical_edge'].mean():.4f}")
            print(f"   • Best opportunity edge: ${opportunities['theoretical_edge'].max():.4f}")
            print(f"   • Average scratch probability: {opportunities['scratch_probability'].mean():.3f}")
            
            # Implementation recommendations
            print(f"\n💡 Implementation Recommendations:")
            if opportunities['theoretical_edge'].mean() > 0.001:
                print("   ✅ Strategy appears viable for implementation")
                print("   ✅ Focus on high-scratch-probability opportunities")
                print("   ✅ Implement sub-50μs latency for competitive advantage")
            else:
                print("   ⚠️  Marginal profitability - optimize transaction costs")
                print("   ⚠️  Consider alternative symbols or market conditions")
            
        else:
            print(f"\n⚠️  No profitable 0+ opportunities found")
            print("   Consider adjusting strategy parameters or market conditions")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()