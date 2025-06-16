#!/usr/bin/env python3
"""
ESP32 Market Data Download Tool

Simple script to download market data from your ESP32 and run analysis.

Usage:
    python download_data.py 192.168.1.82  # Your ESP32's IP address
"""

import requests
import sys
import os
from datetime import datetime
import subprocess

def download_market_data(esp32_ip):
    """Download CSV data from ESP32."""
    try:
        print(f"📡 Connecting to ESP32 at {esp32_ip}...")
        
        # Check if ESP32 is reachable
        status_url = f"http://{esp32_ip}/"
        response = requests.get(status_url, timeout=10)
        
        if "Data collection complete" not in response.text:
            print("⏳ Data collection may still be in progress...")
            print(f"🌐 Check status at: http://{esp32_ip}/")
            return None
        
        # Download CSV file
        download_url = f"http://{esp32_ip}/download"
        print(f"📥 Downloading data from {download_url}...")
        
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kraken_market_data_{timestamp}.csv"
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content)
        print(f"✅ Downloaded {file_size:,} bytes to {filename}")
        
        return filename
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to ESP32 at {esp32_ip}")
        print("   Make sure the ESP32 is connected to WiFi and the IP is correct")
        return None
    except requests.exceptions.Timeout:
        print(f"⏰ Connection timeout to {esp32_ip}")
        return None
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def run_analysis(csv_file):
    """Run the analysis script on downloaded data."""
    try:
        print(f"🔍 Running analysis on {csv_file}...")
        
        # Check if analysis script exists
        if not os.path.exists("analyze_market_data.py"):
            print("❌ analyze_market_data.py not found in current directory")
            return False
        
        # Run analysis
        result = subprocess.run([sys.executable, "analyze_market_data.py", csv_file], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Analysis complete!")
            return True
        else:
            print("❌ Analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python download_data.py <ESP32_IP>")
        print("Example: python download_data.py 192.168.1.82")
        sys.exit(1)
    
    esp32_ip = sys.argv[1]
    
    print("🚀 ESP32 Market Data Download & Analysis Tool")
    print("="*50)
    
    # Download data
    csv_file = download_market_data(esp32_ip)
    
    if csv_file:
        print(f"\n📊 Data file: {csv_file}")
        
        # Ask if user wants to run analysis
        while True:
            choice = input("\n🔍 Run analysis now? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                run_analysis(csv_file)
                break
            elif choice in ['n', 'no']:
                print(f"📝 To analyze later, run: python analyze_market_data.py {csv_file}")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()