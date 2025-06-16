# ESP32 Kraken Futures Market Data Collector

A lightweight ESP32 application that connects to Kraken Futures WebSocket API to collect live orderbook data and save it to CSV format.

## Features

- Connects to WiFi network
- Subscribes to Kraken Futures orderbook data (SOL/USD)
- Collects data for 10 seconds
- Saves data to CSV file on SPIFFS
- Displays collected data via Serial Monitor

## Hardware Requirements

- ESP32 DOIT DevKit V1
- USB cable for programming and power
- WiFi network connection

## Setup Instructions

### 1. Configure WiFi Credentials

Edit `src/config.h` and update your WiFi credentials:

```cpp
#define WIFI_SSID "YourWiFiNetwork"
#define WIFI_PASSWORD "YourWiFiPassword"
```

### 2. Install PlatformIO

If you haven't already, install PlatformIO:
- Install the PlatformIO extension in VS Code, or
- Install PlatformIO CLI: `pip install platformio`

### 3. Build and Upload

```bash
# Build the project
pio run

# Upload to ESP32
pio run --target upload

# Monitor serial output
pio device monitor -b 115200
```

## Expected Output

The application will:

1. Connect to WiFi
2. Connect to Kraken Futures WebSocket
3. Subscribe to SOL/USD orderbook data
4. Collect data for 10 seconds
5. Display the collected CSV data in the Serial Monitor
6. Stop and show statistics

## CSV Format

The collected data is saved in CSV format with the following columns:

```
timestamp_ms,symbol,side,price,size
```

Example:
```
timestamp_ms,symbol,side,price,size
12345,PF_SOLUSD,buy,123.450000,10.500000
12346,PF_SOLUSD,sell,123.480000,5.250000
```

## Troubleshooting

### WiFi Connection Issues
- Ensure your WiFi network is 2.4GHz (ESP32 doesn't support 5GHz)
- Check your WiFi credentials in `config.h`
- Verify the ESP32 is within range of your WiFi router

### WebSocket Connection Issues
- Check your internet connection
- Verify the Kraken Futures API is accessible
- Check the Serial Monitor for error messages

### Memory Issues
- The application uses SPIFFS for file storage
- Monitor free heap memory in the Serial output
- Reduce buffer sizes in `config.h` if needed

## Customization

### Change Symbol
Edit `config.h` to monitor different symbols:
```cpp
#define SYMBOL "PF_ETHUSD"  // For Ethereum futures
```

### Adjust Collection Duration
Modify the duration in `config.h`:
```cpp
#define DATA_COLLECTION_DURATION_MS 30000  // 30 seconds
```

### Debug Mode
Enable/disable debug output:
```cpp
#define DEBUG_MODE 1  // 1 = enabled, 0 = disabled
```

## Next Steps

This basic implementation can be extended with:
- Multiple symbol monitoring
- Real-time data analysis
- Market making strategies
- Order placement functionality
- Historical data storage
- Web interface for monitoring

## License

This project is provided as-is for educational purposes. Use at your own risk when dealing with real market data or trading.