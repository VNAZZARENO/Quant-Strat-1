# ESP32 Kraken Quantitative Trading System

## 📋 Project Overview

This project implements a lightweight quantitative trading system on ESP32 DOIT DevKit V1, utilizing Kraken exchange order book data for high-frequency trading strategies and cross-asset correlation analysis.

### Features
- Real-time order book processing via WebSocket
- Cross-asset contagion detection
- Memory-efficient microstructure strategies
- Dual-core task distribution
- Automatic reconnection and error handling

## 🗂️ Project Structure

```
Quant Strat 1/
├── .gitignore
├── platformio.ini
├── README.md
├── src/
│   ├── main.cpp
│   ├── config.h
│   ├── KrakenWebSocket.h
│   ├── KrakenWebSocket.cpp
│   ├── OrderBookStrategy.h
│   ├── OrderBookStrategy.cpp
│   ├── CrossAssetMonitor.h
│   ├── CrossAssetMonitor.cpp
│   ├── Utils.h
│   └── Utils.cpp
├── lib/
│   └── README
├── include/
│   └── README
├── test/
│   ├── test_orderbook.cpp
│   └── test_strategy.cpp
└── data/
    └── .gitkeep
```

## 📝 Configuration Files

### `platformio.ini`
```ini
[env:esp32doit-devkit-v1]
platform = espressif32
board = esp32doit-devkit-v1
framework = arduino

; Serial Monitor
monitor_speed = 115200
monitor_filters = esp32_exception_decoder

; Build flags
build_flags = 
    -D CORE_DEBUG_LEVEL=2
    -DBOARD_HAS_PSRAM
    -mfix-esp32-psram-cache-issue

; Libraries
lib_deps = 
    bblanchon/ArduinoJson@^6.21.3
    links2004/WebSockets@^2.4.1
    me-no-dev/AsyncTCP@^1.1.1
    khoih-prog/ESP_DoubleResetDetector@^1.3.2

; Partition scheme for more program space
board_build.partitions = huge_app.csv

; Upload settings
upload_speed = 921600
upload_port = COM3  ; Change to your port
```

### `.gitignore`
```gitignore
.pio
.vscode/.browse.c_cpp.db*
.vscode/c_cpp_properties.json
.vscode/launch.json
.vscode/ipch
data/credentials.h
*.log
*.tmp
```

## 💻 Source Code

### `src/config.h`
```cpp
#ifndef CONFIG_H
#define CONFIG_H

// WiFi Configuration
#define WIFI_SSID "YourWiFiSSID"
#define WIFI_PASSWORD "YourWiFiPassword"

// Kraken API Configuration
#define KRAKEN_API_KEY "your_api_key_here"
#define KRAKEN_PRIVATE_KEY "your_private_key_here"

// WebSocket Configuration
#define KRAKEN_WS_HOST "ws.kraken.com"
#define KRAKEN_WS_PORT 443
#define KRAKEN_WS_PATH "/"
#define KRAKEN_FUTURES_WS_HOST "futures.kraken.com"
#define KRAKEN_FUTURES_WS_PATH "/ws/v1"

// Trading Pairs
const char* TRADING_PAIRS[] = {"XBT/USD", "ETH/USD", "XBT/EUR"};
const int NUM_PAIRS = 3;

// Strategy Parameters
#define ORDER_BOOK_DEPTH 10
#define IMBALANCE_THRESHOLD 0.3f
#define SPREAD_THRESHOLD_BPS 2.0f
#define CORRELATION_WINDOW 100
#define CONTAGION_THRESHOLD 0.85f

// System Configuration
#define TASK_STACK_SIZE 8192
#define WEBSOCKET_BUFFER_SIZE 4096
#define JSON_BUFFER_SIZE 8192

// Debug Configuration
#define DEBUG_MODE 1
#define PERFORMANCE_MONITORING 1

#endif
```

### `src/main.cpp`
```cpp
#include <Arduino.h>
#include <WiFi.h>
#include <esp_task_wdt.h>
#include "config.h"
#include "KrakenWebSocket.h"
#include "OrderBookStrategy.h"
#include "CrossAssetMonitor.h"
#include "Utils.h"

// Global objects
KrakenWebSocket krakenWS;
OrderBookStrategy strategy;
CrossAssetMonitor crossAssetMonitor;

// Task handles
TaskHandle_t dataTaskHandle = NULL;
TaskHandle_t strategyTaskHandle = NULL;

// Performance metrics
struct PerformanceMetrics {
    uint32_t messagesProcessed = 0;
    uint32_t signalsGenerated = 0;
    uint32_t errors = 0;
    uint32_t lastHeapSize = 0;
} metrics;

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("\n=== ESP32 Kraken Quant Trading System ===");
    Serial.printf("CPU Frequency: %d MHz\n", getCpuFrequencyMhz());
    Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
    
    // Initialize watchdog (30 seconds timeout)
    esp_task_wdt_init(30, true);
    esp_task_wdt_add(NULL);
    
    // Connect to WiFi
    connectToWiFi();
    
    // Initialize components
    krakenWS.init();
    strategy.init();
    crossAssetMonitor.init();
    
    // Create tasks on different cores
    xTaskCreatePinnedToCore(
        dataCollectionTask,
        "DataCollection",
        TASK_STACK_SIZE,
        NULL,
        2,  // Priority
        &dataTaskHandle,
        0   // Core 0
    );
    
    xTaskCreatePinnedToCore(
        strategyExecutionTask,
        "Strategy",
        TASK_STACK_SIZE,
        NULL,
        1,  // Priority
        &strategyTaskHandle,
        1   // Core 1
    );
    
    Serial.println("Setup complete! Starting trading system...\n");
}

void connectToWiFi() {
    Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected!");
        Serial.printf("IP address: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("Signal strength: %d dBm\n", WiFi.RSSI());
    } else {
        Serial.println("\nWiFi connection failed! Restarting...");
        ESP.restart();
    }
}

void dataCollectionTask(void* parameter) {
    Serial.println("[Core 0] Data collection task started");
    
    while (true) {
        // Maintain WebSocket connection
        if (!krakenWS.isConnected()) {
            Serial.println("[Core 0] Reconnecting to Kraken...");
            krakenWS.connect();
            delay(5000);
            continue;
        }
        
        // Process WebSocket messages
        krakenWS.loop();
        
        // Check for new order book data
        if (krakenWS.hasNewData()) {
            OrderBookUpdate update = krakenWS.getLatestUpdate();
            
            // Update strategy with new data
            strategy.updateOrderBook(update.pair, update.orderBook);
            
            // Update cross-asset monitor
            crossAssetMonitor.updateAsset(update.pair, update.orderBook);
            
            metrics.messagesProcessed++;
        }
        
        // Feed watchdog
        esp_task_wdt_reset();
        
        // Small delay to prevent CPU hogging
        vTaskDelay(1 / portTICK_PERIOD_MS);
    }
}

void strategyExecutionTask(void* parameter) {
    Serial.println("[Core 1] Strategy execution task started");
    
    TickType_t lastExecutionTime = xTaskGetTickCount();
    const TickType_t executionPeriod = pdMS_TO_TICKS(100); // Execute every 100ms
    
    while (true) {
        // Wait for next execution period
        vTaskDelayUntil(&lastExecutionTime, executionPeriod);
        
        // Execute main strategy
        TradingSignal signal = strategy.evaluateSignal();
        
        if (signal.isValid) {
            Serial.printf("[SIGNAL] %s %s @ %.2f (confidence: %.2f)\n",
                signal.side == BUY ? "BUY" : "SELL",
                signal.pair,
                signal.price,
                signal.confidence
            );
            
            metrics.signalsGenerated++;
            
            // Here you would execute the trade
            // executeTrade(signal);
        }
        
        // Check for cross-asset contagion
        ContagionAlert alert = crossAssetMonitor.checkContagion();
        if (alert.detected) {
            Serial.printf("[CONTAGION] Detected between %s and %s (correlation: %.3f)\n",
                alert.asset1, alert.asset2, alert.correlation
            );
            
            // Adjust strategy parameters based on contagion
            strategy.adjustForContagion(alert);
        }
        
        // Performance monitoring
        if (PERFORMANCE_MONITORING) {
            static unsigned long lastMonitor = 0;
            if (millis() - lastMonitor > 10000) { // Every 10 seconds
                printPerformanceMetrics();
                lastMonitor = millis();
            }
        }
    }
}

void printPerformanceMetrics() {
    uint32_t currentHeap = ESP.getFreeHeap();
    
    Serial.println("\n=== Performance Metrics ===");
    Serial.printf("Uptime: %lu seconds\n", millis() / 1000);
    Serial.printf("Messages processed: %u\n", metrics.messagesProcessed);
    Serial.printf("Signals generated: %u\n", metrics.signalsGenerated);
    Serial.printf("Errors: %u\n", metrics.errors);
    Serial.printf("Free heap: %u bytes (delta: %d)\n", 
        currentHeap, (int)(currentHeap - metrics.lastHeapSize));
    Serial.printf("CPU0 usage: %.1f%%\n", Utils::getCPUUsage(0));
    Serial.printf("CPU1 usage: %.1f%%\n", Utils::getCPUUsage(1));
    Serial.println("=======================\n");
    
    metrics.lastHeapSize = currentHeap;
    
    // Check for memory leak
    if (metrics.lastHeapSize > 0 && currentHeap < metrics.lastHeapSize - 5000) {
        Serial.println("[WARNING] Possible memory leak detected!");
    }
}

void loop() {
    // Main loop is empty - all work done in tasks
    delay(1000);
    
    // Emergency restart if heap gets too low
    if (ESP.getFreeHeap() < 10000) {
        Serial.println("[CRITICAL] Low memory! Restarting...");
        ESP.restart();
    }
}
```

### `src/KrakenWebSocket.h`
```cpp
#ifndef KRAKEN_WEBSOCKET_H
#define KRAKEN_WEBSOCKET_H

#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include "config.h"

struct OrderBook {
    float bids[ORDER_BOOK_DEPTH][2];  // [price, volume]
    float asks[ORDER_BOOK_DEPTH][2];
    uint32_t timestamp;
    uint32_t sequenceNumber;
};

struct OrderBookUpdate {
    const char* pair;
    OrderBook orderBook;
    bool isSnapshot;
};

class KrakenWebSocket {
private:
    WebSocketsClient webSocket;
    WebSocketsClient futuresWebSocket;
    
    OrderBook orderBooks[NUM_PAIRS];
    bool connected = false;
    bool hasData = false;
    OrderBookUpdate latestUpdate;
    
    SemaphoreHandle_t dataMutex;
    
    void handleMessage(uint8_t* payload, size_t length);
    void parseOrderBookUpdate(JsonDocument& doc);
    void subscribeToChannels();
    
public:
    KrakenWebSocket();
    void init();
    bool connect();
    void disconnect();
    bool isConnected() { return connected; }
    void loop();
    
    bool hasNewData();
    OrderBookUpdate getLatestUpdate();
    OrderBook* getOrderBook(const char* pair);
};

#endif
```

### `src/OrderBookStrategy.h`
```cpp
#ifndef ORDERBOOK_STRATEGY_H
#define ORDERBOOK_STRATEGY_H

#include "KrakenWebSocket.h"
#include "CrossAssetMonitor.h"

enum Side { BUY, SELL };

struct TradingSignal {
    bool isValid;
    const char* pair;
    Side side;
    float price;
    float quantity;
    float confidence;
    uint32_t timestamp;
};

class OrderBookStrategy {
private:
    struct MarketMicrostructure {
        float spread;
        float midPrice;
        float imbalance;
        float depthRatio;
        float volatility;
    };
    
    MarketMicrostructure microstructure[NUM_PAIRS];
    float priceHistory[NUM_PAIRS][CORRELATION_WINDOW];
    int historyIndex = 0;
    
    float calculateImbalance(const OrderBook& book, int levels = 5);
    float calculateSpreadBps(const OrderBook& book);
    float calculateDepthRatio(const OrderBook& book);
    float calculateVolatility(int pairIndex);
    
public:
    void init();
    void updateOrderBook(const char* pair, const OrderBook& book);
    TradingSignal evaluateSignal();
    void adjustForContagion(const ContagionAlert& alert);
};

#endif
```

### `src/CrossAssetMonitor.h`
```cpp
#ifndef CROSS_ASSET_MONITOR_H
#define CROSS_ASSET_MONITOR_H

#include "config.h"

struct ContagionAlert {
    bool detected;
    const char* asset1;
    const char* asset2;
    float correlation;
    float leadLag;  // Positive = asset1 leads asset2
};

class CrossAssetMonitor {
private:
    struct AssetMetrics {
        float returns[CORRELATION_WINDOW];
        float volatility;
        int momentum;  // -1, 0, 1
        uint32_t lastUpdate;
    };
    
    AssetMetrics assets[NUM_PAIRS];
    float correlationMatrix[NUM_PAIRS][NUM_PAIRS];
    int windowIndex = 0;
    
    void updateCorrelations();
    float calculateCorrelation(int asset1, int asset2);
    float detectLeadLag(int asset1, int asset2);
    
public:
    void init();
    void updateAsset(const char* pair, const OrderBook& book);
    ContagionAlert checkContagion();
    float getCorrelation(const char* pair1, const char* pair2);
};

#endif
```

### `src/Utils.h`
```cpp
#ifndef UTILS_H
#define UTILS_H

#include <Arduino.h>

class Utils {
public:
    static float getCPUUsage(int core);
    static void printHexDump(const uint8_t* data, size_t length);
    static uint32_t getTimestamp();
    static float calculateEMA(float newValue, float oldEMA, float alpha);
    static void safeCopy(char* dest, const char* src, size_t maxLen);
    
    // Fixed-point math for performance
    static int32_t floatToFixed(float value, int fractionalBits = 16);
    static float fixedToFloat(int32_t value, int fractionalBits = 16);
};

// Circular buffer template for efficient data storage
template<typename T, size_t SIZE>
class CircularBuffer {
private:
    T buffer[SIZE];
    size_t head = 0;
    size_t tail = 0;
    size_t count = 0;
    
public:
    void push(const T& item) {
        buffer[head] = item;
        head = (head + 1) % SIZE;
        if (count < SIZE) count++;
        else tail = (tail + 1) % SIZE;
    }
    
    T& operator[](size_t index) {
        return buffer[(tail + index) % SIZE];
    }
    
    size_t size() const { return count; }
    bool empty() const { return count == 0; }
    void clear() { head = tail = count = 0; }
};

#endif
```

## 🧪 Testing

### `test/test_orderbook.cpp`
```cpp
#include <unity.h>
#include "../src/OrderBookStrategy.h"

void test_orderbook_imbalance() {
    OrderBook book;
    // Setup test order book
    book.bids[0][0] = 45000.0; book.bids[0][1] = 2.0;
    book.bids[1][0] = 44999.0; book.bids[1][1] = 1.5;
    book.asks[0][0] = 45001.0; book.asks[0][1] = 0.5;
    book.asks[1][0] = 45002.0; book.asks[1][1] = 1.0;
    
    OrderBookStrategy strategy;
    strategy.init();
    strategy.updateOrderBook("XBT/USD", book);
    
    TradingSignal signal = strategy.evaluateSignal();
    TEST_ASSERT_TRUE(signal.isValid);
    TEST_ASSERT_EQUAL(BUY, signal.side);
}

void setup() {
    UNITY_BEGIN();
    RUN_TEST(test_orderbook_imbalance);
    UNITY_END();
}

void loop() {}
```

## 🚀 Getting Started

### 1. Hardware Setup
- Connect ESP32 DOIT DevKit V1 to computer via USB
- Ensure drivers are installed (CP210x or CH340)

### 2. Software Installation
```bash
# Install PlatformIO CLI
pip install platformio

# Or use PlatformIO IDE in VSCode
```

### 3. Project Setup
```bash
# Clone or create project
cd "Quant Strat 1"

# Install dependencies
pio lib install

# Update config.h with your credentials
```

### 4. Build and Upload
```bash
# Build project
pio run

# Upload to ESP32
pio run --target upload

# Monitor serial output
pio device monitor -b 115200
```

### 5. Testing
```bash
# Run tests
pio test

# Run specific test
pio test -f test_orderbook
```

## 📊 Performance Optimization

### Memory Management
- Use stack allocation where possible
- Implement object pools for frequent allocations
- Monitor heap fragmentation

### CPU Optimization
- Use fixed-point math for non-critical calculations
- Leverage dual-core architecture
- Optimize JSON parsing with streaming

### Network Optimization
- Implement message compression
- Use binary protocols where possible
- Batch operations to reduce overhead

## 🔧 Troubleshooting

### Common Issues

1. **WiFi Connection Failures**
   - Check 2.4GHz compatibility
   - Verify credentials
   - Increase connection timeout

2. **WebSocket Disconnections**
   ```cpp
   // Add exponential backoff
   int reconnectDelay = 1000;
   while (!connected) {
       connect();
       delay(reconnectDelay);
       reconnectDelay = min(reconnectDelay * 2, 60000);
   }
   ```

3. **Memory Issues**
   - Enable PSRAM if available
   - Reduce buffer sizes
   - Use `ESP.getFreeHeap()` monitoring

4. **Watchdog Timeouts**
   - Add `esp_task_wdt_reset()` in long operations
   - Increase watchdog timeout
   - Optimize blocking operations

## 📈 Next Steps

1. **Enhanced Strategies**
   - Implement market making algorithms
   - Add statistical arbitrage
   - Machine learning integration

2. **Risk Management**
   - Position sizing algorithms
   - Stop-loss implementation
   - Portfolio optimization

3. **Production Deployment**
   - Secure credential storage
   - OTA updates
   - Remote monitoring

## 📚 Resources

- [ESP32 Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/)
- [Kraken API Documentation](https://docs.kraken.com/websockets/)
- [PlatformIO Documentation](https://docs.platformio.org/)

---

**Note**: This is a development framework. Always test thoroughly before using with real funds. Implement proper risk management and comply with exchange rate limits.