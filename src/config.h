#ifndef CONFIG_H
#define CONFIG_H

// WiFi Configuration - UPDATE THESE VALUES
#define WIFI_SSID "Freebox-49B068"
#define WIFI_PASSWORD "immittam-agentis6-stimulis*?-subicerer"

// Kraken API Credentials - UPDATE THESE WITH YOUR CREDENTIALS
#define KRAKEN_API_KEY "P0MDWQeRX0Hl5pgB7daUxrS5rz/onqPiC386fk8TAMgWQDm+aniwEe/O"
#define KRAKEN_API_SECRET "IYezMHnEmzsvstztvFb0bkZBCevXYsDz9xJae1Z4LNubSs1tuPuwM93Go+dFIS6GiSEn17MkZLYkHt951pZxEA=="
#define KRAKEN_FUTURES_API_KEY "AOxfE4UNeV5ob4NCQl0vEpuxASWMVFqInnW1JbYBdH1N6VPNKVCuQK5l"
#define KRAKEN_FUTURES_API_SECRET "JizTbgsW6RfHzAU1k/w0R7jQDQ4XXlOgbAitL9SSPdr/W54L2gEH7UNS9IyXEu+CtZctMtSMo7HGql0jtitjpC5H"



// Use Kraken Futures (requires API credentials)
#define USE_KRAKEN_SPOT 0  // 1 = use spot, 0 = use futures

// Kraken Spot WebSocket Configuration (PUBLIC DATA AVAILABLE)
#define KRAKEN_SPOT_WS_HOST "ws.kraken.com"
#define KRAKEN_SPOT_WS_PORT 443
#define KRAKEN_SPOT_WS_PATH "/"

// Kraken Futures WebSocket Configuration (REQUIRES API KEY)
#define KRAKEN_FUTURES_WS_HOST "futures.kraken.com"
#define KRAKEN_FUTURES_WS_PORT 443
#define KRAKEN_FUTURES_WS_PATH "/ws/v1"

// Trading Symbols to monitor
#if USE_KRAKEN_SPOT
#define SYMBOL "XBT/USD"  // Bitcoin/USD spot (public orderbook available)
#else
#define SYMBOL "PI_XBTUSD"  // Bitcoin/USD perpetual inverse futures (more active)
#endif

// Data Collection Configuration
#define DATA_COLLECTION_DURATION_MS 600000  // 10 minutes (600 seconds)
#define CSV_FILENAME "/market_data.csv"
#define ENABLE_HTTP_SERVER 1  // Enable HTTP server for file download

// Buffer Sizes
#define WEBSOCKET_BUFFER_SIZE 4096
#define JSON_BUFFER_SIZE 8192
#define CSV_BUFFER_SIZE 512

// Debug Configuration
#define DEBUG_MODE 1

#endif