#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include <SHA256.h>
#include <SHA512.h>
#include <ESPAsyncWebServer.h>
#include "config.h"

// Global objects
WebSocketsClient webSocket;
File csvFile;
AsyncWebServer server(80);
unsigned long dataCollectionStart = 0;
bool dataCollectionActive = false;
bool collectionComplete = false;
int messageCount = 0;

// Authentication state
bool isAuthenticated = false;
bool isSubscribedToOrderbook = false;
String currentChallenge = "";
String signedChallenge = "";

// Function declarations
void connectToWiFi();
void connectToWebSocket();
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length);
void parseOrderBookData(const char* jsonStr);
void initCSVFile();
void writeCSVHeader();
void writeDataToCSV(const String& timestamp, const String& symbol, const String& side, 
                    float price, float size);
void stopDataCollection();

// Authentication functions
String base64Encode(const uint8_t* data, size_t length);
String base64Decode(const String& input);
String signChallenge(const String& challenge);
void requestAuthentication();
void handleAuthenticationResponse(JsonDocument& doc);

// HTTP Server functions
void setupWebServer();
void handleRoot(AsyncWebServerRequest *request);
void handleDownload(AsyncWebServerRequest *request);
void handleFileList(AsyncWebServerRequest *request);

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("\n=== ESP32 Kraken Futures Market Data Collector ===");
    Serial.printf("CPU Frequency: %d MHz\n", getCpuFrequencyMhz());
    Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
    
    // Initialize SPIFFS for CSV storage
    if (!SPIFFS.begin(true)) {
        Serial.println("Failed to mount SPIFFS!");
        return;
    }
    Serial.println("SPIFFS mounted successfully");
    
    // Connect to WiFi
    connectToWiFi();
    
    // Setup HTTP server for file downloads
    setupWebServer();
    
    // Initialize CSV file
    initCSVFile();
    
    // Connect to Kraken Futures WebSocket
    connectToWebSocket();
    
    // Start data collection timer
    dataCollectionStart = millis();
    dataCollectionActive = true;
    
    Serial.println("Setup complete! Starting 10-minute data collection...\n");
}

void loop() {
    // Handle WebSocket events
    webSocket.loop();
    
    // Check connection status and reconnect if needed (less aggressive)
    static unsigned long lastConnectionCheck = 0;
    static unsigned long lastReconnectAttempt = 0;
    
    if (millis() - lastConnectionCheck > 10000) {  // Check every 10 seconds
        if (!webSocket.isConnected() && dataCollectionActive && 
            (millis() - lastReconnectAttempt > 15000)) {  // Only reconnect every 15 seconds
            Serial.println("WebSocket not connected, attempting reconnection...");
            connectToWebSocket();
            lastReconnectAttempt = millis();
        }
        lastConnectionCheck = millis();
    }
    
    // Check if collection time has passed
    if (dataCollectionActive && (millis() - dataCollectionStart) >= DATA_COLLECTION_DURATION_MS) {
        stopDataCollection();
    }
    
    // Small delay to prevent CPU hogging
    delay(10);  // Slightly longer delay for stability
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
        Serial.println("\nWiFi connection failed! Check your credentials.");
        return;
    }
}

void connectToWebSocket() {
#if USE_KRAKEN_SPOT
    Serial.println("Connecting to Kraken Spot WebSocket (Public Data)...");
    webSocket.beginSSL(KRAKEN_SPOT_WS_HOST, KRAKEN_SPOT_WS_PORT, KRAKEN_SPOT_WS_PATH);
#else
    Serial.println("Connecting to Kraken Futures WebSocket...");
    webSocket.beginSSL(KRAKEN_FUTURES_WS_HOST, KRAKEN_FUTURES_WS_PORT, KRAKEN_FUTURES_WS_PATH);
#endif
    
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocket.enableHeartbeat(30000, 5000, 2);
    
    // Set additional headers
    webSocket.setExtraHeaders("User-Agent: ESP32-Kraken-Client");
    
    Serial.println("WebSocket configuration complete");
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            Serial.printf("WebSocket Disconnected (code: %d)\n", length);
            Serial.printf("Free Heap: %d bytes\n", ESP.getFreeHeap());
            break;
            
        case WStype_CONNECTED:
            {
                Serial.printf("WebSocket Connected to: %s\n", payload);
                
                // Reset authentication state
                isAuthenticated = false;
                isSubscribedToOrderbook = false;
                
#if USE_KRAKEN_SPOT
                // Kraken Spot - no authentication needed
                String subscription = "{\"event\":\"subscribe\",\"pair\":[\"" SYMBOL "\"],\"subscription\":{\"name\":\"book\",\"depth\":10}}";
                webSocket.sendTXT(subscription);
                Serial.printf("Sent subscription: %s\n", subscription.c_str());
#else
                // Kraken Futures - request authentication first
                Serial.println("Requesting authentication for Kraken Futures...");
                requestAuthentication();
#endif
                break;
            }
            
        case WStype_TEXT:
            if (DEBUG_MODE) {
                Serial.printf("Received: %s\n", payload);
            }
            
            // Parse the JSON data
            parseOrderBookData((const char*)payload);
            messageCount++;
            break;
            
        case WStype_PONG:
            Serial.println("Received WebSocket PONG");
            break;
            
        case WStype_ERROR:
            Serial.printf("WebSocket Error: %s\n", payload);
            break;
            
        default:
            Serial.printf("Unknown WebSocket event type: %d\n", type);
            break;
    }
}

void parseOrderBookData(const char* jsonStr) {
    // Parse JSON
    DynamicJsonDocument doc(JSON_BUFFER_SIZE);
    DeserializationError error = deserializeJson(doc, jsonStr);
    
    if (error) {
        Serial.printf("JSON parsing failed: %s\n", error.c_str());
        return;
    }
    
#if !USE_KRAKEN_SPOT
    // Handle authentication responses first for Kraken Futures
    if (doc.containsKey("event")) {
        handleAuthenticationResponse(doc);
        
        // If we're not authenticated yet and not collecting data, return
        if (!isAuthenticated && !dataCollectionActive) {
            return;
        }
    }
#endif
    
    if (!dataCollectionActive) return;
    
    // Handle different message formats
#if USE_KRAKEN_SPOT
    // Kraken Spot orderbook format
    if (doc.containsKey(0) && doc[0].is<JsonArray>()) {
        // This is a Kraken Spot message: [channelID, data, channelName, pair]
        JsonArray data = doc[1];
        String channelName = doc[2];
        String pair = doc[3];
        
        if (channelName == "book-10" && pair == SYMBOL) {
            String timestamp = String(millis());
            
            // Parse bids and asks
            if (data.containsKey("b")) {
                JsonArray bids = data["b"];
                for (JsonArray bid : bids) {
                    float price = bid[0];
                    float qty = bid[1];
                    writeDataToCSV(timestamp, pair, "bid", price, qty);
                    
                    if (DEBUG_MODE) {
                        Serial.printf("[DATA] %s bid: %.6f @ %.2f\n", pair.c_str(), qty, price);
                    }
                }
            }
            
            if (data.containsKey("a")) {
                JsonArray asks = data["a"];
                for (JsonArray ask : asks) {
                    float price = ask[0];
                    float qty = ask[1];
                    writeDataToCSV(timestamp, pair, "ask", price, qty);
                    
                    if (DEBUG_MODE) {
                        Serial.printf("[DATA] %s ask: %.6f @ %.2f\n", pair.c_str(), qty, price);
                    }
                }
            }
        }
    }
#else
    // Kraken Futures orderbook format
    if (doc["feed"] == "book" && doc["product_id"] == SYMBOL) {
        String timestamp = String(millis());
        String symbol = doc["product_id"];
        String side = doc["side"];
        float price = doc["price"];
        float qty = doc["qty"];
        
        // Write to CSV
        writeDataToCSV(timestamp, symbol, side, price, qty);
        
        if (DEBUG_MODE) {
            Serial.printf("[DATA] %s %s: %.2f @ %.4f\n", 
                         symbol.c_str(), side.c_str(), qty, price);
        }
    }
#endif
    
    // Handle subscription confirmation
    if (doc["event"] == "subscribed" || doc["event"] == "subscriptionStatus") {
        Serial.println("Successfully subscribed to order book data");
        if (doc.containsKey("subscription")) {
            Serial.printf("Subscription details: %s\n", doc["subscription"].as<String>().c_str());
        }
    }
    
    // Handle system status
    if (doc["event"] == "systemStatus") {
        Serial.printf("System status: %s\n", doc["status"].as<String>().c_str());
    }
    
    // Handle errors
    if (doc["event"] == "error") {
        Serial.printf("Subscription error: %s\n", doc["errorMessage"].as<String>().c_str());
    }
}

void initCSVFile() {
    // Remove existing file
    if (SPIFFS.exists(CSV_FILENAME)) {
        SPIFFS.remove(CSV_FILENAME);
        Serial.println("Removed existing CSV file");
    }
    
    // Create new CSV file
    csvFile = SPIFFS.open(CSV_FILENAME, FILE_WRITE);
    if (!csvFile) {
        Serial.println("Failed to create CSV file!");
        return;
    }
    
    writeCSVHeader();
    csvFile.close();
    
    Serial.printf("CSV file initialized: %s\n", CSV_FILENAME);
}

void writeCSVHeader() {
    csvFile.println("timestamp_ms,symbol,side,price,size");
}

void writeDataToCSV(const String& timestamp, const String& symbol, const String& side, 
                    float price, float size) {
    csvFile = SPIFFS.open(CSV_FILENAME, FILE_APPEND);
    if (csvFile) {
        csvFile.printf("%s,%s,%s,%.6f,%.6f\n", 
                      timestamp.c_str(), symbol.c_str(), side.c_str(), price, size);
        csvFile.close();
    }
}

void stopDataCollection() {
    dataCollectionActive = false;
    collectionComplete = true;
    webSocket.disconnect();
    
    Serial.println("\n=== Data Collection Complete ===");
    Serial.printf("Duration: %d seconds\n", DATA_COLLECTION_DURATION_MS / 1000);
    Serial.printf("Messages received: %d\n", messageCount);
    
    // Show file size and basic stats
    if (SPIFFS.exists(CSV_FILENAME)) {
        File file = SPIFFS.open(CSV_FILENAME, FILE_READ);
        if (file) {
            size_t fileSize = file.size();
            Serial.printf("CSV file size: %d bytes\n", fileSize);
            file.close();
        }
    }
    
    Serial.println("\n🎉 DATA COLLECTION COMPLETE! 🎉");
    Serial.println("📊 Your market data is ready for analysis!");
    Serial.printf("🌐 Download your data: http://%s/download\n", WiFi.localIP().toString().c_str());
    Serial.printf("📋 File list: http://%s/files\n", WiFi.localIP().toString().c_str());
    Serial.printf("💾 Free Heap: %d bytes\n", ESP.getFreeHeap());
    Serial.println("\nESP32 will continue running the HTTP server for file downloads.");
}

// Base64 Functions
String base64Encode(const uint8_t* data, size_t length) {
    const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    String result = "";
    int pad = length % 3;
    
    for (size_t i = 0; i < length; i += 3) {
        uint32_t tmp = (data[i] << 16);
        if (i + 1 < length) tmp |= (data[i + 1] << 8);
        if (i + 2 < length) tmp |= data[i + 2];
        
        result += chars[(tmp >> 18) & 0x3F];
        result += chars[(tmp >> 12) & 0x3F];
        result += (i + 1 < length) ? chars[(tmp >> 6) & 0x3F] : '=';
        result += (i + 2 < length) ? chars[tmp & 0x3F] : '=';
    }
    
    return result;
}

String base64Decode(const String& input) {
    const String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    String result = "";
    int in_len = input.length();
    int i = 0;
    int bin = 0;
    int val = 0;
    
    while (i < in_len) {
        if (input[i] > 64 && input[i] < 91) val = input[i] - 65;
        else if (input[i] > 96 && input[i] < 123) val = input[i] - 71;
        else if (input[i] > 47 && input[i] < 58) val = input[i] + 4;
        else if (input[i] == '+') val = 62;
        else if (input[i] == '/') val = 63;
        else if (input[i] == '=') break;
        else continue;
        
        bin = (bin << 6) | val;
        if (++i % 4 == 0) {
            result += char((bin >> 16) & 0xFF);
            result += char((bin >> 8) & 0xFF);
            result += char(bin & 0xFF);
            bin = 0;
        }
    }
    
    if (i % 4 == 2) result += char((bin >> 4) & 0xFF);
    else if (i % 4 == 3) {
        result += char((bin >> 10) & 0xFF);
        result += char((bin >> 2) & 0xFF);
    }
    
    return result;
}

// Authentication Functions for Kraken Futures
String signChallenge(const String& challenge) {
    // Step 1: Hash challenge with SHA-256
    SHA256 sha256;
    sha256.update((const uint8_t*)challenge.c_str(), challenge.length());
    uint8_t challengeHash[32];
    sha256.finalize(challengeHash, 32);
    
    // Step 2: Base64-decode API secret
    String decodedSecret = base64Decode(KRAKEN_FUTURES_API_SECRET);
    
    // Step 3: HMAC-SHA-512 with decoded secret
    SHA512 sha512;
    sha512.resetHMAC((const uint8_t*)decodedSecret.c_str(), decodedSecret.length());
    sha512.update(challengeHash, 32);
    uint8_t signature[64];
    sha512.finalizeHMAC((const uint8_t*)decodedSecret.c_str(), decodedSecret.length(), signature, 64);
    
    // Step 4: Base64-encode result
    return base64Encode(signature, 64);
}

void requestAuthentication() {
    String challengeRequest = "{\"event\":\"challenge\",\"api_key\":\"" + String(KRAKEN_FUTURES_API_KEY) + "\"}";
    webSocket.sendTXT(challengeRequest);
    Serial.printf("Sent challenge request: %s\n", challengeRequest.c_str());
}

void handleAuthenticationResponse(JsonDocument& doc) {
    String event = doc["event"];
    
    if (event == "challenge") {
        // Received challenge, sign it and send back
        currentChallenge = doc["message"].as<String>();
        Serial.printf("Received challenge: %s\n", currentChallenge.c_str());
        
        signedChallenge = signChallenge(currentChallenge);
        Serial.println("Challenge signed successfully");
        
        // Send authentication with heartbeat subscription
        String authMessage = "{\"event\":\"subscribe\",\"feed\":\"heartbeat\",\"api_key\":\"" + 
                           String(KRAKEN_FUTURES_API_KEY) + "\",\"original_challenge\":\"" + 
                           currentChallenge + "\",\"signed_challenge\":\"" + signedChallenge + "\"}";
        
        webSocket.sendTXT(authMessage);
        Serial.println("Sent authentication message");
        
    } else if (event == "subscribed" && doc["feed"] == "heartbeat") {
        // Authentication successful
        isAuthenticated = true;
        Serial.println("✅ Authentication successful!");
        
        // Only subscribe to orderbook if we haven't already
        if (!isSubscribedToOrderbook) {
            String subscription = "{\"event\":\"subscribe\",\"feed\":\"book\",\"product_ids\":[\"" + 
                                String(SYMBOL) + "\"],\"api_key\":\"" + String(KRAKEN_FUTURES_API_KEY) + 
                                "\",\"original_challenge\":\"" + currentChallenge + 
                                "\",\"signed_challenge\":\"" + signedChallenge + "\"}";
            
            webSocket.sendTXT(subscription);
            Serial.printf("Sent orderbook subscription: %s\n", subscription.c_str());
        }
        
    } else if (event == "subscribed" && doc["feed"] == "book") {
        // Orderbook subscription successful
        isSubscribedToOrderbook = true;
        Serial.println("✅ Orderbook subscription successful! Waiting for market data...");
        
        JsonArray product_ids = doc["product_ids"];
        if (product_ids.size() > 0) {
            Serial.printf("Subscribed to symbol: %s\n", product_ids[0].as<String>().c_str());
        }
        
    } else if (event == "error") {
        Serial.printf("❌ Authentication error: %s\n", doc["message"].as<String>().c_str());
    }
}

// HTTP Server Implementation
void setupWebServer() {
    // Root page
    server.on("/", HTTP_GET, handleRoot);
    
    // Download CSV file
    server.on("/download", HTTP_GET, handleDownload);
    
    // List files
    server.on("/files", HTTP_GET, handleFileList);
    
    // Serve files from SPIFFS
    server.serveStatic("/", SPIFFS, "/");
    
    // Start server
    server.begin();
    
    Serial.printf("HTTP server started on: http://%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("Download URL: http://%s/download\n", WiFi.localIP().toString().c_str());
}

void handleRoot(AsyncWebServerRequest *request) {
    String html = "<!DOCTYPE html><html><head><title>ESP32 Market Data Collector</title>";
    html += "<style>body{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5;}";
    html += ".container{background:white;padding:30px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}";
    html += "h1{color:#333;border-bottom:2px solid #4CAF50;padding-bottom:10px;}";
    html += ".status{padding:15px;border-radius:5px;margin:10px 0;}";
    html += ".complete{background:#d4edda;color:#155724;border:1px solid #c3e6cb;}";
    html += ".active{background:#d1ecf1;color:#0c5460;border:1px solid #bee5eb;}";
    html += "a{display:inline-block;background:#4CAF50;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;margin:10px 5px 0 0;}";
    html += "a:hover{background:#45a049;}</style></head><body>";
    html += "<div class='container'>";
    html += "<h1>📊 ESP32 Kraken Market Data Collector</h1>";
    
    if (collectionComplete) {
        html += "<div class='status complete'>✅ Data collection complete!</div>";
        html += "<p><strong>Messages collected:</strong> " + String(messageCount) + "</p>";
        html += "<p><strong>Duration:</strong> " + String(DATA_COLLECTION_DURATION_MS / 1000) + " seconds</p>";
        
        if (SPIFFS.exists(CSV_FILENAME)) {
            File file = SPIFFS.open(CSV_FILENAME, FILE_READ);
            if (file) {
                html += "<p><strong>File size:</strong> " + String(file.size()) + " bytes</p>";
                file.close();
            }
        }
        
        html += "<a href='/download'>📥 Download CSV Data</a>";
        html += "<a href='/files'>📋 View File List</a>";
    } else if (dataCollectionActive) {
        html += "<div class='status active'>🔄 Data collection in progress...</div>";
        unsigned long elapsed = (millis() - dataCollectionStart) / 1000;
        unsigned long remaining = (DATA_COLLECTION_DURATION_MS / 1000) - elapsed;
        html += "<p><strong>Elapsed time:</strong> " + String(elapsed) + " seconds</p>";
        html += "<p><strong>Remaining time:</strong> " + String(remaining) + " seconds</p>";
        html += "<p><strong>Messages collected:</strong> " + String(messageCount) + "</p>";
        html += "<p>Please wait for collection to complete...</p>";
        html += "<script>setTimeout(function(){location.reload();}, 5000);</script>";
    } else {
        html += "<div class='status'>⏱️ Waiting to start collection...</div>";
    }
    
    html += "<p><strong>Symbol:</strong> " + String(SYMBOL) + "</p>";
    html += "<p><strong>ESP32 IP:</strong> " + WiFi.localIP().toString() + "</p>";
    html += "</div></body></html>";
    
    request->send(200, "text/html", html);
}

void handleDownload(AsyncWebServerRequest *request) {
    if (SPIFFS.exists(CSV_FILENAME)) {
        // Generate filename with timestamp
        String filename = "kraken_market_data_" + String(millis()) + ".csv";
        request->send(SPIFFS, CSV_FILENAME, String(), true);
    } else {
        request->send(404, "text/plain", "CSV file not found");
    }
}

void handleFileList(AsyncWebServerRequest *request) {
    String html = "<!DOCTYPE html><html><head><title>File List</title></head><body>";
    html += "<h2>📁 Files on ESP32</h2>";
    
    File root = SPIFFS.open("/");
    File file = root.openNextFile();
    
    html += "<ul>";
    while (file) {
        html += "<li><a href='/" + String(file.name()) + "'>" + String(file.name()) + "</a> (" + String(file.size()) + " bytes)</li>";
        file = root.openNextFile();
    }
    html += "</ul>";
    
    html += "<p><a href='/'>← Back to Home</a></p>";
    html += "</body></html>";
    
    request->send(200, "text/html", html);
}