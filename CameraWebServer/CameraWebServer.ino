/*
 * ESP32-CAM AI Thinker -> WebSocket client (envía frames JPEG al servidor)
 *
 * - Conecta a tu WiFi.
 * - Se conecta a un servidor WebSocket en la Raspi.
 * - Captura frames de la cámara y los envía como binario (JPEG) por WebSocket.
 *
 * Ajusta:
 *   - ssid / password
 *   - WS_HOST y WS_PORT (IP/puerto de tu Raspi)
 */

#include <WiFi.h>
#include "esp_camera.h"
#include <ArduinoWebsockets.h>

using namespace websockets;

// ---------------------- CONFIG WiFi + WebSocket ----------------------
const char* ssid     = "DIGIFIBRA-93HD";        // <-- CAMBIA ESTO
const char* password = "Ed7HDktEGA5k";    // <-- CAMBIA ESTO

// IP/host de tu Raspberry Pi (en la misma red)
const char* WS_HOST = "192.168.1.135";     // <-- CAMBIA ESTO (IP de la Raspi)
const uint16_t WS_PORT = 8765;            // Debe coincidir con el script de Python
const char* WS_PATH = "/";                // Path del WebSocket

WebsocketsClient wsClient;
bool ws_connected = false;                // <- estado actual del WS

unsigned long lastConnectAttempt = 0;
const unsigned long RECONNECT_INTERVAL_MS = 5000;

// ---------------------- CONFIG PINS AI THINKER ESP32-CAM ----------------------

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// (Opcional) Flash LED de la ESP32-CAM
#define FLASH_LED_GPIO    4

// ---------------------- EVENTOS WEBSOCKET ----------------------

void onWsEvent(WebsocketsEvent event, String data) {
  switch (event) {
    case WebsocketsEvent::ConnectionOpened:
      ws_connected = true;
      Serial.println("[WS] Conexión abierta");
      Serial.print("[WS] Mi IP local es: ");
      Serial.println(WiFi.localIP());
      // Mensaje opcional al servidor
      wsClient.send("ESP32-CAM conectada");
      break;

    case WebsocketsEvent::ConnectionClosed:
      ws_connected = false;
      Serial.println("[WS] Conexión cerrada");
      break;

    case WebsocketsEvent::GotPing:
      // Ping/pong automático
      // Serial.println("[WS] Ping recibido");
      break;

    case WebsocketsEvent::GotPong:
      // Serial.println("[WS] Pong recibido");
      break;
  }
}

// ---------------------- WIFI ----------------------

void connectToWifi() {
  Serial.printf("[WiFi] Conectando a %s\n", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 40) {
    delay(500);
    Serial.print(".");
    retry++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] Conectado. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[WiFi] No se pudo conectar.");
  }
}

// ---------------------- WEBSOCKET ----------------------

void connectToWebSocket() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  String url = String("ws://") + WS_HOST + ":" + String(WS_PORT) + WS_PATH;
  Serial.print("[WS] Conectando a ");
  Serial.println(url);

  bool ok = wsClient.connect(url);
  if (ok) {
    Serial.println("[WS] Petición de conexión enviada.");
  } else {
    Serial.println("[WS] Error al iniciar conexión WS.");
  }
}

// ---------------------- SETUP CÁMARA ----------------------

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Resolución / calidad
  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 10;          // 0-63 (más bajo = más calidad)
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] Error al inicializar la cámara 0x%x\n", err);
    return false;
  }

  Serial.println("[CAM] Cámara inicializada correctamente.");
  return true;
}

// ---------------------- SETUP & LOOP ----------------------

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n=== ESP32-CAM WebSocket Client ===");

  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);

  if (!initCamera()) {
    Serial.println("[ERROR] No se pudo iniciar la cámara. Reiniciando...");
    delay(5000);
    ESP.restart();
  }

  connectToWifi();

  wsClient.onEvent(onWsEvent);
  connectToWebSocket();
}

void loop() {
  // Mantener WebSocket vivo (recibir eventos, pings, etc.)
  wsClient.poll();

  // Si no estamos conectados al WS, reintentar cada X ms
  if (!ws_connected) {
    unsigned long now = millis();
    if (now - lastConnectAttempt > RECONNECT_INTERVAL_MS) {
      lastConnectAttempt = now;
      Serial.println("[WS] Reintentando conexión...");
      connectToWebSocket();
    }
    delay(10);
    return;
  }

  // Si estamos conectados, capturamos frame y lo enviamos
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAM] Error al capturar frame");
    delay(100);
    return;
  }

  // fb->buf contiene JPEG, fb->len su longitud
  bool ok = wsClient.sendBinary((const char*)fb->buf, fb->len);
  if (!ok) {
    Serial.println("[WS] Error al enviar frame");
  }

  esp_camera_fb_return(fb);

  // Ajusta la cadencia (20 fps aprox -> 50 ms)
  delay(50);
}
