// SolarMesh — panel voltages over the Bridge + DC-motor cleaning

#include <Arduino_RouterBridge.h>

const pin_size_t MOTOR_PIN = D9;       // HIGH = motor on
const char *CLEANER_PANEL = "p1";      // panel that carries the cleaning motor
const uint32_t MAX_CLEAN_MS = 60000;

// Manual request from the "clean" RPC, served in loop().
volatile uint32_t req_ms = 0;
volatile bool req_new = false;
volatile bool req_stop = false;
volatile bool motor_on = false;

// Panels. VOLTS_PER_COUNT turns ADC counts into panel volts:
// 3.3 V full scale / 4095 counts, times the divider ratio (R_top + R_bottom) / R_bottom.
const pin_size_t PANEL_PINS[] = {A1, A2};
const char *PANEL_IDS[] = {"p1", "p2"};
const float DIVIDER_RATIO = 10.0f;      // set to the real divider ratio
const float VOLTS_PER_COUNT = 3.3f / 4095.0f * DIVIDER_RATIO;
const float PANEL_SPACING_M = 0.4f;    // distance between panel centres (cloud speed in m/s)
const int PANEL_SAMPLES = 32;          // averaged per reading

float panel_volts(pin_size_t pin) {
  long sum = 0;
  int n = 0;
  for (int i = 0; i < PANEL_SAMPLES; i++) {
    int v = analogRead(pin);
    if (v < 0) continue;
    sum += v;
    n++;
  }
  return n ? sum * VOLTS_PER_COUNT / n : 0.0f;
}

// Served from the Bridge thread (provide, not provide_safe), so it still
// answers while loop() is running the motor.
String get_mesh() {
  String json = "{\"nodes\":{";
  for (int i = 0; i < 2; i++) {
    if (i) json += ",";
    json += "\"" + String(PANEL_IDS[i]) + "\":" + String(panel_volts(PANEL_PINS[i]), 3);
  }
  json += "},\"edges\":[[\"p1\",\"p2\",1]],\"layout\":{\"p1\":[0,0],\"p2\":[";
  json += String(PANEL_SPACING_M, 2) + ",0]},";
  json += "\"cleaner\":{\"panel\":\"" + String(CLEANER_PANEL) + "\",\"on\":" + String(motor_on ? "true" : "false") + "}}";
  return json;
}

bool rpc_clean(int ms) {
  if (ms <= 0 || ms > (int)MAX_CLEAN_MS) return false;
  req_ms = ms;
  req_new = true;
  return true;
}

bool rpc_stop_clean() {
  req_stop = true;
  return true;
}

// Runs the motor; returns early on "stop_clean" or a new request.
void run_motor(uint32_t ms) {
  Serial.print("motor on for ");
  Serial.print(ms);
  Serial.println(" ms");
  motor_on = true;
  digitalWrite(MOTOR_PIN, HIGH);
  uint32_t t0 = millis();
  while (millis() - t0 < ms && !req_stop && !req_new) delay(10);
  digitalWrite(MOTOR_PIN, LOW);
  motor_on = false;
  Serial.println(req_stop ? "motor stopped" : "motor done");
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  Bridge.begin();
  Bridge.provide("get_mesh", get_mesh);
  Bridge.provide("clean", rpc_clean);
  Bridge.provide("stop_clean", rpc_stop_clean);
}

void loop() {
  if (req_new) {
    req_new = false;
    req_stop = false;
    run_motor(req_ms);
  } else {
    delay(20);
  }
}
