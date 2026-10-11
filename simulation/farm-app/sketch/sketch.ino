// SolarMesh — simulated rooftop farm on the STM32 (simulation only).

#include <Arduino_RouterBridge.h>
#include <math.h>
#include <zephyr/kernel.h>

// ---------- plant ----------
const int ROWS = 7;
const int STRINGS_PER_ROW = 2;
const int MODULES_PER_STRING = 20;
const int COLS = STRINGS_PER_ROW * MODULES_PER_STRING;
const int MODULES = ROWS * COLS;
const int STRINGS = ROWS * STRINGS_PER_ROW;
const float PITCH_X = 1.154f;   // 1.134 m module + 2 cm gap
const float PITCH_Y = 2.0f;     // flat-roof rack row pitch

// ---------- module physics (fitted to a 550 W datasheet) ----------
const float ISC_STC = 13.9f;
const float G_CLEAR = 900.0f;
const float NVT_SUB = 0.74f;
const float I0_SUB = 2.6e-9f;
const float RS_SUB = 0.035f;
const float BETA_V = -0.0026f;
const float V_BYPASS = -0.7f;
const float NOCT = 45.0f;
const float T_AMBIENT = 25.0f;
const float NOISE_V = 0.05f;

// ---------- weather + events ----------
const float DT = 0.5f;               // simulated seconds per reading
const float EDGE_M = 30.0f;
const float EDGE_BOOST = 0.10f;
const float DAY_S = 60.0f;           // soiling: 1 simulated day = 60 simulated seconds
const float MAX_SOILING = 0.4f;
const float SHADE = 0.3f;            // light left under debris / tree shadow
const int TREE_W = 3, TREE_H = 2;
const int MAX_EVENTS = 24;

enum Kind { EV_RESET = 0, EV_CLOUD = 1, EV_SOILING = 2, EV_DIODE = 3, EV_DEAD = 4, EV_DEBRIS = 5, EV_TREE = 6, EV_REPAIR = 7 };
const char *KIND_NAMES[] = {"reset", "cloud", "soiling", "diode", "dead", "debris", "tree", "repair"};

struct Cloud {
  float x0, y0, vx, vy, radius, transmittance, start, duration, heading;
};

struct Event {
  int kind;      // EV_SOILING: target = row; EV_DIODE/EV_DEAD/EV_DEBRIS: module; EV_TREE: top-left module
  int target;
  int sub;       // substring for EV_DIODE / EV_DEBRIS
  float start;
  float rate;    // EV_SOILING: % per simulated day
};

float volts[MODULES];          // being computed
float published[MODULES];      // last complete step, read by get_mesh
float published_t = 0, published_current = 0, published_temp = 25;
float sim_t = 0;
volatile bool step_requested = true;
const size_t JSON_MAX = 20000;
char json_buf[JSON_MAX] = "{}";   // last get_mesh reply, built in loop()
size_t json_len = 2;
struct k_mutex pub_mutex;
Cloud cloud;
bool have_cloud = false;
Event events[MAX_EVENTS];
int n_events = 0;

// Command from the sim_event RPC, applied in loop() between steps.
volatile int cmd_kind = -1, cmd_target = -1, cmd_param = 0;
volatile bool cmd_new = false;

// ---------- helpers ----------
float urand() { return rand() / (RAND_MAX + 1.0f); }

float gauss() {
  float u1 = urand() + 1e-7f, u2 = urand();
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

float module_x(int m) { return (m % COLS) * PITCH_X; }
float module_y(int m) { return (m / COLS) * PITCH_Y; }

void new_cloud(float start, float heading_deg, float speed) {
  float cx = (COLS - 1) * PITCH_X / 2, cy = (ROWS - 1) * PITCH_Y / 2;
  float heading = heading_deg * M_PI / 180.0f;
  cloud.heading = heading_deg;
  cloud.radius = 40 + urand() * 160;
  cloud.transmittance = 0.25f + urand() * 0.25f;
  float lead = cloud.radius + EDGE_M + 40;
  cloud.vx = speed * cosf(heading);
  cloud.vy = speed * sinf(heading);
  cloud.x0 = cx - lead * cosf(heading);
  cloud.y0 = cy - lead * sinf(heading);
  cloud.start = start;
  cloud.duration = 2 * lead / speed;
  have_cloud = true;
}

float sky(float x, float y, float t) {
  if (!have_cloud) return 1.0f;
  float dt = t - cloud.start;
  if (dt < 0 || dt > cloud.duration) return 1.0f;
  float d = hypotf(x - (cloud.x0 + cloud.vx * dt), y - (cloud.y0 + cloud.vy * dt)) - cloud.radius;
  if (d >= EDGE_M) return 1.0f;
  if (d <= 0) return cloud.transmittance;
  float ramp = d / EDGE_M;
  return cloud.transmittance + (1 - cloud.transmittance) * ramp + EDGE_BOOST * sinf(M_PI * ramp);
}

float substring_voltage(float iph, float current, float temp) {
  if (current >= iph) return V_BYPASS;
  float v = NVT_SUB * logf((iph - current) / I0_SUB + 1.0f) - current * RS_SUB;
  return v * (1.0f + BETA_V * (temp - 25.0f));
}

bool in_tree(const Event &e, int m) {
  int r0 = e.target / COLS, c0 = e.target % COLS, r = m / COLS, c = m % COLS;
  return r >= r0 && r < r0 + TREE_H && c >= c0 && c < c0 + TREE_W;
}

bool affects(const Event &e, int m) {
  if (e.kind == EV_SOILING) return m / COLS == e.target;
  if (e.kind == EV_TREE) return in_tree(e, m);
  return e.target == m;
}

// Substring photocurrents (and shorted flags) of module m at time t.
void substrings(int m, float t, float light, float iph[3], bool shorted[3]) {
  float g[3] = {light, light, light};
  bool dead = false;
  for (int k = 0; k < 3; k++) shorted[k] = false;
  for (int i = 0; i < n_events; i++) {
    const Event &e = events[i];
    if (!affects(e, m)) continue;
    if (e.kind == EV_SOILING) {
      float loss = fminf(e.rate / 100.0f * (t - e.start) / DAY_S, MAX_SOILING);
      for (int k = 0; k < 3; k++) g[k] *= 1 - loss;
    } else if (e.kind == EV_TREE) {
      for (int k = 0; k < 3; k++) g[k] *= SHADE;
    } else if (e.kind == EV_DEBRIS) {
      g[e.sub] *= SHADE;
    } else if (e.kind == EV_DIODE) {
      shorted[e.sub] = true;
    } else if (e.kind == EV_DEAD) {
      dead = true;
    }
  }
  for (int k = 0; k < 3; k++) iph[k] = dead ? 0.0f : ISC_STC * G_CLEAR / 1000.0f * g[k];
}

// One string: choose the MPPT current, write module voltages. Returns the current.
float simulate_string(int s, float t, float temp_amb, float &light_sum) {
  int first = (s / STRINGS_PER_ROW) * COLS + (s % STRINGS_PER_ROW) * MODULES_PER_STRING;
  static float iph[MODULES_PER_STRING][3];
  static bool shorted[MODULES_PER_STRING][3];
  float lights = 0;
  for (int i = 0; i < MODULES_PER_STRING; i++) {
    int m = first + i;
    float light = sky(module_x(m), module_y(m), t);
    lights += light;
    substrings(m, t, light, iph[i], shorted[i]);
  }
  light_sum += lights;
  float temp = temp_amb + (NOCT - 20.0f) / 800.0f * G_CLEAR * (lights / MODULES_PER_STRING);

  // Candidate currents: each distinct substring level (rounded to 0.05 A) x 3 fractions.
  float levels[MODULES_PER_STRING * 3];
  int nlev = 0;
  for (int i = 0; i < MODULES_PER_STRING; i++) {
    for (int k = 0; k < 3; k++) {
      float lv = roundf(iph[i][k] * 20) / 20;
      if (lv <= 0 || shorted[i][k]) continue;
      bool seen = false;
      for (int j = 0; j < nlev && !seen; j++) seen = fabsf(levels[j] - lv) < 1e-4f;
      if (!seen) levels[nlev++] = lv;
    }
  }
  const float fracs[3] = {0.90f, 0.94f, 0.97f};
  float best_i = 0, best_p = -1;
  for (int j = 0; j < nlev; j++) {
    for (int f = 0; f < 3; f++) {
      float cur = levels[j] * fracs[f], v = 0;
      for (int i = 0; i < MODULES_PER_STRING; i++)
        for (int k = 0; k < 3; k++) v += shorted[i][k] ? 0.0f : substring_voltage(iph[i][k], cur, temp);
      if (cur * v > best_p) { best_p = cur * v; best_i = cur; }
    }
  }
  for (int i = 0; i < MODULES_PER_STRING; i++) {
    float v = 0;
    for (int k = 0; k < 3; k++) v += shorted[i][k] ? 0.0f : substring_voltage(iph[i][k], best_i, temp);
    volts[first + i] = v + NOISE_V * gauss();
  }
  return best_i;
}

void simulate(float t) {
  if (have_cloud && t > cloud.start + cloud.duration) have_cloud = false;
  float light_sum = 0, current_sum = 0;
  for (int s = 0; s < STRINGS; s++) current_sum += simulate_string(s, t, T_AMBIENT, light_sum);

  k_mutex_lock(&pub_mutex, K_FOREVER);
  memcpy(published, volts, sizeof(volts));
  published_t = t;
  published_current = current_sum / STRINGS;
  published_temp = T_AMBIENT + (NOCT - 20.0f) / 800.0f * G_CLEAR * light_sum / MODULES;
  k_mutex_unlock(&pub_mutex);
}

// ---------- events ----------

int random_module() { return (int)(urand() * MODULES); }

void remove_events(int m) {
  int k = 0;
  for (int i = 0; i < n_events; i++)
    if (events[i].kind == EV_SOILING || !affects(events[i], m)) events[k++] = events[i];
  n_events = k;
}

void add_event(int kind, int target, float rate) {
  if (n_events >= MAX_EVENTS) remove_events(events[0].target);   // drop the oldest
  if (n_events >= MAX_EVENTS) return;
  events[n_events++] = {kind, target, (int)(urand() * 3), sim_t, rate};
}

void apply_command(int kind, int target, int param) {
  Serial.print("event ");
  Serial.print(KIND_NAMES[kind]);
  Serial.print(" target ");
  Serial.print(target);
  Serial.print(" param ");
  Serial.println(param);
  switch (kind) {
    case EV_RESET:
      n_events = 0;
      have_cloud = false;
      break;
    case EV_CLOUD:
      new_cloud(sim_t, target >= 0 ? target : urand() * 360, param > 0 ? param / 10.0f : 3 + urand() * 5);
      break;
    case EV_SOILING: {
      int row = (target >= 0 && target < ROWS) ? target : (int)(urand() * ROWS);
      add_event(EV_SOILING, row, param > 0 ? param : 3);
      break;
    }
    case EV_TREE: {
      int m = (target >= 0 && target < MODULES) ? target : random_module();
      int r = min(m / COLS, ROWS - TREE_H), c = min(m % COLS, COLS - TREE_W);
      add_event(EV_TREE, r * COLS + c, 0);
      break;
    }
    case EV_DIODE:
    case EV_DEAD:
    case EV_DEBRIS:
      add_event(kind, (target >= 0 && target < MODULES) ? target : random_module(), 0);
      break;
    case EV_REPAIR:
      if (target >= 0 && target < MODULES) remove_events(target);
      break;
  }
}

// ---------- RPC ----------

bool rpc_sim_event(int kind, int target, int param) {
  if (kind < EV_RESET || kind > EV_REPAIR) return false;
  cmd_kind = kind;
  cmd_target = target;
  cmd_param = param;
  cmd_new = true;
  return true;
}

// Builds the get_mesh reply in loop() (main thread), right after each step.
// The Bridge thread has a tiny stack, so it must not do this work itself.
void build_json() {
  String json;
  json.reserve(18500);
  json += "{\"time\":" + String(published_t, 1);
  json += ",\"temp\":" + String(published_temp, 1);
  json += ",\"current\":" + String(published_current, 2);
  json += ",\"nodes\":{";
  for (int m = 0; m < MODULES; m++) {
    if (m) json += ",";
    json += "\"m" + String(m + 1) + "\":" + String(published[m], 2);
  }

  json += "},\"edges\":[";
  bool first = true;
  for (int m = 0; m < MODULES; m++) {
    int r = m / COLS, c = m % COLS;
    if (c > 0) {
      json += first ? "" : ",";
      json += "[\"m" + String(m) + "\",\"m" + String(m + 1) + "\"," + String(c % MODULES_PER_STRING ? 1 : 0) + "]";
      first = false;
    }
    if (r > 0) {
      json += first ? "" : ",";
      json += "[\"m" + String(m - COLS + 1) + "\",\"m" + String(m + 1) + "\",0]";
      first = false;
    }
  }
  json += "],\"layout\":{";
  for (int m = 0; m < MODULES; m++) {
    if (m) json += ",";
    json += "\"m" + String(m + 1) + "\":[" + String(module_x(m), 2) + "," + String(module_y(m), 2) + "]";
  }
  json += "},\"sim\":{\"cloud\":" + String(have_cloud ? "true" : "false") + ",\"events\":[";
  for (int i = 0; i < n_events; i++) {
    if (i) json += ",";
    const Event &e = events[i];
    String target = e.kind == EV_SOILING ? "row " + String(e.target + 1) : "m" + String(e.target + 1);
    json += "[\"" + String(KIND_NAMES[e.kind]) + "\",\"" + target + "\"," + String(sim_t - e.start, 0) + "]";
  }
  json += "]}}";

  k_mutex_lock(&pub_mutex, K_FOREVER);
  json_len = json.length() < JSON_MAX ? json.length() : JSON_MAX - 1;
  memcpy(json_buf, json.c_str(), json_len);
  json_buf[json_len] = 0;
  k_mutex_unlock(&pub_mutex);
}

// Bridge thread: only copies the ready reply and asks loop() for the next step.
String get_mesh() {
  k_mutex_lock(&pub_mutex, K_FOREVER);
  String reply(json_buf);
  k_mutex_unlock(&pub_mutex);
  step_requested = true;
  return reply;
}

void setup() {
  Serial.begin(115200);
  k_mutex_init(&pub_mutex);
  srand(12345);
  simulate(0);
  build_json();
  Bridge.begin();
  Bridge.provide("get_mesh", get_mesh);
  Bridge.provide("sim_event", rpc_sim_event);
  Serial.println("simulated farm ready (events from the web interface)");
}

void loop() {
  if (cmd_new) {
    cmd_new = false;
    apply_command(cmd_kind, cmd_target, cmd_param);
  }
  if (step_requested) {
    step_requested = false;
    sim_t += DT;
    uint32_t t0 = millis();
    simulate(sim_t);
    build_json();
    if ((int)(sim_t / DT) % 20 == 0) {
      Serial.print("t=");
      Serial.print(sim_t, 1);
      Serial.print(" s, step ");
      Serial.print(millis() - t0);
      Serial.print(" ms, events ");
      Serial.println(n_events);
    }
  } else {
    delay(5);
  }
}
