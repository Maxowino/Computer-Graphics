// Checkpoint 3 - OpenGL integration
//
// The pure C++ work from scenes  1 and 2 is reused as it is:
//   Vec3 / Mat4 / lookAt / perspective   -> loaded straight into OpenGL (glLoadMatrixd)
//   cubic Bezier + tangent               -> missile paths, missile orientation
//   easing + keyframes                   -> camera path and sky colours
//   procedural city, collapse timing     -> same generator and the same war plan
//
// 
//
// Uses fixed-function OpenGL 1.x + GLFW (window and input only), so no GLAD/GLEW is needed.
//
// Build
//   Windows (MSYS2 UCRT64): g++ -std=c++17 -O2 scene3.cpp -o graphicsdemo.exe -lglfw3 -lopengl32 -lgdi32
//   Linux:                  g++ -std=c++17 -O2 scene3.cpp -o graphicsdemo.exe -lglfw -lGL
//   macOS (Homebrew):       g++ -std=c++17 -O2 scene3.cpp -o graphicsdemo.exe -I/opt/homebrew/include -L/opt/homebrew/lib -lglfw -framework OpenGL
//
// Controls:
//   SPACE pause/resume   LEFT/RIGHT seek -/+ 5 s   UP/DOWN speed x2 / /2   R restart
//   1..9 jump to major story sections, including the final sewer rebellion
//   F free camera (WASD move, Q/E down/up, hold left mouse button to look, SHIFT faster)
//   S save screenshot (PPM)   ESC quit
// Command line (all times are FILM seconds):
//   ./graphicsdemo.exe             play from the start
//   ./graphicsdemo.exe <seconds>             start at that time
//   ./graphicsdemo.exe --shot <seconds> <file.ppm>   render one frame to a file and exit

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

const double PI = 3.14159265358979323846;
const double FOV_Y = 36.0 * PI / 180.0;
const double TOTAL_TIME = 350.0;                       // extended story: sewer-rebellion ending
const double REF_W = 960, REF_H = 560;                 // reference screen size used for the sun arc

// ---- NEW: story beats and slow motion (same as Checkpoint 2) --------------------------
const double SH_T    = 266.0;                          // the statehouse is hit
const double RALLY_T = 274.0;                          // City A leader and defenders appear
const double LEADER_Z = 6.2;                           // leader stands on the podium at z = 6.2
const double SLOW_A = SH_T - 1.0, SLOW_B = SH_T + 3.5, SLOW_K = 0.3;     // film runs at 0.3x speed in [A, B]
const double SLOW_EXTRA = (SLOW_B - SLOW_A) * (1.0 / SLOW_K - 1.0);      // extra film seconds (10.5)
const double FILM_TIME = TOTAL_TIME + SLOW_EXTRA;                        // length of the FILM
double storyTime(double f) {                           // film time -> story time
    if (f <= SLOW_A) return f;
    double fB = SLOW_A + (SLOW_B - SLOW_A) / SLOW_K;
    if (f <= fB) return SLOW_A + (f - SLOW_A) * SLOW_K;
    return f - SLOW_EXTRA;
}
double filmTime(double s) {                            // story time -> film time
    if (s <= SLOW_A) return s;
    if (s <= SLOW_B) return SLOW_A + (s - SLOW_A) / SLOW_K;
    return s + SLOW_EXTRA;
}

struct Color { uint8_t r, g, b; };

// ======================================================================= Math helpers
double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
double lerp(double a, double b, double t) { return a + (b - a) * t; }
double smoothstep(double x) { x = clamp01(x); return x * x * (3 - 2 * x); }
double easeOutCubic(double x) { double u = 1 - clamp01(x); return 1 - u * u * u; }
double easeInCubic(double x) { x = clamp01(x); return x * x * x; }

struct Vec3 { double x, y, z; };
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalize(Vec3 v) { double l = std::sqrt(dot(v, v)); return l > 0 ? v * (1.0 / l) : v; }
Vec3 lerp(Vec3 a, Vec3 b, double t) { return a + (b - a) * t; }

Color lerpColor(Color a, Color b, double t) {
    return Color{static_cast<uint8_t>(lerp(a.r, b.r, t)), static_cast<uint8_t>(lerp(a.g, b.g, t)),
                 static_cast<uint8_t>(lerp(a.b, b.b, t))};
}
Color shade(Color c, double k) {
    auto f = [k](int v) { return static_cast<uint8_t>(std::min(255.0, std::max(0.0, v * k))); };
    return Color{f(c.r), f(c.g), f(c.b)};
}

// Cubic Bezier curve and its derivative (tangent)
Vec3 bezier(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, double u) {
    double v = 1 - u;
    return p0 * (v * v * v) + p1 * (3 * v * v * u) + p2 * (3 * v * u * u) + p3 * (u * u * u);
}
Vec3 bezierTangent(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, double u) {
    double v = 1 - u;
    return ((p1 - p0) * (v * v) + (p2 - p1) * (2 * v * u) + (p3 - p2) * (u * u)) * 3.0;
}

// Deterministic randomness: a sequential generator (city layout) and a hash (particles)
static uint32_t g_seed = 12345;
double rnd() { g_seed = g_seed * 1664525u + 1013904223u; return ((g_seed >> 8) & 0xFFFF) / 65536.0; }
double hash01(uint32_t a, uint32_t b) {
    uint32_t h = a * 374761393u + b * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xFFFFFF) / 16777216.0;
}

// ======================================================================= Matrices
struct Vec4 { double x, y, z, w; };

struct Mat4 {
    double m[4][4] = {};
    static Mat4 identity() { Mat4 r; for (int i = 0; i < 4; ++i) r.m[i][i] = 1; return r; }
    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                double s = 0;
                for (int k = 0; k < 4; ++k) s += m[i][k] * o.m[k][j];
                r.m[i][j] = s;
            }
        return r;
    }
    Vec4 operator*(const Vec4& v) const {
        return { m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
                 m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
                 m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
                 m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w };
    }
    static Mat4 perspective(double fovy, double aspect, double n, double f) {
        Mat4 r; double t = 1.0 / std::tan(fovy / 2);
        r.m[0][0] = t / aspect; r.m[1][1] = t;
        r.m[2][2] = (f + n) / (n - f); r.m[2][3] = 2 * f * n / (n - f);
        r.m[3][2] = -1;
        return r;
    }
};

Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = normalize(target - eye), s = normalize(cross(f, up)), u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
    return r;
}

// CHANGED: y0 = bottom of the box (default 0), so statehouse parts can stand on other parts.
struct Box { double x0, z0, x1, z1, h; Color base; double sx = 0, sz = 0, y0 = 0; };

// ======================================================================= World model
const double PITCH = 8.0, BLOCK = 6.0;
const int    N_MAIN = 9;
const double HALF = N_MAIN * PITCH / 2.0;           // 36
const Vec3   RIVAL_CENTRE = {-125, 0, 0};
const int    N_RIVAL = 5;

struct Building {
    Box box;                       // full-size geometry (h = final height)
    double growStart = 0, growDur = 10;
    double collapseT = 1e9, collapseDur = 4;
    double shearX = 0, shearZ = 0; // random lean when collapsing
    bool tree = false;
    int city = 0;
    // NEW: statehouse chunks fly apart instead of collapsing
    bool sh = false;
    Vec3 vel{0, 0, 0}, spin{0, 0, 0};
    double tland = 0;              // seconds after the strike until the chunk lies on the ground
};
struct Block   { double x0, z0; bool park; int city; bool plaza = false; };      // CHANGED: plaza flag
struct Missile { Vec3 p0, p1, p2, p3; double launchT, flightT; bool big = false; };   // CHANGED: big = statehouse missile
struct Car     { int road; bool alongX; double lane, speed, offset, dir, startT, endT; Color col; };
struct Person  { double x, z, phase, appear; Color col; };                       // NEW: citizen in the crowd

struct World {
    std::vector<Building> buildings;
    std::vector<Block> blocks;
    std::vector<Missile> missiles;
    std::vector<Car> cars;
    std::vector<Person> crowd;     // NEW
};

// NEW: the statehouse, 17 box chunks around the middle block (x, z in -3..3).
// Each chunk gets its own launch velocity and spin for the explosion (from a hash, so
// the sequence of rnd() used for the city is not disturbed).
void addStatehouse(World& w) {
    struct Part { double x0, x1, z0, z1, y0, y1; Color c; };
    const Color stone{200, 195, 185}, white{240, 236, 228}, gold{225, 190, 70};
    std::vector<Part> P = {
        {-3.0, 3.0, -3.0, 3.0, 0.0, 0.5, stone},     // base slab
        {-2.7, 2.7, -2.7, 2.7, 0.5, 1.0, stone},     // steps
        {-2.2, 2.2, -2.0, 1.2, 1.0, 4.6, white},     // main hall
        {-2.4, 2.4,  1.2, 1.9, 4.4, 4.9, stone},     // entablature
        {-2.3, 2.3, -2.1, 1.3, 4.6, 4.9, stone},     // hall roof
        {-2.0, 2.0,  1.2, 1.9, 4.9, 5.4, white},     // pediment
        {-1.2, 1.2,  1.2, 1.9, 5.4, 5.9, white},
        {-1.1, 1.1, -1.5, 0.7, 4.9, 6.4, white},     // drum
        {-0.9, 0.9, -1.3, 0.5, 6.4, 7.1, gold},      // dome
        {-0.55, 0.55, -0.95, 0.15, 7.1, 7.6, gold},
        {-0.08, 0.08, -0.48, -0.32, 7.6, 9.0, gold}, // spire
    };
    for (int i = 0; i < 6; ++i) {                    // six columns
        double x = -2.0 + 0.8 * i;
        P.push_back({x - 0.15, x + 0.15, 1.3, 1.65, 1.0, 4.4, white});
    }
    uint32_t id = 0;
    for (const Part& p : P) {
        Building b; b.sh = true; b.city = 0;
        b.box = {p.x0, p.z0, p.x1, p.z1, p.y1, p.c};
        b.box.y0 = p.y0;
        b.growStart = 25 + p.y0 * 3.0; b.growDur = 14;     // grows bottom-up while the city rises
        b.collapseT = SH_T + 2.5;                           // (only used to start the smoke, after the blast)
        Vec3 c = {(p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2, (p.z0 + p.z1) / 2};
        Vec3 d = c - Vec3{0, 7, 0};                         // blast comes from the dome
        d.x += hash01(id, 1) - 0.5; d.z += hash01(id, 2) - 0.5;
        d = normalize(d);
        b.vel = d * (3 + 6 * hash01(id, 3)) + Vec3{0, 4 + 5 * hash01(id, 4), 0};
        b.vel.z = std::min(b.vel.z, 1.2);                   // keep the debris off the plaza (leader + crowd)
        b.spin = {(hash01(id, 5) - 0.5) * 6, (hash01(id, 6) - 0.5) * 6, (hash01(id, 7) - 0.5) * 6};
        double floorY = 0.5 * std::min({p.x1 - p.x0, p.y1 - p.y0, p.z1 - p.z0});
        double vy = b.vel.y;
        b.tland = (vy + std::sqrt(vy * vy + 19.6 * (c.y - floorY))) / 9.8;   // solve y(t) = floor
        w.buildings.push_back(b);
        ++id;
    }
}

void generateCity(World& w, double cx, double cz, int n, int city) {
    const Color mainDown[3] = {{110, 140, 190}, {95, 125, 170}, {135, 160, 200}};
    const Color mainMid[3]  = {{205, 190, 165}, {190, 175, 150}, {215, 200, 180}};
    const Color mainLow[3]  = {{230, 205, 180}, {220, 170, 140}, {235, 220, 200}};
    const Color rivalCol[3] = {{190, 90, 80}, {165, 70, 70}, {210, 120, 100}};
    const double half = n * PITCH / 2.0, mid = (n - 1) / 2.0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double bx = cx - half + i * PITCH + 1.0, bz = cz - half + j * PITCH + 1.0;
            double dist = std::sqrt((i - mid) * (i - mid) + (j - mid) * (j - mid));
            double nd = dist / mid;
            bool park = city == 0 && nd >= 0.6 && ((i * 7 + j * 13) % 11 == 0);
            w.blocks.push_back({bx, bz, park, city});
            double start = city == 0 ? 20 + dist * 14 : 180;

            // NEW: middle block = statehouse, blocks south of it (x -3..3, z 5..37) = plaza and avenue
            if (city == 0 && i == 4 && j == 4) { addStatehouse(w); continue; }
            if (city == 0 && i == 4 && j >= 5) { w.blocks.back().plaza = true; continue; }

            if (park) {
                for (int t = 0; t < 7; ++t) {
                    double tx = bx + 0.5 + rnd() * (BLOCK - 1.6), tz = bz + 0.5 + rnd() * (BLOCK - 1.6);
                    Building b; b.tree = true; b.city = city;
                    b.box = {tx, tz, tx + 0.9, tz + 0.9, 1.4 + rnd() * 1.2, Color{40, 115, 55}};
                    b.growStart = start + rnd() * 8; b.growDur = 6;
                    w.buildings.push_back(b);
                }
                continue;
            }
            for (int a = 0; a < 2; ++a) {
                for (int c = 0; c < 2; ++c) {
                    if (nd > 0.85 && rnd() < 0.10) continue;
                    double lx = bx + a * 3.2, lz = bz + c * 3.2;
                    double in0 = rnd() * 0.4, in1 = rnd() * 0.4;
                    double h; Color col;
                    int pick = static_cast<int>(rnd() * 3);
                    if (nd < 0.35)      { h = 10 + rnd() * 11; col = mainDown[pick]; }
                    else if (nd < 0.8)  { h = 5 + rnd() * 6;   col = mainMid[pick]; }
                    else                { h = 2 + rnd() * 3.5; col = mainLow[pick]; }
                    if (city == 1)      { h *= 0.85; col = rivalCol[pick]; }
                    Building b; b.city = city;
                    b.box = {lx + in0, lz + in1, lx + 2.8 - in1, lz + 2.8 - in0, h, col};
                    b.growStart = start + rnd() * 6 + (city == 1 ? rnd() * 8 : 0);
                    b.growDur = 8 + rnd() * 5;
                    w.buildings.push_back(b);
                }
            }
        }
    }
}

// NEW: 8 x 7 citizens on the plaza in front of the leader (appear one after another)
void makeCrowd(World& w) {
    const Color shirts[6] = {{180, 60, 60}, {60, 90, 170}, {70, 140, 80}, {200, 160, 60}, {120, 80, 150}, {90, 90, 100}};
    for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 8; ++i) {
            uint32_t id = static_cast<uint32_t>(j * 8 + i);
            Person p;
            p.x = -2.8 + i * 0.8 + (hash01(id, 11) - 0.5) * 0.15;
            p.z = 8.6 + j * 0.85 + (hash01(id, 12) - 0.5) * 0.15;
            p.phase = hash01(id, 13) * 2 * PI;
            p.appear = RALLY_T + 0.15 * id;
            p.col = shirts[(i * 5 + j * 3) % 6];
            w.crowd.push_back(p);
        }
}

// Plans the attack: missiles on Bezier paths, collapse times for nearby buildings, traffic
// CHANGED: instead of 20 evenly spaced missiles the battle now has three beats:
//   1) first volley: 6 missiles on the western districts      (impacts 236 ... 263)
//   2) the statehouse missile                                  (impact SH_T = 266)
//   3) escalation: 26 missiles, the gaps shrink from ~2 s to ~0.3 s (impacts 270 ... 300)
void planWar(World& w) {
    struct Shot { Vec3 T; double impactT; bool big; };
    std::vector<Shot> shots;
    const double volley[6] = {236, 242, 248, 254, 259, 263};
    for (double it : volley) shots.push_back({{-34.0 + rnd() * 20.0, 6.0, (rnd() - 0.5) * 60.0}, it, false});
    shots.push_back({{0.0, 8.0, 0.0}, SH_T, true});
    const int NE = 26;
    for (int k = 0; k < NE; ++k) {
        double it = 270.0 + 30.0 * (1.0 - std::pow(1.0 - static_cast<double>(k) / NE, 1.7));
        Vec3 T;
        do { T = {(rnd() - 0.5) * 64.0, 6.0, (rnd() - 0.5) * 64.0}; }
        while (std::fabs(T.x) < 7 && T.z > -5 && T.z < 20);        // keep the plaza (and the crowd) free
        shots.push_back({T, it, false});
    }
    for (const Shot& s : shots) {
        Vec3 T = s.T;
        Vec3 L = {RIVAL_CENTRE.x + (rnd() - 0.5) * 14, 3.0, (rnd() - 0.5) * 24};
        Missile m;
        m.p0 = L;
        m.p1 = L + Vec3{30, 85 + rnd() * 15, (T.z - L.z) * 0.25};
        m.p2 = T + Vec3{-55, 80 + rnd() * 15, (L.z - T.z) * 0.15};
        m.p3 = T;
        m.flightT = 30.0; m.launchT = s.impactT - m.flightT;
        m.big = s.big;
        w.missiles.push_back(m);
    }
    for (Building& b : w.buildings) {
        if (b.city != 0 || b.sh) continue;                          // CHANGED: statehouse chunks do not collapse, they fly
        double cx = (b.box.x0 + b.box.x1) / 2, cz = (b.box.z0 + b.box.z1) / 2;
        for (const Missile& m : w.missiles) {
            double d = std::hypot(cx - m.p3.x, cz - m.p3.z);
            if (d < 11.0) b.collapseT = std::min(b.collapseT, m.launchT + m.flightT + 0.12 * d);
        }
        b.collapseDur = 3 + rnd() * 2;
        b.shearX = (rnd() - 0.5) * 1.2; b.shearZ = (rnd() - 0.5) * 1.2;
    }
    // City B continues its assault; City A is the city that collapses.
    // City B is not counter-attacked or destroyed in this storyline.
    const Color carCols[4] = {{200, 50, 50}, {240, 240, 240}, {60, 90, 190}, {230, 190, 40}};
    for (int i = 0; i < 36; ++i) {
        Car c;
        c.road = 1 + static_cast<int>(rnd() * 8); c.alongX = rnd() < 0.5;
        c.lane = rnd() < 0.5 ? -0.4 : 0.4; c.speed = 3 + rnd() * 4; c.offset = rnd() * 2 * HALF;
        c.dir = rnd() < 0.5 ? 1 : -1; c.startT = 60 + rnd() * 60; c.endT = 238 + rnd() * 10;
        c.col = carCols[static_cast<int>(rnd() * 4)];
        w.cars.push_back(c);
    }
}

// ======================================================================= Keyframes
// CHANGED: a key with cut = true is a hard cut: the camera holds the previous key until this time.
struct CamKey { double t; Vec3 eye, target; bool cut = false; };
const CamKey CAM[] = {
    {  0, {70, 48, 70},  {0, 4, 0}},
    {120, {95, 50, 20},  {0, 6, 0}},
    {180, {20, 55, 95},  {0, 6, 0}},
    {188, {-62, 10, 23}, {-62, 1.5, 0}},       // close view of the two arguing leaders
    // HARD CUT 1: hold a clear, front-facing wide shot of City A's cabinet room.
    // The room's +Z side is open; the camera looks through that opening at the table.
    {198, {0, 6.5, 29}, {0, 2.1, -0.5}, true},
    // HARD CUT 2: switch to the matching front-facing wide shot inside City B.
    {212, {-125, 6.5, 29}, {-125, 2.1, -0.5}, true},
    // HARD CUT 3: return to the exterior attack-preparation view, then flow into battle.
    {226, {-95, 28, 40}, {-125, 4, 0}, true},
    {240, {60, 40, 45},  {-55, 15, 0}},
    {258, {50, 34, 62},  {-6, 9, 0}},
    {261, {0, 3.5, 22},  {0, 6.0, 0}, true},
    {266.5, {0, 4.5, 18.5}, {0, 6.5, 0}},
    {273, {-3.5, 7, 22}, {0, 4, 0}},
    {274, {0, 9, 46}, {0, 3, 26}, true}, // City A leader rallies the people
    {290, {0, 7, 39}, {0, 3, 26}},
    {300, {0, 35, 48}, {0, 3, 0}, true}, // City A resistance is overwhelmed
    {318, {0, 28, 40}, {0, 2, 0}},
    {325, {0, -2.2, 1.0}, {0, -5.0, 11.0}, true}, // camera is inside the ground cutaway, looking along the underground tunnel
    {340, {0, -2.2, 1.0}, {0, -5.0, 11.0}},
    {350, {0, -2.2, 1.0}, {0, -5.0, 11.0}},
};
void cameraAt(double t, Vec3& eye, Vec3& target) {
    // Dedicated interior camera shots. These overrides are intentional:
    // do not interpolate from the exterior city camera during either meeting.
    if (t >= 198.0 && t < 212.0) {
        eye = {0.0, 6.0, 28.0};
        target = {0.0, 2.0, -0.5};
        return;
    }
    if (t >= 212.0 && t < 226.0) {
        eye = {RIVAL_CENTRE.x, 6.0, 28.0};
        target = {RIVAL_CENTRE.x, 2.0, -0.5};
        return;
    }
    const int n = sizeof(CAM) / sizeof(CAM[0]);
    if (t <= CAM[0].t) { eye = CAM[0].eye; target = CAM[0].target; return; }
    for (int i = 0; i + 1 < n; ++i) {
        if (t <= CAM[i + 1].t) {
            if (CAM[i + 1].cut) { eye = CAM[i].eye; target = CAM[i].target; return; }   // NEW: hold until the cut
            double u = smoothstep((t - CAM[i].t) / (CAM[i + 1].t - CAM[i].t));
            eye = lerp(CAM[i].eye, CAM[i + 1].eye, u); target = lerp(CAM[i].target, CAM[i + 1].target, u);
            return;
        }
    }
    eye = CAM[n - 1].eye; target = CAM[n - 1].target;
}

struct SkyKey { double t; Color top, horizon; double bright; };
const SkyKey SKY[] = {
    {  0, {110, 140, 200}, {250, 190, 150}, 0.85},
    { 40, { 60, 115, 200}, {170, 205, 235}, 1.00},
    {180, { 60, 115, 200}, {170, 205, 235}, 1.00},
    {210, { 70,  70,  95}, {150, 110, 100}, 0.80},
    {240, { 50,  40,  50}, {170,  70,  40}, 0.62},
    {300, { 45,  40,  50}, {120,  60,  45}, 0.55},
    {320, { 90, 100, 150}, {250, 170, 120}, 0.85},
    {331, { 90, 100, 150}, {250, 170, 120}, 0.85},
};
SkyKey skyAt(double t) {
    const int n = sizeof(SKY) / sizeof(SKY[0]);
    for (int i = 0; i + 1 < n; ++i) {
        if (t <= SKY[i + 1].t) {
            double u = clamp01((t - SKY[i].t) / (SKY[i + 1].t - SKY[i].t));
            return {t, lerpColor(SKY[i].top, SKY[i + 1].top, u), lerpColor(SKY[i].horizon, SKY[i + 1].horizon, u),
                    lerp(SKY[i].bright, SKY[i + 1].bright, u)};
        }
    }
    return SKY[n - 1];
}


// ======================================================================= OpenGL helpers
// Our row-major Mat4 -> OpenGL's column-major array
void loadMatrix(const Mat4& m) {
    double a[16];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) a[j * 4 + i] = m.m[i][j];
    glLoadMatrixd(a);
}

static GLuint g_cubeList = 0, g_softTex = 0;

// Unit cube (0..1 on every axis) with outward normals, compiled into a display list.
// Every building is this one cube moved by the matrix stack (translate, shear, scale).
void buildCubeList() {
    g_cubeList = glGenLists(1);
    glNewList(g_cubeList, GL_COMPILE);
    glBegin(GL_QUADS);
    glNormal3f( 1, 0, 0); glVertex3f(1, 0, 0); glVertex3f(1, 1, 0); glVertex3f(1, 1, 1); glVertex3f(1, 0, 1);
    glNormal3f(-1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1); glVertex3f(0, 1, 1); glVertex3f(0, 1, 0);
    glNormal3f( 0, 1, 0); glVertex3f(0, 1, 0); glVertex3f(0, 1, 1); glVertex3f(1, 1, 1); glVertex3f(1, 1, 0);
    glNormal3f( 0, 0, 1); glVertex3f(0, 0, 1); glVertex3f(1, 0, 1); glVertex3f(1, 1, 1); glVertex3f(0, 1, 1);
    glNormal3f( 0, 0,-1); glVertex3f(0, 0, 0); glVertex3f(0, 1, 0); glVertex3f(1, 1, 0); glVertex3f(1, 0, 0);
    glEnd();
    glEndList();
}

// Soft round sprite (radial alpha falloff) used for smoke, fire and trails
void buildSoftTexture() {
    const int S = 64;
    std::vector<unsigned char> px(S * S * 4);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            double dx = (x + 0.5) / S * 2 - 1, dy = (y + 0.5) / S * 2 - 1;
            double d = std::sqrt(dx * dx + dy * dy);
            double a = d >= 1 ? 0 : std::pow(1 - d, 1.4);
            unsigned char* p = &px[(y * S + x) * 4];
            p[0] = p[1] = p[2] = 255; p[3] = static_cast<unsigned char>(255 * a);
        }
    glGenTextures(1, &g_softTex);
    glBindTexture(GL_TEXTURE_2D, g_softTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, S, S, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
}

void colorOf(Color c) { glColor3ub(c.r, c.g, c.b); }

// Flat horizontal quad at height y, facing up
void groundQuad(double x0, double z0, double x1, double z1, double y, Color c) {
    if (x1 <= x0) return;
    colorOf(c);
    glNormal3f(0, 1, 0);
    glBegin(GL_QUADS);
    glVertex3d(x0, y, z0); glVertex3d(x0, y, z1); glVertex3d(x1, y, z1); glVertex3d(x1, y, z0);
    glEnd();
}

// A box = the unit cube transformed by the OpenGL matrix stack.
// The collapse "lean" is a shear matrix multiplied in with glMultMatrixd.
// CHANGED: the box goes from y0 to h (y0 = 0 for ordinary buildings).
void drawBoxGL(const Box& b) {
    double h = b.h - b.y0;
    if (h < 0.03) return;
    glPushMatrix();
    glTranslated(b.x0, b.y0, b.z0);
    double shear[16] = {1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1};
    shear[4] = b.sx / h;                       // x' = x + (sx/h) * y
    shear[6] = b.sz / h;                       // z' = z + (sz/h) * y
    glMultMatrixd(shear);
    glScaled(b.x1 - b.x0, h, b.z1 - b.z0);
    colorOf(b.base);
    glCallList(g_cubeList);
    glPopMatrix();
}

// NEW: the same cube centred on the origin, so that rotations turn it around its middle.
// The caller sets up translate * rotate * scale on the matrix stack first.
void drawCentredCube(Color c) {
    colorOf(c);
    glPushMatrix();
    glTranslated(-0.5, -0.5, -0.5);
    glCallList(g_cubeList);
    glPopMatrix();
}

// Low-poly person built from transformed cubes. The same model is reused for
// leaders, ministers, soldiers and the underground resistance.
void drawPersonGL(double x, double z, Color shirt, double scale = 1.0, bool armsRaised = false) {
    const Color skin{225, 178, 145}, trousers{48, 52, 62};
    glPushMatrix(); glTranslated(x, 0, z); glScaled(scale, scale, scale);
    glPushMatrix(); glTranslated(0, 1.05, 0); glScaled(0.52, 0.92, 0.34); drawCentredCube(shirt); glPopMatrix();
    glPushMatrix(); glTranslated(0, 1.78, 0); glScaled(0.34, 0.34, 0.34); drawCentredCube(skin); glPopMatrix();
    for (int side = -1; side <= 1; side += 2) {
        glPushMatrix(); glTranslated(side * 0.15, 0.36, 0); glScaled(0.17, 0.68, 0.20); drawCentredCube(trousers); glPopMatrix();
        glPushMatrix(); glTranslated(side * 0.36, armsRaised ? 1.55 : 1.12, 0);
        glRotated(armsRaised ? -35.0 : 12.0, 0, 0, 1);
        glScaled(0.16, 0.72, 0.18); drawCentredCube(shirt); glPopMatrix();
    }
    glPopMatrix();
}

void drawStoryScenes(double t) {
    const Color leaderA{35, 75, 155}, leaderB{155, 48, 42}, ministerA{70, 110, 175}, ministerB{175, 90, 70};
    // Peace meeting in a neutral plaza halfway between the cities.
    if (t >= 185 && t < 198) {
        groundQuad(-68, -5, -56, 5, 0.35, Color{165, 155, 135});
        drawBoxGL(Box{-65, -1.2, -60, 1.2, 1.15, Color{95, 65, 42}}); // negotiation table
        // Both leaders visibly argue: alternating raised arms and forward gestures.
        const bool gestureA = (std::sin(t * 3.8) > -0.15);
        const bool gestureB = (std::sin(t * 3.8 + 2.2) > -0.15);
        drawPersonGL(-67.0 + 0.25 * std::sin(t * 2.0), 0, leaderA, 1.45, gestureA);
        drawPersonGL(-58.0 - 0.25 * std::sin(t * 2.0), 0, leaderB, 1.45, gestureB);
        // Two ministers on each side react to the heated exchange.
        drawPersonGL(-68, -3.2, ministerA, 0.95, gestureB);
        drawPersonGL(-66, -3.2, ministerA, 0.95, false);
        drawPersonGL(-59, -3.2, ministerB, 0.95, gestureA);
        drawPersonGL(-57, -3.2, ministerB, 0.95, false);
    }
    // Separate cabinet meetings are shown as dedicated camera shots INSIDE each statehouse.
    // The open +Z wall gives a clear frontal view of the leader, ministers and table.
    auto drawCabinetRoom = [&](double cx, Color wall, Color leader, Color ministers) {
        drawBoxGL(Box{cx-11, -11, cx+11, 11, 0.35, Color{105, 92, 75}}); // raised chamber floor
        drawBoxGL(Box{cx-11, -11, cx-10.2, 11, 7.0, wall});
        drawBoxGL(Box{cx+10.2, -11, cx+11, 11, 7.0, wall});
        drawBoxGL(Box{cx-11, -11, cx+11, -10.2, 7.0, wall}); // far wall; camera-facing +Z entrance is OPEN
        // Statehouse columns and a raised rear dais
        for (int side : {-1, 1}) for (int z : {-1, 1})
            drawBoxGL(Box{cx+side*9.2-0.45, z*9.2-0.45, cx+side*9.2+0.45, z*9.2+0.45, 6.5, Color{190, 177, 145}});
        drawBoxGL(Box{cx-4.8, 5.0, cx+4.8, 7.8, 0.65, Color{115, 82, 52}});
        // A large, unmistakable cabinet table sits in the centre of the room.
        drawBoxGL(Box{cx-4.4, -1.35, cx+4.4, 1.35, 1.25, Color{75, 49, 32}});
        // The leader stands at the far end, facing the ministers and addressing them.
        const bool leaderSpeaking = (std::sin(t * 2.7) > -0.25);
        drawPersonGL(cx, 4.0, leader, 1.55, leaderSpeaking);
        // Four ministers are spaced along the near side and visibly respond in turn.
        for (int i=0;i<4;++i) {
            const bool responding = (std::sin(t * 2.1 + i * 1.45) > 0.45);
            drawPersonGL(cx-6.0+i*4.0, -4.8, ministers, 1.12, responding);
        }
        // Papers/map on the tabletop make this read as a cabinet briefing, not a plaza.
        groundQuad(cx-2.1, -0.55, cx+2.1, 0.55, 1.91, Color{190, 185, 155});
        groundQuad(cx-1.5, -0.35, cx+1.5, 0.35, 1.93, Color{65, 105, 75});
    };
    if (t >= 198 && t < 212) drawCabinetRoom(0, Color{105, 125, 155}, leaderA, ministerA);
    if (t >= 212 && t < 226) drawCabinetRoom(RIVAL_CENTRE.x, Color{150, 92, 78}, leaderB, ministerB);
    // Soldiers and missile racks make the attack preparation legible.
    if (t >= 226 && t < 240) {
        for (int i = 0; i < 4; ++i) {
            drawPersonGL(RIVAL_CENTRE.x - 5 + i * 3.0, 12, ministerB, 0.95);
            drawBoxGL(Box{RIVAL_CENTRE.x - 6 + i * 3.0, 16, RIVAL_CENTRE.x - 5.1 + i * 3.0, 17, 2.2, Color{105, 110, 120}});
        }
    }
    // City B occupies the ruins after City A has fallen.
    if (t >= 318) {
        drawBoxGL(Box{-0.12, -0.12, 0.12, 0.12, 12.0, Color{185, 185, 190}});
        drawBoxGL(Box{0.12, -0.12, 4.2, 0.12, 9.0, Color{35, 85, 175}});
        drawPersonGL(-5, 4, leaderB, 1.1);
        drawPersonGL(5, 4, leaderB, 1.1);
    }
    // Underground sewer rebellion: the tunnel floor and rebels are BELOW y=0.
    // The terrain renderer cuts a rectangular opening above this chamber so the camera
    // can look down the access shaft without the city ground plane hiding the scene.
    if (t >= 325) {
        const double floorY = -8.5;
        const double wallH = 7.8; // walls rise to y=-0.7, still below street level
        const Color concrete{78, 82, 88}, darkFloor{28, 34, 39}, pipeColor{105, 112, 118};
        groundQuad(-9, 3, 9, 19, floorY, darkFloor);
        // Three retaining walls; the near end (z=3) is open as the camera entrance.
        drawBoxGL(Box{-9, 3, -8.35, 19, wallH, concrete, 0, 0, floorY});
        drawBoxGL(Box{8.35, 3, 9, 19, wallH, concrete, 0, 0, floorY});
        drawBoxGL(Box{-9, 18.35, 9, 19, wallH, concrete, 0, 0, floorY});
        // Street-level concrete rim clearly communicates that this is a shaft below the city.
        drawBoxGL(Box{-10.0, 1.8, -9.0, 20.0, 0.45, Color{115, 119, 123}, 0, 0, -0.55});
        drawBoxGL(Box{9.0, 1.8, 10.0, 20.0, 0.45, Color{115, 119, 123}, 0, 0, -0.55});
        drawBoxGL(Box{-10.0, 19.0, 10.0, 20.0, 0.45, Color{115, 119, 123}, 0, 0, -0.55});
        groundQuad(-8.2, 3.2, -7.9, 18.2, floorY + 0.02, Color{190, 150, 45});
        groundQuad(7.9, 3.2, 8.2, 18.2, floorY + 0.02, Color{190, 150, 45});
        groundQuad(-0.8, 3.3, 0.8, 18.2, floorY + 0.03, Color{12, 18, 23});
        // Overhead pipes remain underground, attached just below street level.
        drawBoxGL(Box{-8.0, 5.0, -7.55, 17.0, 0.28, pipeColor, 0, 0, -1.0});
        drawBoxGL(Box{7.55, 5.0, 8.0, 17.0, 0.28, pipeColor, 0, 0, -1.0});
        glPushMatrix(); glTranslated(0, floorY, 0);
        drawBoxGL(Box{-2.3, 9.0, 2.3, 12.0, 1.35, Color{92, 66, 43}, 0, 0, 0.0});
        groundQuad(-1.8, 9.5, 1.8, 11.5, 1.37, Color{55, 105, 65});
        for (int i = 0; i < 5; ++i)
            drawPersonGL(-6.0 + i * 3.0, 7.0 + (i % 2) * 1.0, Color{65, 85, 72}, 0.95, i == 2);
        for (int i = 0; i < 3; ++i)
            drawPersonGL(-3.0 + i * 3.0, 14.0, Color{75, 90, 78}, 0.9, false);
        glPopMatrix();
    }
}

// Missile: a box body and a pyramid nose, rotated so that +Z follows the Bezier tangent
void drawMissileGL(const Missile& m, double t) {
    if (t < m.launchT || t > m.launchT + m.flightT) return;
    double u = (t - m.launchT) / m.flightT;
    Vec3 head = bezier(m.p0, m.p1, m.p2, m.p3, u);
    Vec3 dir = normalize(bezierTangent(m.p0, m.p1, m.p2, m.p3, u));
    double yaw = std::atan2(dir.x, dir.z) * 180.0 / PI, pitch = -std::asin(dir.y) * 180.0 / PI;
    glPushMatrix();
    glTranslated(head.x - dir.x * 0.9, head.y - dir.y * 0.9, head.z - dir.z * 0.9);
    glRotated(yaw, 0, 1, 0);
    glRotated(pitch, 1, 0, 0);
    glPushMatrix();
    glTranslated(-0.25, -0.25, -1.5);
    glScaled(0.5, 0.5, 2.1);
    glColor3ub(235, 235, 240);
    glCallList(g_cubeList);
    glPopMatrix();
    glColor3ub(220, 50, 50);
    glBegin(GL_TRIANGLES);
    glNormal3f( 0.9f, 0, 0.4f); glVertex3f( 0.25f,-0.25f, 0.6f); glVertex3f( 0.25f, 0.25f, 0.6f); glVertex3f(0, 0, 1.5f);
    glNormal3f(-0.9f, 0, 0.4f); glVertex3f(-0.25f, 0.25f, 0.6f); glVertex3f(-0.25f,-0.25f, 0.6f); glVertex3f(0, 0, 1.5f);
    glNormal3f(0, 0.9f, 0.4f);  glVertex3f( 0.25f, 0.25f, 0.6f); glVertex3f(-0.25f, 0.25f, 0.6f); glVertex3f(0, 0, 1.5f);
    glNormal3f(0,-0.9f, 0.4f);  glVertex3f(-0.25f,-0.25f, 0.6f); glVertex3f( 0.25f,-0.25f, 0.6f); glVertex3f(0, 0, 1.5f);
    glEnd();
    glPopMatrix();
}

// ======================================================================= Particles (billboards)
struct Part { Vec3 p; double r; float cr, cg, cb, a; };
struct Ring { Vec3 c; double R; float a; };

void addPart(std::vector<Part>& v, Vec3 p, double r, Color c, double a) {
    v.push_back({p, r, c.r / 255.f, c.g / 255.f, c.b / 255.f, static_cast<float>(a)});
}

// Same stateless particle formulas as Checkpoint 2: any time t can be rendered on its own
void collectEffects(const World& w, double t, std::vector<Part>& smoke, std::vector<Part>& fire, std::vector<Ring>& rings) {
    for (size_t i = 0; i < w.buildings.size(); ++i) {               // smoke from every collapsed building (and the statehouse ruins)
        const Building& b = w.buildings[i];
        if (b.collapseT > 1e8 || t < b.collapseT) continue;
        double cx = (b.box.x0 + b.box.x1) / 2, cz = (b.box.z0 + b.box.z1) / 2, hr = 0.12 * b.box.h;
        for (int k = 0; k < 5; ++k) {
            double tt = t - b.collapseT - 0.5 - k * 1.3;
            if (tt < 0 || tt > 120) continue;
            double age = std::fmod(tt, 7.0), f = age / 7.0;
            Vec3 p = {cx + 0.8 * age * std::cos(k + i), hr + age * 2.4, cz + 0.8 * age * std::sin(k + i)};
            addPart(smoke, p, 0.9 + 2.4 * f, Color{80, 77, 80}, 0.45 * (1 - f) * clamp01(tt));
        }
    }
    for (size_t i = 0; i < w.missiles.size(); ++i) {
        const Missile& m = w.missiles[i];
        if (t >= m.launchT && t <= m.launchT + m.flightT) {
            double u = (t - m.launchT) / m.flightT;
            for (int k = 14; k >= 1; --k) {                           // exhaust trail sampled along the curve
                double uk = std::max(0.0, u - k * 0.006);
                addPart(smoke, bezier(m.p0, m.p1, m.p2, m.p3, uk), 0.6 + 0.1 * k, Color{210, 210, 215}, 0.45 * (1 - k / 15.0));
            }
            Vec3 dir = normalize(bezierTangent(m.p0, m.p1, m.p2, m.p3, u));
            addPart(fire, bezier(m.p0, m.p1, m.p2, m.p3, u) - dir * 2.4, 1.3, Color{255, 170, 60}, 0.9);   // engine flame
        }
        double age = t - (m.launchT + m.flightT);
        if (age < 0 || age > 3.0) continue;
        Vec3 P = {m.p3.x, 3.0, m.p3.z};
        int id = static_cast<int>(i);
        const double sc = m.big ? 1.3 : 1.0;                           // NEW: the statehouse strike is bigger
        const double ps = m.big ? 1.6 : 1.0;
        if (age < 1.5) rings.push_back({P, 16.0 * sc * age / 1.5, static_cast<float>(1 - age / 1.5)});
        if (age < 1.4) {
            double k = std::sin(PI * age / 1.4);
            addPart(fire, {P.x, P.y + 2 + age * 3, P.z}, 6.0 * sc * k, Color{255, 150, 40}, 0.7);
            addPart(fire, {P.x, P.y + 2 + age * 3, P.z}, 3.6 * sc * k, Color{255, 240, 150}, 0.9);
        }
        for (int k = 0; k < 40; ++k) {
            double life = 1.5 + 1.5 * hash01(id * 100 + k, 4);
            if (age > life) continue;
            double th = 2 * PI * hash01(id * 100 + k, 1), up = 0.3 + 0.7 * hash01(id * 100 + k, 2);
            double sp = (6 + 14 * hash01(id * 100 + k, 3)) * (m.big ? 1.4 : 1.0);
            Vec3 v = {sp * std::cos(th) * (1 - up * 0.5), sp * up * 1.2 + 4, sp * std::sin(th) * (1 - up * 0.5)};
            Vec3 p = P + v * age;
            p.y = std::max(0.0, p.y - 0.5 * 9.8 * age * age);
            double f = age / life;
            Color c = f < 0.5 ? lerpColor(Color{255, 230, 90}, Color{255, 120, 30}, f * 2)
                              : lerpColor(Color{255, 120, 30}, Color{90, 50, 40}, (f - 0.5) * 2);
            addPart(fire, p, 1.1 * ps * (1 - 0.5 * f), c, 1.0 - f);
        }
    }
}

void drawParts(const std::vector<Part>& v, Vec3 right, Vec3 up) {
    glBegin(GL_QUADS);
    for (const Part& q : v) {
        double s = q.r * 1.4;
        glColor4f(q.cr, q.cg, q.cb, q.a);
        Vec3 a = q.p - right * s - up * s, b = q.p + right * s - up * s;
        Vec3 c = q.p + right * s + up * s, d = q.p - right * s + up * s;
        glTexCoord2f(0, 0); glVertex3d(a.x, a.y, a.z);
        glTexCoord2f(1, 0); glVertex3d(b.x, b.y, b.z);
        glTexCoord2f(1, 1); glVertex3d(c.x, c.y, c.z);
        glTexCoord2f(0, 1); glVertex3d(d.x, d.y, d.z);
    }
    glEnd();
}

// ======================================================================= Frame renderer
struct FreeCam { bool on = false; Vec3 eye{0, 40, 80}; double yaw = 0, pitch = -0.4; };

Vec3 freeDir(const FreeCam& f) { return {std::cos(f.pitch) * std::sin(f.yaw), std::sin(f.pitch), -std::cos(f.pitch) * std::cos(f.yaw)}; }

void screenQuad(double w, double h, float r, float g, float b, float a) {
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS); glVertex2d(0, 0); glVertex2d(w, 0); glVertex2d(w, h); glVertex2d(0, h); glEnd();
}

// CHANGED: the argument is FILM time; the first line converts it to STORY time (slow motion).
void renderGL(const World& w, double filmT, int fbW, int fbH, const FreeCam& fc) {
    const double t = storyTime(filmT);
    SkyKey sky = skyAt(t);
    const double bright = sky.bright;
    glViewport(0, 0, fbW, fbH);
    glClearColor(sky.horizon.r / 255.f, sky.horizon.g / 255.f, sky.horizon.b / 255.f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ---- 2D pass: sky gradient and sun (screen-space orthographic projection)
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG); glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND); glDisable(GL_TEXTURE_2D);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, fbW, fbH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glBegin(GL_QUADS);
    glColor3ub(sky.top.r, sky.top.g, sky.top.b);         glVertex2d(0, 0);   glVertex2d(fbW, 0);
    glColor3ub(sky.horizon.r, sky.horizon.g, sky.horizon.b); glVertex2d(fbW, fbH); glVertex2d(0, fbH);
    glEnd();
    {
        double u = -1;                                      // sun on the same cubic Bezier arc as Checkpoint 2
        if (t <= 205) u = t / 200.0; else if (t >= 300) u = (t - 300) / 30.0 * 0.15;
        if (u >= 0) {
            Vec3 p = bezier({60, 380, 0}, {240, -140, 0}, {720, -140, 0}, {900, 380, 0}, clamp01(u));
            double cx = p.x / REF_W * fbW, cy = p.y / REF_H * fbH, r = 32.0 / REF_H * fbH;
            glColor3ub(255, 215, 110);
            glBegin(GL_TRIANGLE_FAN);
            glVertex2d(cx, cy);
            for (int i = 0; i <= 40; ++i) glVertex2d(cx + r * std::cos(2 * PI * i / 40), cy + r * std::sin(2 * PI * i / 40));
            glEnd();
        }
    }

    // Cabinet meetings are rendered as isolated interior shots. Do not let
    // free-camera mode replace their dedicated room camera.
    const bool cabinetInterior = (t >= 198.0 && t < 226.0);

    // ---- camera: scripted keyframes (or the free camera), plus a shake after each impact
    Vec3 eye, target; cameraAt(t, eye, target);
    if (fc.on && !cabinetInterior) { eye = fc.eye; target = fc.eye + freeDir(fc); }
    else {
        Vec3 shake = {0, 0, 0};
        for (size_t i = 0; i < w.missiles.size(); ++i) {
            double age = t - (w.missiles[i].launchT + w.missiles[i].flightT);
            double dur = w.missiles[i].big ? 2.5 : 1.2;            // CHANGED: the statehouse strike shakes harder and longer
            if (age >= 0 && age < dur) {
                double a = (1 - age / dur) * (1 - age / dur) * (w.missiles[i].big ? 2.5 : 0.9);
                shake = shake + Vec3{a * std::sin(t * 47 + i), a * std::cos(t * 61 + 2 * i) * 0.6, a * std::sin(t * 53 + 3 * i)};
            }
        }
        eye = eye + shake; target = target + shake;
    }
    Vec3 fwd = normalize(target - eye), right = normalize(cross(fwd, {0, 1, 0})), up = cross(right, fwd);

    // ---- 3D pass: our own Mat4 matrices go straight into OpenGL
    glMatrixMode(GL_PROJECTION);
    loadMatrix(Mat4::perspective(FOV_Y, static_cast<double>(fbW) / fbH, 1.0, 1500.0));
    glMatrixMode(GL_MODELVIEW);
    loadMatrix(lookAt(eye, target, {0, 1, 0}));

    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LEQUAL); glDepthMask(GL_TRUE);
    glEnable(GL_NORMALIZE);

    // fog fades distant objects (the rival city) into the horizon colour
    glEnable(GL_FOG);
    GLfloat fogCol[4] = {sky.horizon.r / 255.f, sky.horizon.g / 255.f, sky.horizon.b / 255.f, 1};
    glFogi(GL_FOG_MODE, GL_LINEAR); glFogfv(GL_FOG_COLOR, fogCol);
    glFogf(GL_FOG_START, 260.0f); glFogf(GL_FOG_END, 1500.0f);

    // lighting: one directional "sun" light that follows the sky brightness and colour
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    Color tint = lerpColor(Color{255, 255, 255}, sky.horizon, 0.35);
    GLfloat amb[4] = {0.30f * (float)bright, 0.30f * (float)bright, 0.32f * (float)bright, 1};
    GLfloat dif[4] = {0.75f * (float)bright * tint.r / 255.f, 0.75f * (float)bright * tint.g / 255.f, 0.75f * (float)bright * tint.b / 255.f, 1};
    GLfloat zero[4] = {0, 0, 0, 1};
    GLfloat lpos[4] = {0.45f, 0.8f, 0.35f, 0};          // w = 0: directional
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, lpos); glLightfv(GL_LIGHT0, GL_DIFFUSE, dif); glLightfv(GL_LIGHT0, GL_AMBIENT, zero);

    // explosions light up the nearby buildings: up to 4 point lights at the newest impacts
    int nl = 0;
    for (size_t i = 0; i < w.missiles.size() && nl < 4; ++i) {
        double age = t - (w.missiles[i].launchT + w.missiles[i].flightT);
        GLenum L = GL_LIGHT1 + nl;
        if (age >= 0 && age < 2.0) {
            double k = (1 - age / 2.0); k *= k * 3.0;
            if (w.missiles[i].big) k *= 2.5;                    // NEW: the statehouse strike is much brighter
            GLfloat pos[4] = {(GLfloat)w.missiles[i].p3.x, 8.0f, (GLfloat)w.missiles[i].p3.z, 1};
            GLfloat col[4] = {(GLfloat)(1.0 * k), (GLfloat)(0.55 * k), (GLfloat)(0.2 * k), 1};
            glEnable(L); glLightfv(L, GL_POSITION, pos); glLightfv(L, GL_DIFFUSE, col); glLightfv(L, GL_AMBIENT, zero);
            glLightf(L, GL_CONSTANT_ATTENUATION, 1.0f); glLightf(L, GL_LINEAR_ATTENUATION, 0.02f);
            glLightf(L, GL_QUADRATIC_ATTENUATION, 0.004f);
            ++nl;
        }
    }
    for (int i = nl; i < 4; ++i) glDisable(GL_LIGHT1 + i);

    // During cabinet scenes, suppress the entire exterior world. Otherwise
    // city buildings/statehouse geometry can sit in front of the room and
    // make the interior impossible to see from the dedicated camera.
    if (!cabinetInterior) {
    // ---- ground layers (heights differ slightly so the layers do not flicker)
    glDisable(GL_CULL_FACE);
    if (t < 325) {
        groundQuad(-600, -600, 600, 600, 0.0, Color{105, 150, 90});                   // land
    } else {
        // Keep the city ground intact except for the access opening over the underground sewer.
        groundQuad(-600, -600, -10.0, 600, 0.0, Color{105, 150, 90});
        groundQuad(10.0, -600, 600, 600, 0.0, Color{105, 150, 90});
        groundQuad(-10.0, -600, 10.0, 2.0, 0.0, Color{105, 150, 90});
        groundQuad(-10.0, 20.0, 10.0, 600, 0.0, Color{105, 150, 90});
    }
    colorOf(Color{45, 105, 160}); glNormal3f(0, 1, 0);                                 // sea: strip with a wavy shoreline
    glBegin(GL_TRIANGLE_STRIP);
    for (double z = -600; z <= 600; z += 10) {
        glVertex3d(-46 + 3.0 * std::sin(z * 0.15), 0.05, z);
        glVertex3d(-800, 0.05, z);
    }
    glEnd();
    const bool rivalVisible = t >= 170;                                                // rival city appears in the tension phase
    const double ri = N_RIVAL * PITCH / 2 + 8;
    if (rivalVisible) groundQuad(RIVAL_CENTRE.x - ri, -ri, RIVAL_CENTRE.x + ri, ri, 0.08, Color{95, 120, 85});

    double reveal = easeOutCubic((t - 2) / 18.0);                                       // roads draw in west to east
    double xr = -HALF - 1 + reveal * (2 * HALF + 2);
    groundQuad(-HALF - 1, -HALF - 1, xr, HALF + 1, 0.10, Color{58, 58, 64});
    for (const Block& b : w.blocks) {
        if (b.city == 0 && b.x0 + BLOCK > xr) continue;
        if (b.city == 1 && !rivalVisible) continue;
        Color c = b.park ? Color{85, 160, 90} : (b.city == 0 ? Color{150, 150, 148} : Color{140, 120, 115});
        if (b.plaza) c = Color{190, 185, 172};                                          // NEW: plaza paving
        groundQuad(b.x0, b.z0, b.x0 + BLOCK, b.z0 + BLOCK, 0.20, c);
    }
    for (int k = 0; k <= N_MAIN; ++k) {                                                 // dashed centre lines
        double c = -HALF + k * PITCH;
        for (double s = -HALF; s < HALF; s += 4.0) {
            if (c <= xr) groundQuad(c - 0.1, s, c + 0.1, s + 2, 0.30, Color{225, 200, 80});
            if (s + 2 <= xr) groundQuad(s, c - 0.1, s + 2, c + 0.1, 0.30, Color{225, 200, 80});
        }
    }

    // ---- solid objects: the depth buffer replaces the painter's algorithm
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK); glFrontFace(GL_CCW);
    for (const Building& b : w.buildings) {
        double grow = t <= b.growStart ? 0 : easeOutCubic((t - b.growStart) / b.growDur);
        if (grow <= 0) continue;
        if (b.sh) {                                              // NEW: statehouse chunk
            if (t < SH_T) {
                Box x = b.box; x.y0 = b.box.y0 * grow; x.h = b.box.h * grow;      // scale animation, bottom-up
                drawBoxGL(x);
            } else {
                double s = t - SH_T;                             // story time is already slowed by storyTime()
                double tc = std::min(s, b.tland);                // chunk freezes when it lies on the ground
                Vec3 c = {(b.box.x0 + b.box.x1) / 2, (b.box.y0 + b.box.h) / 2, (b.box.z0 + b.box.z1) / 2};
                Vec3 p = c + b.vel * tc + Vec3{0, -4.9 * tc * tc, 0};            // translation with gravity
                glPushMatrix();                                  // M = T * Rx * Ry * Rz * S
                glTranslated(p.x, p.y, p.z);
                glRotated(b.spin.x * tc * 180.0 / PI, 1, 0, 0);
                glRotated(b.spin.y * tc * 180.0 / PI, 0, 1, 0);
                glRotated(b.spin.z * tc * 180.0 / PI, 0, 0, 1);
                glScaled(b.box.x1 - b.box.x0, b.box.h - b.box.y0, b.box.z1 - b.box.z0);
                drawCentredCube(lerpColor(b.box.base, Color{55, 50, 50}, smoothstep(s / 8.0)));
                glPopMatrix();
            }
            continue;
        }
        double ce = t < b.collapseT ? 0 : easeInCubic((t - b.collapseT) / b.collapseDur);
        Box x = b.box;
        x.h = b.box.h * grow * (1 - 0.88 * ce);                 // scale animation (growth / collapse)
        x.sx = b.shearX * ce * b.box.h * grow * 0.5;            // shear animation (lean)
        x.sz = b.shearZ * ce * b.box.h * grow * 0.5;
        x.base = lerpColor(b.box.base, Color{55, 50, 50}, ce);
        drawBoxGL(x);
    }
    for (const Car& c : w.cars) {                               // translation along the roads
        if (t < c.startT || t > c.endT) continue;
        double p = std::fmod(c.offset + t * c.speed, 2 * HALF);
        if (c.dir < 0) p = 2 * HALF - p;
        double along = -HALF + p, centre = -HALF + c.road * PITCH + c.lane;
        Box car = c.alongX ? Box{along, centre - 0.35, along + 1.4, centre + 0.35, 0.6, c.col}
                           : Box{centre - 0.35, along, centre + 0.35, along + 1.4, 0.6, c.col};
        drawBoxGL(car);
    }
    if (t > 305) {                                              // a new building starts to rise at dawn (on the statehouse ruins)
        double g = easeOutCubic((t - 305) / 20.0);
        drawBoxGL(Box{-1.4, -1.4, 1.4, 1.4, 9.0 * g, Color{200, 225, 255}});
    }

    // NEW: the leader on his podium and the crowd (hierarchy with the OpenGL matrix stack)
    if (t >= RALLY_T && t < 306) {
        glPushMatrix(); glTranslated(0, 0, 20.0); // City A's leader rallies the defenders
        const Color skin{225, 175, 140}, gold{220, 170, 40}, grey{150, 140, 130};
        double pop = easeOutCubic((t - RALLY_T) / 1.5);
        auto mk = [](double x0, double x1, double z0, double z1, double y0, double y1, Color c) {
            Box b{x0, z0, x1, z1, y1, c}; b.y0 = y0; return b;
        };
        drawBoxGL(mk(-1.3, 1.3, 5.4, 7.0, 0.0, 1.2 * pop, grey));                // podium
        drawBoxGL(mk(-0.32, 0.32, 6.0, 6.4, 1.2 * pop, 2.7 * pop, gold));        // torso
        drawBoxGL(mk(-0.20, 0.20, 6.05, 6.35, 2.8 * pop, 3.2 * pop, skin));      // head
        // arms: children of the shoulder joints.  T(shoulder) -> Rz(angle) -> T(0,-len/2,0) -> S(size)
        const double armLen = 1.0 * pop;
        double aR = 2.3 + 0.35 * std::sin(t * 3.2);                              // right arm raised and pumping
        double aL = -(2.0 + 0.35 * std::sin(t * 2.7 + 1));                       // left arm raised
        for (int side = 0; side < 2; ++side) {
            glPushMatrix();
            glTranslated(side == 0 ? 0.42 : -0.42, 2.55 * pop, LEADER_Z);
            glRotated((side == 0 ? aR : aL) * 180.0 / PI, 0, 0, 1);
            glTranslated(0, -armLen / 2, 0);
            glScaled(0.18 * pop, armLen, 0.18 * pop);
            drawCentredCube(gold);
            glPopMatrix();
        }
        for (const Person& p : w.crowd) {
            if (t < p.appear) continue;
            double pp = easeOutCubic((t - p.appear) / 1.0);
            double bob = 0.10 * std::fabs(std::sin(t * 4 + p.phase)) * pp;       // cheering up and down
            glPushMatrix();
            glTranslated(p.x, 0.5 * pp + bob, p.z); glScaled(0.34, 0.95 * pp, 0.28);
            drawCentredCube(p.col);
            glPopMatrix();
            glPushMatrix();
            glTranslated(p.x, 1.2 * pp + bob, p.z); glScaled(0.26 * pp, 0.26 * pp, 0.26 * pp);
            drawCentredCube(skin);
            glPopMatrix();
        }
        glPopMatrix();
    }
    } // end exterior-world pass; cabinet interiors are drawn separately below

    drawStoryScenes(t);

    if (!cabinetInterior) {
    for (const Missile& m : w.missiles) drawMissileGL(m, t);

    // ---- transparent effects: no lighting, no depth writes, blending
    std::vector<Part> smoke, fire; std::vector<Ring> rings;
    collectEffects(w, t, smoke, fire, rings);
    glDisable(GL_LIGHTING); glDisable(GL_CULL_FACE); glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glLineWidth(2.0f);
    for (const Ring& r : rings) {                               // shockwave rings on the ground
        glColor4f(1.0f, 0.92f, 0.75f, r.a);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 40; ++i) glVertex3d(r.c.x + r.R * std::cos(2 * PI * i / 40), 0.4, r.c.z + r.R * std::sin(2 * PI * i / 40));
        glEnd();
    }
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, g_softTex);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);          // smoke: normal alpha blending
    drawParts(smoke, right, up);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);                          // fire and sparks: additive blending
    drawParts(fire, right, up);
    glDisable(GL_TEXTURE_2D); glDepthMask(GL_TRUE);
    } // end missile and particle effects pass

    // ---- screen overlays: impact flash and fade in / out
    glDisable(GL_DEPTH_TEST); glDisable(GL_FOG);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, fbW, fbH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    double flash = 0, flashCap = 0.3;
    for (const Missile& m : w.missiles) {
        double age = t - (m.launchT + m.flightT);
        if (age >= 0 && age < 0.15) {
            flash += (m.big ? 0.55 : 0.10) * (1 - age / 0.15);
            if (m.big) flashCap = 0.75;                          // NEW: the strike flashes the whole screen
        }
    }
    if (flash > 0) screenQuad(fbW, fbH, 1.0f, 0.96f, 0.86f, static_cast<float>(std::min(flash, flashCap)));

    // Cinematic blackout between exterior and interior shots. The black frame
    // hides the old city view completely, then the next frame reveals the room.
    // Three cuts: enter City A room, switch to City B room, return to the attack.
    const bool cabinetCutBlack =
        (t >= 197.65 && t < 198.25) ||
        (t >= 211.65 && t < 212.25) ||
        (t >= 225.65 && t < 226.25);
    if (cabinetCutBlack) {
        screenQuad(fbW, fbH, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    double fade = std::min(clamp01(t / 2.0), clamp01((TOTAL_TIME - t) / 4.0));
    if (fade < 1.0) screenQuad(fbW, fbH, 0, 0, 0, static_cast<float>(1.0 - fade));
    if (t >= 338) {
        // High-contrast, centered end card built from scalable 5x7 pixel glyphs.
        glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, fbW, 0, fbH, -1, 1);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glColor4f(0.015f, 0.02f, 0.035f, 0.94f);
        glBegin(GL_QUADS); glVertex2f(0,0); glVertex2f((float)fbW,0); glVertex2f((float)fbW,(float)fbH); glVertex2f(0,(float)fbH); glEnd();
        const char* msg = "TO BE CONTINUED...";
        auto glyph = [](char c) -> const unsigned char* {
            static const unsigned char T[7]={31,4,4,4,4,4,4}, O[7]={14,17,17,17,17,17,14};
            static const unsigned char B[7]={30,17,17,30,17,17,30}, E[7]={31,16,16,30,16,16,31};
            static const unsigned char C[7]={14,17,16,16,16,17,14}, N[7]={17,25,25,21,19,19,17};
            static const unsigned char U[7]={17,17,17,17,17,17,14}, D[7]={30,17,17,17,17,17,30};
            static const unsigned char I[7]={14,4,4,4,4,4,14}, dot[7]={0,0,0,0,0,4,4}, blank[7]={0,0,0,0,0,0,0};
            switch(c){case 'T':return T;case 'O':return O;case 'B':return B;case 'E':return E;case 'C':return C;case 'N':return N;case 'U':return U;case 'D':return D;case 'I':return I;case '.':return dot;default:return blank;}
        };
        const float scale = std::max(4.0f, std::min(9.0f, (float)fbW/125.0f));
        const float advance = 6.0f*scale;
        const float totalW = (float)std::strlen(msg)*advance-scale;
        const float x0 = ((float)fbW-totalW)*0.5f;
        const float y0 = (float)fbH*0.5f-3.5f*scale;
        glColor3f(1.0f,0.88f,0.48f);
        glBegin(GL_QUADS);
        for (int n=0; msg[n]; ++n) {
            const unsigned char* rows=glyph(msg[n]);
            for(int row=0;row<7;++row) for(int col=0;col<5;++col) if(rows[row] & (1u<<(4-col))) {
                float x=x0+n*advance+col*scale, y=y0+(6-row)*scale;
                glVertex2f(x,y); glVertex2f(x+scale*0.88f,y); glVertex2f(x+scale*0.88f,y+scale*0.88f); glVertex2f(x,y+scale*0.88f);
            }
        }
        glEnd();
    }
    glDisable(GL_BLEND);
}

// ======================================================================= Application
struct App {
    double t = 0, speed = 1.0;                           // t is FILM time
    bool paused = false, wantShot = false;
    FreeCam cam;
    World world;
} g_app;

bool saveScreenshot(const char* path, int w, int h) {
    std::vector<unsigned char> px(static_cast<size_t>(w) * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = h - 1; y >= 0; --y) std::fwrite(&px[static_cast<size_t>(y) * w * 3], 1, static_cast<size_t>(w) * 3, f);
    std::fclose(f);
    return true;
}

void onKey(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    App& a = g_app;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(win, GLFW_TRUE); break;
        case GLFW_KEY_SPACE:  a.paused = !a.paused; break;
        case GLFW_KEY_R:      a.t = 0; a.paused = false; break;
        case GLFW_KEY_RIGHT:  a.t = std::min(FILM_TIME, a.t + 5); break;
        case GLFW_KEY_LEFT:   a.t = std::max(0.0, a.t - 5); break;
        case GLFW_KEY_UP:     a.speed = std::min(16.0, a.speed * 2); break;
        case GLFW_KEY_DOWN:   a.speed = std::max(0.125, a.speed / 2); break;
        case GLFW_KEY_S:      a.wantShot = true; break;
        case GLFW_KEY_F: {
            a.cam.on = !a.cam.on;
            if (a.cam.on) {                                      // start from the current scripted view
                Vec3 e, tg; cameraAt(storyTime(a.t), e, tg);
                Vec3 d = normalize(tg - e);
                a.cam.eye = e; a.cam.yaw = std::atan2(d.x, -d.z); a.cam.pitch = std::asin(d.y);
            }
            break;
        }
        // CHANGED: the jump keys take story seconds and convert them to film time; 7 and 8 are new
        case GLFW_KEY_1: a.t = 0;                 break;
        case GLFW_KEY_2: a.t = filmTime(120);     break;
        case GLFW_KEY_3: a.t = filmTime(180);     break;
        case GLFW_KEY_4: a.t = filmTime(210);     break;
        case GLFW_KEY_5: a.t = filmTime(240);     break;
        case GLFW_KEY_6: a.t = filmTime(260.5);   break;   // statehouse close-up
        case GLFW_KEY_7: a.t = filmTime(274);     break;   // City A leader rallies the crowd
        case GLFW_KEY_8: a.t = filmTime(318);     break;   // City B occupation of City A ruins
        case GLFW_KEY_9: a.t = filmTime(330);     break;   // sewer rebellion ending
        default: break;
    }
}

int main(int argc, char** argv) {
    generateCity(g_app.world, 0, 0, N_MAIN, 0);
    generateCity(g_app.world, RIVAL_CENTRE.x, 0, N_RIVAL, 1);
    planWar(g_app.world);
    makeCrowd(g_app.world);                                      // NEW

    bool shotMode = false; double shotT = 0; const char* shotFile = "scene3_shot.ppm";
    if (argc > 3 && std::strcmp(argv[1], "--shot") == 0) { shotMode = true; shotT = std::atof(argv[2]); shotFile = argv[3]; }
    else if (argc > 1) g_app.t = std::atof(argv[1]);

    if (!glfwInit()) { std::fprintf(stderr, "Could not start GLFW\n"); return 1; }
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1280, 720, "Rise and Fall of a City", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "Could not create an OpenGL window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetKeyCallback(win, onKey);
    buildCubeList();
    buildSoftTexture();

    if (shotMode) {
        int fw, fh;
        for (int i = 0; i < 2; ++i) {
            glfwGetFramebufferSize(win, &fw, &fh);
            renderGL(g_app.world, shotT, fw, fh, g_app.cam);
            if (i == 0) glfwSwapBuffers(win);
        }
        bool ok = saveScreenshot(shotFile, fw, fh);
        std::printf(ok ? "Saved %s\n" : "Could not write %s\n", shotFile);
        glfwTerminate();
        return ok ? 0 : 1;
    }

    double last = glfwGetTime(), titleTimer = 0, lastX = 0, lastY = 0;
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime(), dt = std::min(now - last, 0.1);
        last = now;
        if (!g_app.paused) {
            g_app.t += dt * g_app.speed;
            if (g_app.t >= FILM_TIME) { g_app.t = FILM_TIME; g_app.paused = true; }       // CHANGED: film length
        }
        // free camera input (movement and mouse look)
        if (g_app.cam.on) {
            FreeCam& c = g_app.cam;
            double mx, my; glfwGetCursorPos(win, &mx, &my);
            if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
                c.yaw += (mx - lastX) * 0.004;
                c.pitch = std::max(-1.5, std::min(1.5, c.pitch - (my - lastY) * 0.004));
            }
            lastX = mx; lastY = my;
            Vec3 f = freeDir(c), r = normalize(cross(f, {0, 1, 0}));
            double sp = (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 90.0 : 30.0) * dt;
            if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) c.eye = c.eye + f * sp;
            if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) c.eye = c.eye - f * sp;
            if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) c.eye = c.eye + r * sp;
            if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) c.eye = c.eye - r * sp;
            if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) c.eye.y += sp;
            if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) c.eye.y -= sp;
        }
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        if (fw > 0 && fh > 0) renderGL(g_app.world, g_app.t, fw, fh, g_app.cam);
        if (g_app.wantShot) {
            char name[64]; std::snprintf(name, sizeof name, "scene3_shot_%03d.ppm", static_cast<int>(g_app.t));
            if (saveScreenshot(name, fw, fh)) std::printf("Saved %s\n", name);
            g_app.wantShot = false;
        }
        glfwSwapBuffers(win);
        glfwPollEvents();

        titleTimer += dt;
        if (titleTimer > 0.25) {
            titleTimer = 0;
            char title[192]; int s = static_cast<int>(g_app.t), e = static_cast<int>(FILM_TIME);
            std::snprintf(title, sizeof title, "Rise and Fall of a City | %d:%02d / %d:%02d | x%.2g%s%s",
                          s / 60, s % 60, e / 60, e % 60, g_app.speed,
                          g_app.paused ? " | PAUSED" : "", g_app.cam.on ? " | FREE CAMERA" : "");
            glfwSetWindowTitle(win, title);
        }
    }
    glfwTerminate();
    return 0;
}