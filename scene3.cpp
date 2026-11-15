// Checkpoint 3 - OpenGL integration
// Project: "Rise and Fall of a City" - 5 min 30 s animated story
//
// The pure C++ work from scenes  1 and 2 is reused as it is:
//   Vec3 / Mat4 / lookAt / perspective   -> loaded straight into OpenGL (glLoadMatrixd)
//   cubic Bezier + tangent               -> missile paths, missile orientation
//   easing + keyframes                   -> camera path and sky colours
//   procedural city, collapse timing     -> same generator and the same war plan
// Only the RENDERER changes: the software rasteriser is replaced by OpenGL, which gives us
//   depth buffer (no painter's sorting), lighting, fog, alpha/additive blending, textured particles,
//   the OpenGL matrix stack (glPushMatrix / glTranslated / glScaled / glMultMatrixd for shear).
//
// Uses fixed-function OpenGL 1.x + GLFW (window and input only), so no GLAD/GLEW is needed.
//
// Build
//   Windows (MSYS2 UCRT64): g++ -std=c++17 -O2 checkpoint3.cpp -o cp3 -lglfw3 -lopengl32 -lgdi32
//   Linux:                  g++ -std=c++17 -O2 checkpoint3.cpp -o cp3 -lglfw -lGL
//   macOS (Homebrew):       g++ -std=c++17 -O2 checkpoint3.cpp -o cp3 -I/opt/homebrew/include -L/opt/homebrew/lib -lglfw -framework OpenGL
//
// Controls:
//   SPACE pause/resume   LEFT/RIGHT seek -/+ 5 s   UP/DOWN speed x2 / /2   R restart   1..6 jump to phase
//   F free camera (WASD move, Q/E down/up, hold left mouse button to look, SHIFT faster)
//   S save screenshot (PPM)   ESC quit
// Command line:
//   ./cp3                       play from the start
//   ./cp3 <seconds>             start at that time
//   ./cp3 --shot <seconds> <file.ppm>   render one frame to a file and exit

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
const double TOTAL_TIME = 330.0;                       // 5 min 30 s
const double REF_W = 960, REF_H = 560;                 // reference screen size used for the sun arc

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

struct Box { double x0, z0, x1, z1, h; Color base; double sx = 0, sz = 0; };

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
};
struct Block   { double x0, z0; bool park; int city; };
struct Missile { Vec3 p0, p1, p2, p3; double launchT, flightT; };
struct Car     { int road; bool alongX; double lane, speed, offset, dir, startT, endT; Color col; };

struct World {
    std::vector<Building> buildings;
    std::vector<Block> blocks;
    std::vector<Missile> missiles;
    std::vector<Car> cars;
};

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

// Plans the attack: 20 missiles on Bezier paths, collapse times for nearby buildings, traffic
void planWar(World& w) {
    std::vector<Vec3> targets;
    for (int xi = 0; xi < 5; ++xi) {                          // west (sea side) to east
        double zs[4] = {-27, -9, 9, 27};
        for (int k = 3; k > 0; --k) std::swap(zs[k], zs[static_cast<int>(rnd() * (k + 1))]);
        for (double z : zs) targets.push_back({-30.0 + 15.0 * xi, 6.0, z + (rnd() - 0.5) * 4});
    }
    for (size_t i = 0; i < targets.size(); ++i) {
        double impactT = 240.0 + 3.0 * i;
        Vec3 T = targets[i];
        Vec3 L = {RIVAL_CENTRE.x + (rnd() - 0.5) * 14, 3.0, (rnd() - 0.5) * 24};
        Missile m;
        m.p0 = L;
        m.p1 = L + Vec3{30, 85 + rnd() * 15, (T.z - L.z) * 0.25};
        m.p2 = T + Vec3{-55, 80 + rnd() * 15, (L.z - T.z) * 0.15};
        m.p3 = T;
        m.flightT = 30.0; m.launchT = impactT - m.flightT;
        w.missiles.push_back(m);
    }
    for (Building& b : w.buildings) {
        if (b.city != 0) continue;
        double cx = (b.box.x0 + b.box.x1) / 2, cz = (b.box.z0 + b.box.z1) / 2;
        for (const Missile& m : w.missiles) {
            double d = std::hypot(cx - m.p3.x, cz - m.p3.z);
            if (d < 11.0) b.collapseT = std::min(b.collapseT, m.launchT + m.flightT + 0.12 * d);
        }
        b.collapseDur = 3 + rnd() * 2;
        b.shearX = (rnd() - 0.5) * 1.2; b.shearZ = (rnd() - 0.5) * 1.2;
    }
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
struct CamKey { double t; Vec3 eye, target; };
const CamKey CAM[] = {
    {  0, {70, 48, 70},  {0, 4, 0}},      // dawn, wide shot
    {120, {95, 50, 20},  {0, 6, 0}},      // orbit while the city rises
    {180, {20, 55, 95},  {0, 6, 0}},      // peak
    {210, {70, 35, 40},  {-70, 10, 0}},   // tension: pan to the rival city
    {240, {60, 40, 45},  {-55, 15, 0}},   // launch
    {270, {65, 45, 55},  {-10, 6, 0}},    // strike
    {300, {70, 52, 60},  {0, 3, 0}},      // aftermath
    {330, {110, 70, 95}, {0, 3, 0}},      // slow pull-back
};
void cameraAt(double t, Vec3& eye, Vec3& target) {
    const int n = sizeof(CAM) / sizeof(CAM[0]);
    if (t <= CAM[0].t) { eye = CAM[0].eye; target = CAM[0].target; return; }
    for (int i = 0; i + 1 < n; ++i) {
        if (t <= CAM[i + 1].t) {
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
void drawBoxGL(const Box& b) {
    double h = b.h;
    if (h < 0.03) return;
    glPushMatrix();
    glTranslated(b.x0, 0, b.z0);
    double shear[16] = {1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1};
    shear[4] = b.sx / h;                       // x' = x + (sx/h) * y
    shear[6] = b.sz / h;                       // z' = z + (sz/h) * y
    glMultMatrixd(shear);
    glScaled(b.x1 - b.x0, h, b.z1 - b.z0);
    colorOf(b.base);
    glCallList(g_cubeList);
    glPopMatrix();
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
    for (size_t i = 0; i < w.buildings.size(); ++i) {               // smoke from every collapsed building
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
        if (age < 1.5) rings.push_back({P, 16.0 * age / 1.5, static_cast<float>(1 - age / 1.5)});
        if (age < 1.4) {
            double k = std::sin(PI * age / 1.4);
            addPart(fire, {P.x, P.y + 2 + age * 3, P.z}, 6.0 * k, Color{255, 150, 40}, 0.7);
            addPart(fire, {P.x, P.y + 2 + age * 3, P.z}, 3.6 * k, Color{255, 240, 150}, 0.9);
        }
        for (int k = 0; k < 40; ++k) {
            double life = 1.5 + 1.5 * hash01(id * 100 + k, 4);
            if (age > life) continue;
            double th = 2 * PI * hash01(id * 100 + k, 1), up = 0.3 + 0.7 * hash01(id * 100 + k, 2);
            double sp = 6 + 14 * hash01(id * 100 + k, 3);
            Vec3 v = {sp * std::cos(th) * (1 - up * 0.5), sp * up * 1.2 + 4, sp * std::sin(th) * (1 - up * 0.5)};
            Vec3 p = P + v * age;
            p.y = std::max(0.0, p.y - 0.5 * 9.8 * age * age);
            double f = age / life;
            Color c = f < 0.5 ? lerpColor(Color{255, 230, 90}, Color{255, 120, 30}, f * 2)
                              : lerpColor(Color{255, 120, 30}, Color{90, 50, 40}, (f - 0.5) * 2);
            addPart(fire, p, 1.1 * (1 - 0.5 * f), c, 1.0 - f);
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

void renderGL(const World& w, double t, int fbW, int fbH, const FreeCam& fc) {
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

    // ---- camera: scripted keyframes (or the free camera), plus a shake after each impact
    Vec3 eye, target; cameraAt(t, eye, target);
    if (fc.on) { eye = fc.eye; target = fc.eye + freeDir(fc); }
    else {
        Vec3 shake = {0, 0, 0};
        for (size_t i = 0; i < w.missiles.size(); ++i) {
            double age = t - (w.missiles[i].launchT + w.missiles[i].flightT);
            if (age >= 0 && age < 1.2) {
                double a = (1 - age / 1.2) * (1 - age / 1.2) * 0.9;
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
            GLfloat pos[4] = {(GLfloat)w.missiles[i].p3.x, 8.0f, (GLfloat)w.missiles[i].p3.z, 1};
            GLfloat col[4] = {(GLfloat)(1.0 * k), (GLfloat)(0.55 * k), (GLfloat)(0.2 * k), 1};
            glEnable(L); glLightfv(L, GL_POSITION, pos); glLightfv(L, GL_DIFFUSE, col); glLightfv(L, GL_AMBIENT, zero);
            glLightf(L, GL_CONSTANT_ATTENUATION, 1.0f); glLightf(L, GL_LINEAR_ATTENUATION, 0.02f);
            glLightf(L, GL_QUADRATIC_ATTENUATION, 0.004f);
            ++nl;
        }
    }
    for (int i = nl; i < 4; ++i) glDisable(GL_LIGHT1 + i);

    // ---- ground layers (heights differ slightly so the layers do not flicker)
    glDisable(GL_CULL_FACE);
    groundQuad(-600, -600, 600, 600, 0.0, Color{105, 150, 90});                       // land
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
        double ce = t < b.collapseT ? 0 : easeInCubic((t - b.collapseT) / b.collapseDur);
        if (grow <= 0) continue;
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
    if (t > 305) {                                              // a new building starts to rise at dawn
        double g = easeOutCubic((t - 305) / 20.0);
        drawBoxGL(Box{-1.4, -1.4, 1.4, 1.4, 9.0 * g, Color{200, 225, 255}});
    }
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

    // ---- screen overlays: impact flash and fade in / out
    glDisable(GL_DEPTH_TEST); glDisable(GL_FOG);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, fbW, fbH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    double flash = 0;
    for (const Missile& m : w.missiles) {
        double age = t - (m.launchT + m.flightT);
        if (age >= 0 && age < 0.15) flash += 0.10 * (1 - age / 0.15);
    }
    if (flash > 0) screenQuad(fbW, fbH, 1.0f, 0.96f, 0.86f, static_cast<float>(std::min(flash, 0.3)));
    double fade = std::min(clamp01(t / 2.0), clamp01((TOTAL_TIME - t) / 4.0));
    if (fade < 1.0) screenQuad(fbW, fbH, 0, 0, 0, static_cast<float>(1.0 - fade));
    glDisable(GL_BLEND);
}

// ======================================================================= Application
struct App {
    double t = 0, speed = 1.0;
    bool paused = false, wantShot = false;
    FreeCam cam;
    World world;
} g_app;

const char* phaseName(double t) {
    if (t < 120) return "Rise";
    if (t < 180) return "Peak";
    if (t < 210) return "Tension";
    if (t < 240) return "Launch";
    if (t < 300) return "Strike";
    return "Aftermath";
}

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
        case GLFW_KEY_RIGHT:  a.t = std::min(TOTAL_TIME, a.t + 5); break;
        case GLFW_KEY_LEFT:   a.t = std::max(0.0, a.t - 5); break;
        case GLFW_KEY_UP:     a.speed = std::min(16.0, a.speed * 2); break;
        case GLFW_KEY_DOWN:   a.speed = std::max(0.125, a.speed / 2); break;
        case GLFW_KEY_S:      a.wantShot = true; break;
        case GLFW_KEY_F: {
            a.cam.on = !a.cam.on;
            if (a.cam.on) {                                      // start from the current scripted view
                Vec3 e, tg; cameraAt(a.t, e, tg);
                Vec3 d = normalize(tg - e);
                a.cam.eye = e; a.cam.yaw = std::atan2(d.x, -d.z); a.cam.pitch = std::asin(d.y);
            }
            break;
        }
        case GLFW_KEY_1: a.t = 0;   break;
        case GLFW_KEY_2: a.t = 120; break;
        case GLFW_KEY_3: a.t = 180; break;
        case GLFW_KEY_4: a.t = 210; break;
        case GLFW_KEY_5: a.t = 240; break;
        case GLFW_KEY_6: a.t = 300; break;
        default: break;
    }
}

int main(int argc, char** argv) {
    generateCity(g_app.world, 0, 0, N_MAIN, 0);
    generateCity(g_app.world, RIVAL_CENTRE.x, 0, N_RIVAL, 1);
    planWar(g_app.world);

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
            if (g_app.t >= TOTAL_TIME) { g_app.t = TOTAL_TIME; g_app.paused = true; }
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
            char title[160]; int s = static_cast<int>(g_app.t);
            std::snprintf(title, sizeof title, "Rise and Fall of a City | %s | %d:%02d / 5:30 | x%.2g%s%s", phaseName(g_app.t),
                          s / 60, s % 60, g_app.speed, g_app.paused ? " | PAUSED" : "", g_app.cam.on ? " | FREE CAMERA" : "");
            glfwSetWindowTitle(win, title);
        }
    }
    glfwTerminate();
    return 0;
}
