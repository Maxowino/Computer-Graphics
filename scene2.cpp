// Checkpoint 2 - Transformations and animation in pure C++ (no graphics libraries)

//
// Builds on Checkpoint 1 (framebuffer, Bresenham, midpoint circle, scanline fill,
// 4x4 matrices, perspective projection, procedural city) and adds:
//   - Cohen-Sutherland line clipping and near-plane polygon clipping
//   - Cubic Bezier curves (missile flight paths, sun arc)
//   - Easing functions and keyframe interpolation (scripted camera, sky colours)
//   - Scale animation (buildings grow), shear + scale (collapse), translation (cars)
//   - Rotation of the missile along the curve tangent
//   - Stateless particle system with gravity (explosions, smoke), alpha blending
//
// Build:  g++ -std=c++17 -O2 -Wall -Wextra checkpoint2.cpp -o cp2
// Usage:
//   ./cp2                         key stills of the story  -> cp2_still_XXX.ppm
//   ./cp2 still <seconds>         one frame at that time   -> cp2_still.ppm
//   ./cp2 seq <fps> <from> <to>   frame sequence in seconds -> frame_00000.ppm ...
//   ./cp2 pipe <fps> [from to]    raw PPM frames to stdout (pipe into ffmpeg)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

const double PI = 3.14159265358979323846;
const int    W = 960, H = 560;
const double FOV_Y = 36.0 * PI / 180.0;
const double FOCAL = (H / 2.0) / std::tan(FOV_Y / 2.0);    // pixels
const double NEAR_W = 0.5;                                 // near plane distance
const double TOTAL_TIME = 330.0;                           // time of the story in seconds

// ======================================================================= Framebuffer
struct Color { uint8_t r, g, b; };

class Framebuffer {
public:
    int w, h;
    std::vector<Color> px;
    Framebuffer(int w_, int h_) : w(w_), h(h_), px(static_cast<size_t>(w_) * h_, Color{0, 0, 0}) {}

    void setPixel(int x, int y, Color c) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        px[static_cast<size_t>(y) * w + x] = c;
    }
    void blendPixel(int x, int y, Color c, double a) {          // alpha blending
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        Color& d = px[static_cast<size_t>(y) * w + x];
        d.r = static_cast<uint8_t>(d.r + (c.r - d.r) * a);
        d.g = static_cast<uint8_t>(d.g + (c.g - d.g) * a);
        d.b = static_cast<uint8_t>(d.b + (c.b - d.b) * a);
    }
    void overlay(Color c, double a) { for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) blendPixel(x, y, c, a); }
    void writePPM(FILE* f) const {
        std::fprintf(f, "P6\n%d %d\n255\n", w, h);
        std::fwrite(px.data(), sizeof(Color), px.size(), f);
    }
    bool savePPM(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        writePPM(f);
        std::fclose(f);
        return true;
    }
};

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

// ======================================================================= 2D primitives
// Cohen-Sutherland line clipping against the screen rectangle
bool clipLine(double& x0, double& y0, double& x1, double& y1, int w, int h) {
    const double xmin = 0, ymin = 0, xmax = w - 1, ymax = h - 1;
    auto code = [&](double x, double y) {
        int c = 0;
        if (x < xmin) c |= 1; else if (x > xmax) c |= 2;
        if (y < ymin) c |= 4; else if (y > ymax) c |= 8;
        return c;
    };
    int c0 = code(x0, y0), c1 = code(x1, y1);
    while (true) {
        if (!(c0 | c1)) return true;      // both inside
        if (c0 & c1) return false;        // both outside the same side
        int co = c0 ? c0 : c1;
        double x = 0, y = 0;
        if (co & 8)      { x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0); y = ymax; }
        else if (co & 4) { x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0); y = ymin; }
        else if (co & 2) { y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0); x = xmax; }
        else             { y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0); x = xmin; }
        if (co == c0) { x0 = x; y0 = y; c0 = code(x0, y0); }
        else          { x1 = x; y1 = y; c1 = code(x1, y1); }
    }
}

// Bresenham line (after clipping)
void drawLine(Framebuffer& fb, int ax, int ay, int bx, int by, Color c) {
    double fx0 = ax, fy0 = ay, fx1 = bx, fy1 = by;
    if (!clipLine(fx0, fy0, fx1, fy1, fb.w, fb.h)) return;
    int x0 = static_cast<int>(std::lround(fx0)), y0 = static_cast<int>(std::lround(fy0));
    int x1 = static_cast<int>(std::lround(fx1)), y1 = static_cast<int>(std::lround(fy1));
    int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true) {
        fb.setPixel(x0, y0, c);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void drawThickLine(Framebuffer& fb, int x0, int y0, int x1, int y1, Color c) {
    drawLine(fb, x0, y0, x1, y1, c);
    drawLine(fb, x0 + 1, y0, x1 + 1, y1, c);
    drawLine(fb, x0, y0 + 1, x1, y1 + 1, c);
}

// Midpoint circle outline
void drawCircle(Framebuffer& fb, int cx, int cy, int r, Color c) {
    int x = 0, y = r, d = 1 - r;
    while (x <= y) {
        fb.setPixel(cx + x, cy + y, c); fb.setPixel(cx - x, cy + y, c);
        fb.setPixel(cx + x, cy - y, c); fb.setPixel(cx - x, cy - y, c);
        fb.setPixel(cx + y, cy + x, c); fb.setPixel(cx - y, cy + x, c);
        fb.setPixel(cx + y, cy - x, c); fb.setPixel(cx - y, cy - x, c);
        if (d < 0) d += 2 * x + 3; else { d += 2 * (x - y) + 5; --y; }
        ++x;
    }
}

// Filled circle with alpha blending (sun, particles, smoke)
void fillCircle(Framebuffer& fb, int cx, int cy, int r, Color c, double alpha = 1.0) {
    int y0 = std::max(-r, -cy), y1 = std::min(r, fb.h - 1 - cy);
    for (int dy = y0; dy <= y1; ++dy) {
        int half = static_cast<int>(std::sqrt(static_cast<double>(r * r - dy * dy)));
        int xa = std::max(cx - half, 0), xb = std::min(cx + half, fb.w - 1);
        for (int x = xa; x <= xb; ++x) {
            if (alpha >= 1.0) fb.setPixel(x, cy + dy, c); else fb.blendPixel(x, cy + dy, c, alpha);
        }
    }
}

// Scanline polygon fill (even-odd rule), clamped to the screen
struct P2 { int x, y; };

void fillPolygon(Framebuffer& fb, const std::vector<P2>& v, Color c) {
    if (v.size() < 3) return;
    int minY = v[0].y, maxY = v[0].y;
    for (const P2& p : v) { minY = std::min(minY, p.y); maxY = std::max(maxY, p.y); }
    minY = std::max(minY, 0); maxY = std::min(maxY, fb.h - 1);
    std::vector<int> xs;
    for (int y = minY; y <= maxY; ++y) {
        xs.clear();
        for (size_t i = 0; i < v.size(); ++i) {
            P2 a = v[i], b = v[(i + 1) % v.size()];
            if (a.y == b.y) continue;
            if (y < std::min(a.y, b.y) || y >= std::max(a.y, b.y)) continue;
            double t = static_cast<double>(y - a.y) / (b.y - a.y);
            xs.push_back(static_cast<int>(std::lround(a.x + t * (b.x - a.x))));
        }
        std::sort(xs.begin(), xs.end());
        for (size_t i = 0; i + 1 < xs.size(); i += 2)
            for (int x = std::max(xs[i], 0); x <= std::min(xs[i + 1], fb.w - 1); ++x) fb.setPixel(x, y, c);
    }
}

// ======================================================================= 3D -> screen
// Projects a world-space polygon, clipping it against the near plane first
// (Sutherland-Hodgman in clip space), then perspective divide to pixels.
std::vector<P2> projectPolygon(const Mat4& vp, const std::vector<Vec3>& pts) {
    std::vector<Vec4> in, out;
    for (const Vec3& p : pts) in.push_back(vp * Vec4{p.x, p.y, p.z, 1.0});
    for (size_t i = 0; i < in.size(); ++i) {
        const Vec4& a = in[i]; const Vec4& b = in[(i + 1) % in.size()];
        double da = a.w - NEAR_W, db = b.w - NEAR_W;
        if (da >= 0) out.push_back(a);
        if ((da >= 0) != (db >= 0)) {
            double t = da / (da - db);
            out.push_back({a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t});
        }
    }
    std::vector<P2> res;
    for (const Vec4& v : out)
        res.push_back({static_cast<int>(std::lround((v.x / v.w + 1) * 0.5 * W)),
                       static_cast<int>(std::lround((1 - v.y / v.w) * 0.5 * H))});
    return res;
}

bool projectPoint(const Mat4& vp, Vec3 p, int& sx, int& sy, double* wOut = nullptr) {
    Vec4 c = vp * Vec4{p.x, p.y, p.z, 1.0};
    if (c.w < NEAR_W) return false;
    sx = static_cast<int>(std::lround((c.x / c.w + 1) * 0.5 * W));
    sy = static_cast<int>(std::lround((1 - c.y / c.w) * 0.5 * H));
    if (wOut) *wOut = c.w;
    return true;
}

void fillGround(Framebuffer& fb, const Mat4& vp, const std::vector<Vec3>& pts, Color c) {
    fillPolygon(fb, projectPolygon(vp, pts), c);
}
void fillGroundRect(Framebuffer& fb, const Mat4& vp, double x0, double z0, double x1, double z1, Color c) {
    if (x1 <= x0) return;
    fillGround(fb, vp, {{x0, 0, z0}, {x1, 0, z0}, {x1, 0, z1}, {x0, 0, z1}}, c);
}

// A particle drawn as a soft disc whose pixel size follows perspective
void drawParticle(Framebuffer& fb, const Mat4& vp, Vec3 p, double radiusWorld, Color c, double alpha) {
    int sx, sy; double w;
    if (!projectPoint(vp, p, sx, sy, &w)) return;
    int r = std::max(1, static_cast<int>(std::lround(radiusWorld * FOCAL / w)));
    fillCircle(fb, sx, sy, r, c, alpha);
}

// ======================================================================= Solid boxes
// A box with optional top shear (used for the collapse animation)
struct Box { double x0, z0, x1, z1, h; Color base; double sx = 0, sz = 0; };

void drawBox(Framebuffer& fb, const Mat4& vp, Vec3 eye, const Box& b, double bright) {
    if (b.h < 0.03) return;
    const Vec3 L = normalize({0.45, 0.8, 0.35});
    const double x0 = b.x0, x1 = b.x1, z0 = b.z0, z1 = b.z1, h = b.h, sx = b.sx, sz = b.sz;
    struct Face { Vec3 p[4]; Vec3 n; };
    const Face faces[5] = {
        {{{x0 + sx, h, z0 + sz}, {x1 + sx, h, z0 + sz}, {x1 + sx, h, z1 + sz}, {x0 + sx, h, z1 + sz}}, { 0, 1,  0}},
        {{{x1, 0, z0}, {x1, 0, z1}, {x1 + sx, h, z1 + sz}, {x1 + sx, h, z0 + sz}}, { 1, 0,  0}},
        {{{x0, 0, z0}, {x0, 0, z1}, {x0 + sx, h, z1 + sz}, {x0 + sx, h, z0 + sz}}, {-1, 0,  0}},
        {{{x0, 0, z1}, {x1, 0, z1}, {x1 + sx, h, z1 + sz}, {x0 + sx, h, z1 + sz}}, { 0, 0,  1}},
        {{{x0, 0, z0}, {x1, 0, z0}, {x1 + sx, h, z0 + sz}, {x0 + sx, h, z0 + sz}}, { 0, 0, -1}},
    };
    for (const Face& f : faces) {
        Vec3 centre = (f.p[0] + f.p[2]) * 0.5;
        if (dot(f.n, eye - centre) <= 0) continue;                 // back-face culling
        std::vector<P2> poly = projectPolygon(vp, {f.p[0], f.p[1], f.p[2], f.p[3]});
        if (poly.size() < 3) continue;
        double k = (0.35 + 0.65 * std::max(0.0, dot(f.n, L))) * bright;
        fillPolygon(fb, poly, shade(b.base, k));
        Color edge = shade(b.base, k * 0.55);
        for (size_t i = 0; i < poly.size(); ++i)
            drawLine(fb, poly[i].x, poly[i].y, poly[(i + 1) % poly.size()].x, poly[(i + 1) % poly.size()].y, edge);
    }
}

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

// ======================================================================= Effects
void drawExplosion(Framebuffer& fb, const Mat4& vp, Vec3 P, double age, int id) {
    if (age < 0) return;
    if (age < 1.5) {                                           // shockwave ring on the ground
        double R = 16.0 * age / 1.5;
        Color c = shade(Color{255, 235, 190}, 1.0 - age / 1.5);
        int px = 0, py = 0, fx = 0, fy = 0; bool have = false, first = true;
        for (int i = 0; i <= 28; ++i) {
            double a = 2 * PI * i / 28;
            int sx, sy;
            if (projectPoint(vp, {P.x + R * std::cos(a), 0.2, P.z + R * std::sin(a)}, sx, sy)) {
                if (have) drawLine(fb, px, py, sx, sy, c);
                px = sx; py = sy; have = true;
                if (first) { fx = sx; fy = sy; first = false; }
            } else have = false;
        }
        (void)fx; (void)fy;
    }
    if (age < 1.4) {                                           // fireball core
        double k = std::sin(PI * age / 1.4);
        drawParticle(fb, vp, {P.x, P.y + 2 + age * 3, P.z}, 5.0 * k, Color{255, 150, 40}, 0.55);
        drawParticle(fb, vp, {P.x, P.y + 2 + age * 3, P.z}, 3.0 * k, Color{255, 240, 150}, 0.75);
    }
    for (int k = 0; k < 40; ++k) {                             // debris and sparks with gravity
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
        drawParticle(fb, vp, p, 0.8 * (1 - 0.5 * f), c, 1.0 - f);
    }
}

void drawMissile(Framebuffer& fb, const Mat4& vp, const Missile& m, double t) {
    if (t < m.launchT || t > m.launchT + m.flightT) return;
    double u = (t - m.launchT) / m.flightT;
    Vec3 head = bezier(m.p0, m.p1, m.p2, m.p3, u);
    Vec3 dir = normalize(bezierTangent(m.p0, m.p1, m.p2, m.p3, u));   // missile rotates to follow the curve
    for (int k = 14; k >= 1; --k) {                                     // exhaust trail sampled along the curve
        double uk = std::max(0.0, u - k * 0.006);
        drawParticle(fb, vp, bezier(m.p0, m.p1, m.p2, m.p3, uk), 0.5 + 0.09 * k, Color{200, 200, 205}, 0.32 * (1 - k / 15.0));
    }
    int ax, ay, bx, by;
    if (projectPoint(vp, head, ax, ay) && projectPoint(vp, head - dir * 3.0, bx, by)) {
        drawThickLine(fb, ax, ay, bx, by, Color{235, 235, 240});
        drawParticle(fb, vp, head - dir * 3.0, 0.9, Color{255, 170, 60}, 0.9);   // engine flame
        drawParticle(fb, vp, head, 0.5, Color{220, 50, 50}, 1.0);                // nose
    }
}

// ======================================================================= Frame renderer
void renderFrame(Framebuffer& fb, const World& w, double t) {
    SkyKey sky = skyAt(t);

    // sky gradient
    for (int y = 0; y < H; ++y) {
        Color c = lerpColor(sky.top, sky.horizon, static_cast<double>(y) / H);
        for (int x = 0; x < W; ++x) fb.setPixel(x, y, c);
    }
    // sun on a cubic Bezier arc (screen space): crosses the sky until 200 s, rises again after 300 s
    {
        double u = -1;
        if (t <= 205) u = t / 200.0; else if (t >= 300) u = (t - 300) / 30.0 * 0.15;
        if (u >= 0) {
            Vec3 p = bezier({60, 380, 0}, {240, -140, 0}, {720, -140, 0}, {900, 380, 0}, clamp01(u));
            fillCircle(fb, static_cast<int>(p.x), static_cast<int>(p.y), 32, Color{255, 215, 110});
            drawCircle(fb, static_cast<int>(p.x), static_cast<int>(p.y), 34, Color{255, 240, 170});
        }
    }

    // camera with a short shake after each impact
    Vec3 eye, target; cameraAt(t, eye, target);
    Vec3 shake = {0, 0, 0};
    for (size_t i = 0; i < w.missiles.size(); ++i) {
        double age = t - (w.missiles[i].launchT + w.missiles[i].flightT);
        if (age >= 0 && age < 1.2) {
            double a = (1 - age / 1.2) * (1 - age / 1.2) * 0.9;
            shake = shake + Vec3{a * std::sin(t * 47 + i), a * std::cos(t * 61 + 2 * i) * 0.6, a * std::sin(t * 53 + 3 * i)};
        }
    }
    eye = eye + shake; target = target + shake;
    Mat4 vp = Mat4::perspective(FOV_Y, static_cast<double>(W) / H, 0.1, 2000.0) * lookAt(eye, target, {0, 1, 0});
    const double bright = sky.bright;

    // land, sea, rival island
    fillGroundRect(fb, vp, -600, -600, 600, 600, shade(Color{105, 150, 90}, bright));
    std::vector<Vec3> sea;
    for (double z = -600; z <= 600; z += 10) sea.push_back({-46 + 3.0 * std::sin(z * 0.15), 0, z});
    sea.push_back({-800, 0, 600}); sea.push_back({-800, 0, -600});
    fillGround(fb, vp, sea, shade(Color{45, 105, 160}, bright));
    const double ri = N_RIVAL * PITCH / 2 + 8;
    const bool rivalVisible = t >= 170;                         // the rival city appears during the tension phase
    if (rivalVisible) fillGroundRect(fb, vp, RIVAL_CENTRE.x - ri, -ri, RIVAL_CENTRE.x + ri, ri, shade(Color{95, 120, 85}, bright));

    // roads draw in from west to east (0 - 20 s)
    double reveal = easeOutCubic((t - 2) / 18.0);
    double xr = -HALF - 1 + reveal * (2 * HALF + 2);
    fillGroundRect(fb, vp, -HALF - 1, -HALF - 1, xr, HALF + 1, shade(Color{58, 58, 64}, bright));
    for (int k = 0; k <= N_MAIN; ++k) {
        double c = -HALF + k * PITCH;
        for (double s = -HALF; s < HALF; s += 4.0) {
            int ax, ay, bx, by;
            if (c <= xr && projectPoint(vp, {c, 0.01, s}, ax, ay) && projectPoint(vp, {c, 0.01, s + 2}, bx, by))
                drawLine(fb, ax, ay, bx, by, shade(Color{225, 200, 80}, bright));
            if (s + 2 <= xr && projectPoint(vp, {s, 0.01, c}, ax, ay) && projectPoint(vp, {s + 2, 0.01, c}, bx, by))
                drawLine(fb, ax, ay, bx, by, shade(Color{225, 200, 80}, bright));
        }
    }
    for (const Block& b : w.blocks) {
        if (b.city == 0 && b.x0 + BLOCK > xr) continue;
        if (b.city == 1 && !rivalVisible) continue;
        Color c = b.park ? Color{85, 160, 90} : (b.city == 0 ? Color{150, 150, 148} : Color{140, 120, 115});
        fillGroundRect(fb, vp, b.x0, b.z0, b.x0 + BLOCK, b.z0 + BLOCK, shade(c, bright));
    }

    // dynamic boxes: buildings (grow / collapse), cars, the new building at the end
    std::vector<Box> boxes;
    for (const Building& b : w.buildings) {
        double grow = t <= b.growStart ? 0 : easeOutCubic((t - b.growStart) / b.growDur);
        double ce = t < b.collapseT ? 0 : easeInCubic((t - b.collapseT) / b.collapseDur);
        if (grow <= 0) continue;
        Box x = b.box;
        x.h = b.box.h * grow * (1 - 0.88 * ce);                 // scale down to rubble
        x.sx = b.shearX * ce * b.box.h * grow * 0.5;            // shear: the top leans over
        x.sz = b.shearZ * ce * b.box.h * grow * 0.5;
        x.base = lerpColor(b.box.base, Color{55, 50, 50}, ce);  // burnt
        boxes.push_back(x);
    }
    for (const Car& c : w.cars) {                               // translation along the roads
        if (t < c.startT || t > c.endT) continue;
        double p = std::fmod(c.offset + t * c.speed, 2 * HALF);
        if (c.dir < 0) p = 2 * HALF - p;
        double along = -HALF + p, centre = -HALF + c.road * PITCH + c.lane;
        if (c.alongX) boxes.push_back({along, centre - 0.35, along + 1.4, centre + 0.35, 0.6, c.col});
        else          boxes.push_back({centre - 0.35, along, centre + 0.35, along + 1.4, 0.6, c.col});
    }
    if (t > 305) {                                              // a new building starts to rise
        double g = easeOutCubic((t - 305) / 20.0);
        boxes.push_back({-1.4, -1.4, 1.4, 1.4, 9.0 * g, Color{200, 225, 255}});
    }
    std::sort(boxes.begin(), boxes.end(), [&](const Box& a, const Box& b) {   // painter's algorithm
        double da = std::pow((a.x0 + a.x1) / 2 - eye.x, 2) + std::pow((a.z0 + a.z1) / 2 - eye.z, 2);
        double db = std::pow((b.x0 + b.x1) / 2 - eye.x, 2) + std::pow((b.z0 + b.z1) / 2 - eye.z, 2);
        return da > db;
    });
    for (const Box& b : boxes) drawBox(fb, vp, eye, b, bright);

    // smoke rising from every collapsed building (stateless particles: any frame can be rendered alone)
    for (size_t i = 0; i < w.buildings.size(); ++i) {
        const Building& b = w.buildings[i];
        if (b.collapseT > 1e8 || t < b.collapseT) continue;
        double cx = (b.box.x0 + b.box.x1) / 2, cz = (b.box.z0 + b.box.z1) / 2, hr = 0.12 * b.box.h;
        for (int k = 0; k < 5; ++k) {
            double tt = t - b.collapseT - 0.5 - k * 1.3;
            if (tt < 0 || tt > 120) continue;
            double age = std::fmod(tt, 7.0), f = age / 7.0;
            Vec3 p = {cx + 0.8 * age * std::cos(k + i), hr + age * 2.4, cz + 0.8 * age * std::sin(k + i)};
            drawParticle(fb, vp, p, 0.7 + 2.0 * f, Color{75, 72, 75}, 0.32 * (1 - f) * clamp01(tt));
        }
    }
    // missiles and explosions
    double flash = 0;
    for (size_t i = 0; i < w.missiles.size(); ++i) {
        const Missile& m = w.missiles[i];
        drawMissile(fb, vp, m, t);
        double age = t - (m.launchT + m.flightT);
        if (age >= 0 && age < 3.0) drawExplosion(fb, vp, {m.p3.x, 3.0, m.p3.z}, age, static_cast<int>(i));
        if (age >= 0 && age < 0.15) flash += 0.10 * (1 - age / 0.15);
    }
    if (flash > 0) fb.overlay(Color{255, 245, 220}, std::min(flash, 0.3));

    // fade in at the start, fade out at the end
    double fade = std::min(clamp01(t / 2.0), clamp01((TOTAL_TIME - t) / 4.0));
    if (fade < 1.0) fb.overlay(Color{0, 0, 0}, 1.0 - fade);
}

// Main Scene 2 
int main(int argc, char** argv) {
    World world;
    generateCity(world, 0, 0, N_MAIN, 0);
    generateCity(world, RIVAL_CENTRE.x, 0, N_RIVAL, 1);
    planWar(world);
    Framebuffer fb(W, H);
    std::string mode = argc > 1 ? argv[1] : "stills";

    if (mode == "still" && argc > 2) {
        renderFrame(fb, world, std::atof(argv[2]));
        fb.savePPM("scene2_still.ppm");
        std::puts("Saved scene2_still.ppm");
    } else if (mode == "seq" && argc > 4) {
        int fps = std::atoi(argv[2]); double from = std::atof(argv[3]), to = std::atof(argv[4]);
        int f0 = static_cast<int>(from * fps), f1 = static_cast<int>(to * fps);
        for (int f = f0; f <= f1; ++f) {
            renderFrame(fb, world, static_cast<double>(f) / fps);
            char name[64]; std::snprintf(name, sizeof name, "frame_%05d.ppm", f);
            fb.savePPM(name);
        }
        std::printf("Wrote %d frames\n", f1 - f0 + 1);
    } else if (mode == "pipe" && argc > 2) {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        int fps = std::atoi(argv[2]);
        double from = argc > 3 ? std::atof(argv[3]) : 0.0, to = argc > 4 ? std::atof(argv[4]) : TOTAL_TIME;
        for (int f = static_cast<int>(from * fps); f <= static_cast<int>(to * fps); ++f) {
            renderFrame(fb, world, static_cast<double>(f) / fps);
            fb.writePPM(stdout);
        }
    } else {
        const double times[] = {10, 45, 100, 150, 195, 225, 245, 262, 285, 318};
        for (double t : times) {
            auto t0 = std::chrono::steady_clock::now();
            renderFrame(fb, world, t);
            char name[64]; std::snprintf(name, sizeof name, "scene2_still_%03d.ppm", static_cast<int>(t));
            fb.savePPM(name);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("t=%6.1fs  %s  (%.0f ms)\n", t, name, ms);
        }
    }
    return 0;
}
