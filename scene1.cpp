// Checkpoint 1 - Pure C++ graphics foundation (no graphics libraries)
// Project: "Rise of a City" - animated time-lapse
// Build:  g++ -std=c++17 -O2 -Wall -Wextra checkpoint1.cpp -o cp1


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

// ---------------------------------------------------------------- Framebuffer
struct Color { uint8_t r, g, b; };

class Framebuffer {
public:
    int w, h;
    std::vector<Color> px;

    Framebuffer(int w_, int h_) : w(w_), h(h_), px(static_cast<size_t>(w_) * h_, Color{0, 0, 0}) {}

    void clear(Color c) { std::fill(px.begin(), px.end(), c); }

    void setPixel(int x, int y, Color c) {
        if (x < 0 || y < 0 || x >= w || y >= h) return;   // simple clipping
        px[static_cast<size_t>(y) * w + x] = c;
    }

    bool savePPM(const char* path) const {
        FILE* f = std::fopen(path, "wb");
        if (!f) return false;
        std::fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (const Color& c : px) {
            std::fputc(c.r, f); std::fputc(c.g, f); std::fputc(c.b, f);
        }
        std::fclose(f);
        return true;
    }
};

// ---------------------------------------------------------------- 2D primitives
// Bresenham line technique
void drawLine(Framebuffer& fb, int x0, int y0, int x1, int y1, Color c) {
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

// Midpoint circle outline
void drawCircle(Framebuffer& fb, int cx, int cy, int r, Color c) {
    int x = 0, y = r, d = 1 - r;
    while (x <= y) {
        fb.setPixel(cx + x, cy + y, c); fb.setPixel(cx - x, cy + y, c);
        fb.setPixel(cx + x, cy - y, c); fb.setPixel(cx - x, cy - y, c);
        fb.setPixel(cx + y, cy + x, c); fb.setPixel(cx - y, cy + x, c);
        fb.setPixel(cx + y, cy - x, c); fb.setPixel(cx - y, cy - x, c);
        if (d < 0) d += 2 * x + 3;
        else { d += 2 * (x - y) + 5; --y; }
        ++x;
    }
}

void fillCircle(Framebuffer& fb, int cx, int cy, int r, Color c) {
    for (int dy = -r; dy <= r; ++dy) {
        int half = static_cast<int>(std::sqrt(static_cast<double>(r * r - dy * dy)));
        for (int x = cx - half; x <= cx + half; ++x) fb.setPixel(x, cy + dy, c);
    } 
}

// Scanline polygon fill (even-odd rule)
struct P2 { int x, y; };

void fillPolygon(Framebuffer& fb, const std::vector<P2>& v, Color c) {
    if (v.size() < 3) return;
    int minY = v[0].y, maxY = v[0].y;
    for (const P2& p : v) { minY = std::min(minY, p.y); maxY = std::max(maxY, p.y); }
    for (int y = minY; y <= maxY; ++y) {
        std::vector<int> xs;
        for (size_t i = 0; i < v.size(); ++i) {
            P2 a = v[i], b = v[(i + 1) % v.size()];
            if (a.y == b.y) continue;                                   // skip horizontal edges
            if (y < std::min(a.y, b.y) || y >= std::max(a.y, b.y)) continue;
            double t = static_cast<double>(y - a.y) / (b.y - a.y);
            xs.push_back(static_cast<int>(std::lround(a.x + t * (b.x - a.x))));
        }
        std::sort(xs.begin(), xs.end());
        for (size_t i = 0; i + 1 < xs.size(); i += 2)
            for (int x = xs[i]; x <= xs[i + 1]; ++x) fb.setPixel(x, y, c);
    }
}

// ---------------------------------------------------------------- Vectors and matrices
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

    static Mat4 translate(double x, double y, double z) {
        Mat4 r = identity(); r.m[0][3] = x; r.m[1][3] = y; r.m[2][3] = z; return r;
    }
    static Mat4 scale(double x, double y, double z) {
        Mat4 r = identity(); r.m[0][0] = x; r.m[1][1] = y; r.m[2][2] = z; return r;
    }
    static Mat4 rotateX(double a) {
        Mat4 r = identity(); double c = std::cos(a), s = std::sin(a);
        r.m[1][1] = c; r.m[1][2] = -s; r.m[2][1] = s; r.m[2][2] = c; return r;
    }
    static Mat4 rotateY(double a) {
        Mat4 r = identity(); double c = std::cos(a), s = std::sin(a);
        r.m[0][0] = c; r.m[0][2] = s; r.m[2][0] = -s; r.m[2][2] = c; return r;
    }
    // OpenGL-style perspective: camera looks down -Z
    static Mat4 perspective(double fovy, double aspect, double n, double f) {
        Mat4 r; double t = 1.0 / std::tan(fovy / 2);
        r.m[0][0] = t / aspect; r.m[1][1] = t;
        r.m[2][2] = (f + n) / (n - f); r.m[2][3] = 2 * f * n / (n - f);
        r.m[3][2] = -1;
        return r;
    }
};

//  3D helpers
const double PI = 3.14159265358979323846;

struct Vec3 { double x, y, z; };
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec3 normalize(Vec3 v) { double l = std::sqrt(dot(v, v)); return {v.x / l, v.y / l, v.z / l}; }

// Camera matrix: eye position looking at a target
Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = normalize(target - eye), s = normalize(cross(f, up)), u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
    return r;
}

Color shade(Color c, double k) {
    auto f = [k](int v) { return static_cast<uint8_t>(std::min(255.0, std::max(0.0, v * k))); };
    return Color{f(c.r), f(c.g), f(c.b)};
}

// 3D -> screen
struct Screen { int x, y; bool ok; };

Screen project(const Mat4& vp, double x, double y, double z, int W, int H) {
    Vec4 c = vp * Vec4{x, y, z, 1.0};
    if (c.w <= 0.1) return {0, 0, false};                // behind the camera
    double nx = c.x / c.w, ny = c.y / c.w;               // perspective divide
    return { static_cast<int>(std::lround((nx + 1) * 0.5 * W)),
             static_cast<int>(std::lround((1 - ny) * 0.5 * H)), true };
}

// Flat polygon on the ground plane filled in screen space
void fillGround(Framebuffer& fb, const Mat4& vp, const std::vector<Vec3>& pts, Color c) {
    std::vector<P2> poly;
    for (const Vec3& p : pts) {
        Screen s = project(vp, p.x, p.y, p.z, fb.w, fb.h);
        if (!s.ok) return;
        poly.push_back({s.x, s.y});
    }
    fillPolygon(fb, poly, c);
}

void fillGroundRect(Framebuffer& fb, const Mat4& vp, double x0, double z0, double x1, double z1, Color c) {
    fillGround(fb, vp, {{x0, 0, z0}, {x1, 0, z0}, {x1, 0, z1}, {x0, 0, z1}}, c);
}

// ---------------------------------------------------------------- Solid boxes (buildings, trees)
struct Box { double x0, z0, x1, z1, h; Color base; };

// back-face culling ie (Draws the faces that point towards the camera)
// shaded by a fixed light direction, with Bresenham outlines.
void drawBox(Framebuffer& fb, const Mat4& vp, Vec3 eye, const Box& b) {
    const Vec3 L = normalize({0.45, 0.8, 0.35});
    struct Face { Vec3 p[4]; Vec3 n; };
    const double x0 = b.x0, x1 = b.x1, z0 = b.z0, z1 = b.z1, h = b.h;
    const Face faces[5] = {
        {{{x0, h, z0}, {x1, h, z0}, {x1, h, z1}, {x0, h, z1}}, { 0, 1,  0}},   // top
        {{{x1, 0, z0}, {x1, 0, z1}, {x1, h, z1}, {x1, h, z0}}, { 1, 0,  0}},   // +x side
        {{{x0, 0, z0}, {x0, 0, z1}, {x0, h, z1}, {x0, h, z0}}, {-1, 0,  0}},   // -x side
        {{{x0, 0, z1}, {x1, 0, z1}, {x1, h, z1}, {x0, h, z1}}, { 0, 0,  1}},   // +z side
        {{{x0, 0, z0}, {x1, 0, z0}, {x1, h, z0}, {x0, h, z0}}, { 0, 0, -1}},   // -z side
    };
    for (const Face& f : faces) {
        Vec3 centre{(f.p[0].x + f.p[2].x) / 2, (f.p[0].y + f.p[2].y) / 2, (f.p[0].z + f.p[2].z) / 2};
        if (dot(f.n, eye - centre) <= 0) continue;                 // facing away: skip
        std::vector<P2> poly; bool ok = true;
        for (const Vec3& p : f.p) {
            Screen s = project(vp, p.x, p.y, p.z, fb.w, fb.h);
            if (!s.ok) { ok = false; break; }
            poly.push_back({s.x, s.y});
        }
        if (!ok) continue;
        double k = 0.35 + 0.65 * std::max(0.0, dot(f.n, L));
        fillPolygon(fb, poly, shade(b.base, k));
        Color edge = shade(b.base, k * 0.55);
        for (size_t i = 0; i < 4; ++i)
            drawLine(fb, poly[i].x, poly[i].y, poly[(i + 1) % 4].x, poly[(i + 1) % 4].y, edge);
    }
}

//  Procedural city
static uint32_t g_seed = 12345;
double rnd() {                                    // small deterministic random generator
    g_seed = g_seed * 1664525u + 1013904223u;
    return static_cast<double>((g_seed >> 8) & 0xFFFF) / 65536.0;
}

const int    N     = 9;      // 9 x 9 city blocks
const double PITCH = 8.0;    // block + road
const double BLOCK = 6.0;    // block size
const double HALF  = N * PITCH / 2.0;   // half the city width (36)

// Fills 'boxes' with buildings/trees and draws the ground layer
void buildCity(Framebuffer& fb, const Mat4& vp, std::vector<Box>& boxes) {
    // roads: one large slab, dashed centre lines drawn on top
    fillGroundRect(fb, vp, -HALF - 1, -HALF - 1, HALF + 1, HALF + 1, Color{58, 58, 64});
    for (int k = 0; k <= N; ++k) {
        double c = -HALF + k * PITCH;
        for (double t = -HALF; t < HALF; t += 4.0) {
            Screen a = project(vp, c, 0.01, t, fb.w, fb.h), b = project(vp, c, 0.01, t + 2.0, fb.w, fb.h);
            if (a.ok && b.ok) drawLine(fb, a.x, a.y, b.x, b.y, Color{225, 200, 80});
            Screen d = project(vp, t, 0.01, c, fb.w, fb.h), e = project(vp, t + 2.0, 0.01, c, fb.w, fb.h);
            if (d.ok && e.ok) drawLine(fb, d.x, d.y, e.x, e.y, Color{225, 200, 80});
        }
    }
    const Color downtown[3] = {{110, 140, 190}, {95, 125, 170}, {135, 160, 200}};
    const Color midrise[3]  = {{205, 190, 165}, {190, 175, 150}, {215, 200, 180}};
    const Color suburb[3]   = {{230, 205, 180}, {220, 170, 140}, {235, 220, 200}};

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double bx = -HALF + i * PITCH + 1.0, bz = -HALF + j * PITCH + 1.0;
            double dist = std::sqrt((i - 4.0) * (i - 4.0) + (j - 4.0) * (j - 4.0));
            bool park = dist >= 2.5 && ((i * 7 + j * 13) % 11 == 0);

            if (park) {                                                    // park with trees
                fillGroundRect(fb, vp, bx, bz, bx + BLOCK, bz + BLOCK, Color{85, 160, 90});
                for (int t = 0; t < 7; ++t) {
                    double tx = bx + 0.5 + rnd() * (BLOCK - 1.6), tz = bz + 0.5 + rnd() * (BLOCK - 1.6);
                    boxes.push_back({tx, tz, tx + 0.9, tz + 0.9, 1.4 + rnd() * 1.2, Color{40, 115, 55}});
                }
                continue;
            }
            fillGroundRect(fb, vp, bx, bz, bx + BLOCK, bz + BLOCK, Color{150, 150, 148});   // pavement
            for (int a = 0; a < 2; ++a) {                                  // 2 x 2 lots per block
                for (int c = 0; c < 2; ++c) {
                    if (dist > 3.5 && rnd() < 0.10) continue;              // some empty lots at the edge
                    double lx = bx + a * 3.2, lz = bz + c * 3.2;           // lot size 2.8
                    double in0 = rnd() * 0.4, in1 = rnd() * 0.4;
                    double h; Color col;
                    if (dist < 1.5)      { h = 10 + rnd() * 11; col = downtown[static_cast<int>(rnd() * 3)]; }
                    else if (dist < 3.2) { h = 5 + rnd() * 6;   col = midrise[static_cast<int>(rnd() * 3)]; }
                    else                 { h = 2 + rnd() * 3.5; col = suburb[static_cast<int>(rnd() * 3)]; }
                    boxes.push_back({lx + in0, lz + in1, lx + 2.8 - in1, lz + 2.8 - in0, h, col});
                }
            }
        }
    }
}

// Main Scene
int main() {
    auto t0 = std::chrono::steady_clock::now();
    const int W = 960, H = 560;
    Framebuffer fb(W, H);

    // Sky: vertical gradient
    for (int y = 0; y < H; ++y) {
        double t = static_cast<double>(y) / H;
        Color c{ static_cast<uint8_t>(60 + 170 * t), static_cast<uint8_t>(110 + 100 * t),
                 static_cast<uint8_t>(190 + 40 * t) };
        for (int x = 0; x < W; ++x) fb.setPixel(x, y, c);
    }
    fillCircle(fb, 800, 85, 34, Color{255, 215, 110});      // sun
    drawCircle(fb, 800, 85, 36, Color{255, 240, 170});

    // Camera
    Vec3 eye{70, 48, 70};
    Mat4 view = lookAt(eye, Vec3{0, 4, 0}, Vec3{0, 1, 0});
    Mat4 proj = Mat4::perspective(36.0 * PI / 180.0, static_cast<double>(W) / H, 0.1, 400.0);
    Mat4 vp = proj * view;

    // Land and sea (sea edge uses a wavy shoreline)
    fillGroundRect(fb, vp, -60, -60, 60, 60, Color{105, 150, 90});
    std::vector<Vec3> sea;
    for (double z = -60; z <= 60; z += 4) sea.push_back({-46 + 3.0 * std::sin(z * 0.15), 0, z});
    sea.push_back({-60, 0, 60}); sea.push_back({-60, 0, -60});
    fillGround(fb, vp, sea, Color{70, 130, 190});

    // City ground layer, then solid boxes sorted back-to-front (painter's algorithm)
    std::vector<Box> boxes;
    buildCity(fb, vp, boxes);
    std::sort(boxes.begin(), boxes.end(), [&](const Box& a, const Box& b) {
        double da = std::pow((a.x0 + a.x1) / 2 - eye.x, 2) + std::pow((a.z0 + a.z1) / 2 - eye.z, 2);
        double db = std::pow((b.x0 + b.x1) / 2 - eye.x, 2) + std::pow((b.z0 + b.z1) / 2 - eye.z, 2);
        return da > db;
    });
    for (const Box& b : boxes) drawBox(fb, vp, eye, b);

    if (!fb.savePPM("scene1_scene.ppm")) { std::fprintf(stderr, "Could not write file\n"); return 1; }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Saved scene1_scene.ppm (%zu solids, %.0f ms)\n", boxes.size(), ms);
    return 0;
}
