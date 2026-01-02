#pragma once
#include <Vec3.h>
// Perlin.h
class Perlin {
private:
    static const int POINT_COUNT = 256;
    Vec3* ranvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    static int* perlin_generate_perm() {
        auto p = new int[POINT_COUNT];
        for (int i = 0; i < POINT_COUNT; i++)
            p[i] = i;
        permute(p, POINT_COUNT);
        return p;
    }

    static void permute(int* p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = (int)Vec3::random_float(0, (float)i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }

    static float trilinear_interp(float c[2][2][2], float u, float v, float w) {
        float accum = 0.0f;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++)
                    accum += (i * u + (1 - i) * (1 - u)) *
                    (j * v + (1 - j) * (1 - v)) *
                    (k * w + (1 - k) * (1 - w)) * c[i][j][k];
        return accum;
    }

    static float perlin_interp(Vec3 c[2][2][2], float u, float v, float w) {
        float uu = u * u * (3 - 2 * u);
        float vv = v * v * (3 - 2 * v);
        float ww = w * w * (3 - 2 * w);
        float accum = 0.0f;

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                for (int k = 0; k < 2; k++) {
                    Vec3 weight_v((float)(u - i), (float)(v - j), (float)(w - k));
                    accum += (i * uu + (1 - i) * (1 - uu))
                        * (j * vv + (1 - j) * (1 - vv))
                        * (k * ww + (1 - k) * (1 - ww))
                        * Vec3::dot(c[i][j][k], weight_v);
                }
        return accum;
    }

public:
    Perlin() {
        ranvec = new Vec3[POINT_COUNT];
        for (int i = 0; i < POINT_COUNT; ++i) {
            ranvec[i] = unit_vector(Vec3::random(-1, 1));
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }

    ~Perlin() {
        delete[] ranvec;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
    }

    float noise(const Vec3& p) const {
        float u = p.x - floor(p.x);
        float v = p.y - floor(p.y);
        float w = p.z - floor(p.z);

        int i = (int)floor(p.x);
        int j = (int)floor(p.y);
        int k = (int)floor(p.z);
        Vec3 c[2][2][2];

        for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
                for (int dk = 0; dk < 2; dk++)
                    c[di][dj][dk] = ranvec[
                        perm_x[(i + di) & 255] ^
                            perm_y[(j + dj) & 255] ^
                            perm_z[(k + dk) & 255]
                    ];

        return perlin_interp(c, u, v, w);
    }

    float turb(const Vec3& p, int depth = 7) const {
        float accum = 0.0f;
        Vec3 temp_p = p;
        float weight = 1.0f;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2;
        }

        return fabs(accum);
    }
};

