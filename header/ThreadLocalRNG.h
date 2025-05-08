#ifndef THREAD_LOCAL_RNG_H
#define THREAD_LOCAL_RNG_H

#include <random>
#include <thread>

class ThreadLocalRNG {
private:
    std::mt19937 gen;  // Mersenne Twister 19937 generator
    std::uniform_real_distribution<float> dis;  // Uniform distribution for floats [0.0, 1.0)

public:
    ThreadLocalRNG();
    float get();
};

#endif // THREAD_LOCAL_RNG_H
