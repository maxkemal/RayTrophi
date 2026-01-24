#include "ThreadLocalRNG.h"

ThreadLocalRNG::ThreadLocalRNG()
    : gen(std::hash<std::thread::id>{}(std::this_thread::get_id())),
    dis(0.0f, 1.0f) {
}

 float ThreadLocalRNG::get() {
    return dis(gen);
}
