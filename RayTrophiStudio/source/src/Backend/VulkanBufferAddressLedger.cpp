#include "Backend/VulkanBufferAddressLedger.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace VulkanRT {
namespace BufferLedger {
namespace {

constexpr int kFrames = 8;
// Destroyed records kept for use-after-free lookups. A loss is usually seen
// within a frame or two of the free, and a frame allocates few hundred buffers
// at most (staging included).
constexpr size_t kDestroyedRing = 16384;

struct Record {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    uint64_t address = 0;
    uint64_t size = 0;
    uint64_t createSerial = 0;
    uint64_t destroySerial = 0;   // 0 = live
    unsigned short frames = 0;
    void* stack[kFrames] = {};
};

std::mutex g_mutex;
std::unordered_map<VkBuffer, Record> g_live;
std::deque<Record> g_destroyed;
uint64_t g_serial = 0;

std::string hex(uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
    return buf;
}

std::string symbolize(const Record& r) {
    static bool initialized = false;
    HANDLE process = GetCurrentProcess();
    if (!initialized) {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitialize(process, nullptr, TRUE);   // fails harmlessly if already done
        initialized = true;
    }
    std::string out;
    alignas(SYMBOL_INFO) char storage[sizeof(SYMBOL_INFO) + 256];
    for (unsigned i = 0; i < r.frames; ++i) {
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(storage);
        std::memset(storage, 0, sizeof(storage));
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 255;
        DWORD64 displacement = 0;
        const DWORD64 pc = reinterpret_cast<DWORD64>(r.stack[i]);
        out += "\n        #" + std::to_string(i) + " ";
        if (SymFromAddr(process, pc, &displacement, symbol)) {
            out += symbol->Name;
            IMAGEHLP_LINE64 line{};
            line.SizeOfStruct = sizeof(line);
            DWORD lineDisp = 0;
            if (SymGetLineFromAddr64(process, pc, &lineDisp, &line)) {
                const char* file = line.FileName;
                if (const char* slash = std::strrchr(file, '\\')) file = slash + 1;
                out += std::string(" (") + file + ":" + std::to_string(line.LineNumber) + ")";
            }
        } else {
            out += hex(pc);
        }
    }
    return out;
}

std::string line(const Record& r, uint64_t address) {
    std::string s = r.destroySerial ? "  DESTROYED buffer " : "  LIVE buffer ";
    s += hex(r.address) + ".." + hex(r.address + r.size) + " size=" + std::to_string(r.size) +
         " fault_offset=" + std::to_string(static_cast<long long>(address - r.address)) +
         " created#" + std::to_string(r.createSerial);
    if (r.destroySerial) {
        s += " destroyed#" + std::to_string(r.destroySerial) + " (" +
             std::to_string(g_serial - r.destroySerial) + " ledger events ago)";
    }
    s += " created by:" + symbolize(r);
    return s;
}

} // namespace

void noteCreated(VkDevice device, VkBuffer buffer, uint64_t address, uint64_t size) {
    if (!buffer || !address) return;
    Record r;
    r.device = device;
    r.buffer = buffer;
    r.address = address;
    r.size = size;
    // Skip this frame and VulkanDevice::createBuffer: frame 0 is the caller.
    r.frames = CaptureStackBackTrace(2, kFrames, r.stack, nullptr);
    std::lock_guard<std::mutex> lock(g_mutex);
    r.createSerial = ++g_serial;
    g_live[buffer] = r;
}

void noteDestroyed(VkDevice device, VkBuffer buffer) {
    if (!buffer) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_live.find(buffer);
    if (it == g_live.end() || it->second.device != device) return;
    Record r = it->second;
    g_live.erase(it);
    r.destroySerial = ++g_serial;
    g_destroyed.push_back(r);
    if (g_destroyed.size() > kDestroyedRing) g_destroyed.pop_front();
}

void forgetDevice(VkDevice device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto it = g_live.begin(); it != g_live.end();) {
        if (it->second.device == device) it = g_live.erase(it);
        else ++it;
    }
    std::deque<Record> kept;
    for (const auto& r : g_destroyed) if (r.device != device) kept.push_back(r);
    g_destroyed.swap(kept);
}

std::string describe(VkDevice device, uint64_t address, uint64_t slack) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string out;
    int matches = 0;
    auto consider = [&](const Record& r) {
        if (r.device != device) return;
        const uint64_t lo = r.address;
        const uint64_t hi = r.address + r.size;
        if (address >= lo && address < hi + slack) {
            out += "\n" + line(r, address);
            ++matches;
        }
    };
    for (const auto& kv : g_live) consider(kv.second);
    // Newest frees first: the most recent owner of a reused range is the suspect.
    for (auto it = g_destroyed.rbegin(); it != g_destroyed.rend() && matches < 6; ++it) consider(*it);
    if (!matches) {
        out = "\n  no live or recently destroyed buffer covers this address "
              "(not a createBuffer allocation: image, AS-internal or older than the ring); live=" +
              std::to_string(g_live.size()) + " destroyed_ring=" + std::to_string(g_destroyed.size());
    }
    return out;
}

} // namespace BufferLedger
} // namespace VulkanRT
