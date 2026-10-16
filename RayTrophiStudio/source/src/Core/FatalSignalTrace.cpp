#include "Core/FatalSignalTrace.h"

#ifdef _WIN32
#include <Windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstring>
#include <cwchar>
#pragma comment(lib, "dbghelp.lib")

namespace {
volatile LONG writing = 0;

void append(HANDLE file, const char* text) {
    DWORD written = 0;
    WriteFile(file, text, static_cast<DWORD>(std::strlen(text)), &written, nullptr);
}
}

void writeFatalSignalTrace(int signal) noexcept {
    if (InterlockedCompareExchange(&writing, 1, 0) != 0) {
        return;
    }
    wchar_t path[MAX_PATH]{};
    const auto length = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        return;
    }
    auto* separator = std::wcsrchr(path, L'\\');
    if (!separator || separator - path > MAX_PATH - 40) {
        return;
    }
    const auto available = static_cast<std::size_t>(MAX_PATH - (separator + 1 - path));
    wcscpy_s(separator + 1, available, L"StartupCrashStack.txt");
    HANDLE file = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ, nullptr,
        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    char line[1200]{};
    SYSTEMTIME now{};
    GetLocalTime(&now);
    std::snprintf(line, sizeof(line),
        "\r\n[%04u-%02u-%02u %02u:%02u:%02u] signal=%d pid=%lu tid=%lu\r\n",
        now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond,
        signal, GetCurrentProcessId(), GetCurrentThreadId());
    append(file, line);
    void* frames[64]{};
    const auto count = CaptureStackBackTrace(0, 64, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module = nullptr;
        wchar_t wideName[MAX_PATH]{};
        char name[900]{};
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(frames[i]), &module);
        if (module) {
            GetModuleFileNameW(module, wideName, MAX_PATH);
            WideCharToMultiByte(CP_UTF8, 0, wideName, -1, name, sizeof(name), nullptr, nullptr);
        }
        const auto offset = reinterpret_cast<ULONG_PTR>(frames[i]) -
            reinterpret_cast<ULONG_PTR>(module);
        std::snprintf(line, sizeof(line), "#%u %p %s +0x%llx\r\n", i, frames[i],
            module ? name : "unknown", static_cast<unsigned long long>(offset));
        append(file, line);
    }
    FlushFileBuffers(file);
    CloseHandle(file);

    // Capture current-thread context so the dump is useful even without a CRT
    // exception. Raw module-relative addresses above survive dump-writer failure.
    wcscpy_s(separator + 1, available, L"StartupCrashSignal.dmp");
    HANDLE dump = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (dump != INVALID_HANDLE_VALUE) {
        CONTEXT context{};
        RtlCaptureContext(&context);
        EXCEPTION_RECORD record{};
        record.ExceptionCode = 0xE0000000u | static_cast<DWORD>(signal & 0xFFFF);
        record.ExceptionAddress = count ? frames[0] : nullptr;
        EXCEPTION_POINTERS pointers{&record, &context};
        MINIDUMP_EXCEPTION_INFORMATION info{GetCurrentThreadId(), &pointers, FALSE};
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), dump,
            MiniDumpNormal, &info, nullptr, nullptr);
        CloseHandle(dump);
    }
}
#else
void writeFatalSignalTrace(int) noexcept {}
#endif
