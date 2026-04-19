// SafeSEH.h - lightweight SEH logging helper for startup crash diagnostics
#pragma once
#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#include <stdio.h>
#include <stdlib.h>
#include <psapi.h>
#pragma comment(lib,"Dbghelp.lib")

// Log and symbol resolution helper used by the vectored handler and __except.
inline LONG WINAPI LogSEHAndWrite(EXCEPTION_POINTERS* ep, const char* tag)
{
    FILE* f = fopen("StartupCrash_SEH.log", "a");
    if (!f) return EXCEPTION_EXECUTE_HANDLER;

    if (!ep || !ep->ExceptionRecord) {
        fclose(f);
        return EXCEPTION_EXECUTE_HANDLER;
    }

    DWORD code = (DWORD)ep->ExceptionRecord->ExceptionCode;
    void* addr = ep->ExceptionRecord->ExceptionAddress;
    DWORD tid = GetCurrentThreadId();
    fprintf(f, "[SEH] tid=%u tag=%s code=0x%08X addr=%p\n", tid, tag ? tag : "(null)", code, addr);

    if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2) {
        ULONG_PTR violationType = ep->ExceptionRecord->ExceptionInformation[0];
        void* faultAddr = (void*)ep->ExceptionRecord->ExceptionInformation[1];
        const char* vt = (violationType == 0) ? "READ" : (violationType == 1) ? "WRITE" : (violationType == 8) ? "EXEC" : "UNKNOWN";
        fprintf(f, "  AccessViolation type=%s fault_addr=%p\n", vt, faultAddr);
    }

    void* frames[64];
    USHORT count = CaptureStackBackTrace(0, _countof(frames), frames, NULL);
    for (USHORT i = 0; i < count; ++i) {
        fprintf(f, "  %02d  [0x%p]\n", i, frames[i]);
    }

    fclose(f);

    // Avoid forcing a debug break inside the vectored handler — when a debugger
    // is attached Visual Studio will already report first-chance exceptions.
    // Do not call DebugBreak() here to prevent unnecessary breakpoints.
    return EXCEPTION_EXECUTE_HANDLER;
}

// Vectored exception handler at file scope (avoids nested function definitions).
static LONG CALLBACK VectoredSEHHandler(PEXCEPTION_POINTERS ep)
{
    if (!ep || !ep->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    DWORD code = (DWORD)ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_BREAKPOINT || code == EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
    LogSEHAndWrite(ep, "VectoredHandler");
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void InitSEHLogging()
{
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), NULL, TRUE);
    AddVectoredExceptionHandler(1, VectoredSEHHandler);
}

// Convenience macros for wrapping critical blocks in code that may crash at startup.
#define CRITICAL_BLOCK(name) __try {
#define END_CRITICAL_BLOCK(name) } __except(LogSEHAndWrite(GetExceptionInformation(), name)) {}

#endif // _WIN32
