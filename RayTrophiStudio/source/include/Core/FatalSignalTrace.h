#pragma once

// Best-effort fatal-signal evidence, independent of the buffered scene logger.
void writeFatalSignalTrace(int signal) noexcept;
