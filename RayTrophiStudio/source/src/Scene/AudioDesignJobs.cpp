#include "Audio/AudioDesignJobs.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace audio {
namespace {
struct Cancelled {};
}

struct AudioDesignJobs::Impl {
    struct Request {
        std::uint64_t id;
        std::uint64_t revision;
        Task task;
    };
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::atomic<std::uint64_t> latest{0};
    bool shutdown = false;
    std::optional<Request> pending;
    std::unique_ptr<Result> result;
    nlohmann::json report = {{"state", "idle"}, {"id", 0}};
    std::thread worker;

    Impl() = default;

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            shutdown = true;
            ++latest;
            pending.reset();
        }
        wake.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void run() {
        for (;;) {
            std::unique_lock lock(mutex);
            wake.wait(lock, [this] { return shutdown || pending.has_value(); });
            if (shutdown) {
                return;
            }
            auto request = std::move(*pending);
            pending.reset();
            report["state"] = "running";
            lock.unlock();
            const auto check = [this, id = request.id] {
                if (latest.load() != id) {
                    throw Cancelled{};
                }
            };
            try {
                check();
                auto value = std::make_unique<Result>(request.task(check));
                check();
                value->id = request.id;
                value->revision = request.revision;
                lock.lock();
                if (latest.load() == request.id) {
                    result = std::move(value);
                    report["state"] = "completed";
                }
            } catch (const Cancelled&) {
                // Cancellation/newer ownership was already published by the caller.
            } catch (const std::exception& error) {
                if (!lock.owns_lock()) {
                    lock.lock();
                }
                if (latest.load() == request.id) {
                    report["state"] = "failed";
                    report["error"] = error.what();
                }
            } catch (...) {
                if (!lock.owns_lock()) {
                    lock.lock();
                }
                if (latest.load() == request.id) {
                    report["state"] = "failed";
                    report["error"] = "Unknown audio worker failure";
                }
            }
        }
    }
};

AudioDesignJobs::AudioDesignJobs() : impl_(std::make_unique<Impl>()) {}
AudioDesignJobs::~AudioDesignJobs() = default;

void AudioDesignJobs::start(const std::string& kind, std::uint64_t revision, Task task) {
    std::lock_guard lock(impl_->mutex);
    if (!impl_->worker.joinable()) {
        impl_->worker = std::thread([state = impl_.get()] { state->run(); });
    }
    const auto id = ++impl_->latest;
    impl_->result.reset();
    impl_->pending = Impl::Request{id, revision, std::move(task)};
    impl_->report = {{"id", id}, {"kind", kind}, {"revision", revision}, {"state", "queued"}};
    impl_->wake.notify_one();
}

void AudioDesignJobs::cancel() {
    std::lock_guard lock(impl_->mutex);
    ++impl_->latest;
    impl_->pending.reset();
    impl_->result.reset();
    impl_->report["state"] = "cancelled";
}

nlohmann::json AudioDesignJobs::status() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->report;
}

std::unique_ptr<AudioDesignJobs::Result> AudioDesignJobs::take() {
    std::lock_guard lock(impl_->mutex);
    auto result = std::move(impl_->result);
    if (result) {
        impl_->report["state"] = "consumed";
    }
    return result;
}
} // namespace audio
