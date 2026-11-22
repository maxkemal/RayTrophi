#pragma once
// Self-contained SHA-256 (FIPS 180-4). Replaces OpenSSL EVP in the IPC token and session
// code so the Android Player does not need libcrypto. The digest is the standard one, so
// token hashes already stored in ipc_tokens.json stay valid.
// TLS transport (Windows only) still uses OpenSSL for the handshake itself.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace rtutil {

class Sha256 {
public:
    using Digest = std::array<std::uint8_t, 32>;

    Sha256() { reset(); }

    void reset() {
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        total_bytes_ = 0;
        buffered_ = 0;
    }

    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        total_bytes_ += size;
        if (buffered_ > 0) {
            const std::size_t take = (size < 64 - buffered_) ? size : 64 - buffered_;
            std::memcpy(buffer_.data() + buffered_, bytes, take);
            buffered_ += take;
            bytes += take;
            size -= take;
            if (buffered_ < 64) return;
            compress(buffer_.data());
            buffered_ = 0;
        }
        while (size >= 64) {
            compress(bytes);
            bytes += 64;
            size -= 64;
        }
        if (size > 0) {
            std::memcpy(buffer_.data(), bytes, size);
            buffered_ = size;
        }
    }

    Digest finish() {
        const std::uint64_t bit_length = total_bytes_ * 8u;
        const std::uint8_t pad_start = 0x80u;
        update(&pad_start, 1);
        const std::uint8_t zero = 0;
        while (buffered_ != 56) update(&zero, 1);
        std::uint8_t length_bytes[8];
        for (int i = 0; i < 8; ++i)
            length_bytes[i] = static_cast<std::uint8_t>(bit_length >> (56 - 8 * i));
        update(length_bytes, 8);
        Digest digest{};
        for (int i = 0; i < 8; ++i) {
            digest[i * 4 + 0] = static_cast<std::uint8_t>(state_[i] >> 24);
            digest[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
            digest[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
            digest[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        reset();
        return digest;
    }

    static Digest hash(const void* data, std::size_t size) {
        Sha256 hasher;
        hasher.update(data, size);
        return hasher.finish();
    }

    static std::string hexDigest(const std::string& value) {
        const Digest digest = hash(value.data(), value.size());
        static constexpr char digits[] = "0123456789abcdef";
        std::string result(64, '0');
        for (std::size_t i = 0; i < digest.size(); ++i) {
            result[i * 2] = digits[digest[i] >> 4u];
            result[i * 2 + 1] = digits[digest[i] & 0x0fu];
        }
        return result;
    }

private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress(const std::uint8_t* block) {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
            0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
            0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
            0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
            0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t(block[i * 4]) << 24) | (std::uint32_t(block[i * 4 + 1]) << 16) |
                   (std::uint32_t(block[i * 4 + 2]) << 8) | std::uint32_t(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t total_bytes_ = 0;
    std::size_t buffered_ = 0;
};

} // namespace rtutil
