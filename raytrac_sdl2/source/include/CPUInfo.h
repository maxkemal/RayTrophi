/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          CPUInfo.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <iostream>
#include <array>
#include <bitset>
#include <vector>
#include <string>
#include <intrin.h>

class CPUInfo {
private:
    static constexpr int MAX_INTEL_TOP_LVL = 4;

    struct CPUData {
        std::string vendor;
        std::string brand;
        int family;
        int model;
        int stepping;
        std::string architecture;
        std::vector<std::string> features;
    };

    static CPUData data;

    static void detect_vendor() {
        std::array<int, 4> cpui;
        __cpuid(cpui.data(), 0);
        int ids = cpui[0];
        std::array<char, 48> vendor;
        *reinterpret_cast<int*>(vendor.data()) = cpui[1];
        *reinterpret_cast<int*>(vendor.data() + 4) = cpui[3];
        *reinterpret_cast<int*>(vendor.data() + 8) = cpui[2];
        data.vendor = std::string(vendor.data());
    }

    static void detect_brand() {
        std::array<char, 64> brand;
        std::array<int, 4> cpui;

        __cpuid(cpui.data(), 0x80000000);
        int highest = cpui[0];
        if (highest >= 0x80000004) {
            for (int i = 2; i <= 4; ++i) {
                __cpuid(cpui.data(), 0x80000000 + i);
                std::copy_n(reinterpret_cast<char*>(cpui.data()), 16, brand.data() + (i - 2) * 16);
            }
            data.brand = brand.data();
        }
        else {
            data.brand = "Unknown";
        }
    }

    static void detect_extended_info() {
        std::array<int, 4> cpui;
        __cpuid(cpui.data(), 1);
        data.family = ((cpui[0] >> 8) & 0xF) + ((cpui[0] >> 20) & 0xFF);
        data.model = ((cpui[0] >> 4) & 0xF) + ((cpui[0] >> 12) & 0xF0);
        data.stepping = cpui[0] & 0xF;
    }

    static void detect_architecture() {
        if (data.vendor.find("Intel") != std::string::npos) {
            if (data.family == 6) {
                switch (data.model) {
                case 0x8E: case 0x9E:
                    data.architecture = "Skylake"; break;
                case 0x9C: case 0x9D: case 0x9F:
                    data.architecture = "Jasper Lake"; break;
                case 0xA5: case 0xA6: case 0xA7:
                    data.architecture = "Comet Lake"; break;
                case 0xA8: case 0xA9: case 0xAA:
                    data.architecture = "Rocket Lake"; break;
                case 0x97: case 0x98: case 0x99: case 0x9A:
                    data.architecture = "Alder Lake"; break;
                case 0xB7: case 0xB8: case 0xBA: case 0xBB:
                    data.architecture = "Raptor Lake"; break;
                default:
                    data.architecture = "Unknown Intel Architecture";
                }
            }
        }
        else if (data.vendor.find("AMD") != std::string::npos) {
            if (data.family == 23) {
                if (data.model >= 1 && data.model <= 9)
                    data.architecture = "Zen";
                else if (data.model >= 49 && data.model <= 71)
                    data.architecture = "Zen+";
                else if (data.model >= 96 && data.model <= 127)
                    data.architecture = "Zen 2";
                else if (data.model >= 144 && data.model <= 159)
                    data.architecture = "Zen 3";
            }
            else if (data.family == 25) {
                if (data.model >= 1 && data.model <= 31)
                    data.architecture = "Zen 4";
            }
            else {
                data.architecture = "Unknown AMD Architecture";
            }
        }
        else {
            data.architecture = "Unknown Architecture";
        }
    }

    static void detect_features() {
        std::array<int, 4> cpui;
        __cpuid(cpui.data(), 1);

        std::bitset<32> f_1_ECX(cpui[2]);
        std::bitset<32> f_1_EDX(cpui[3]);

        if (f_1_ECX[0])  data.features.push_back("SSE3");
        if (f_1_ECX[9])  data.features.push_back("SSSE3");
        if (f_1_ECX[19]) data.features.push_back("SSE4.1");
        if (f_1_ECX[20]) data.features.push_back("SSE4.2");
        if (f_1_ECX[28]) data.features.push_back("AVX");
        if (f_1_EDX[25]) data.features.push_back("SSE");
        if (f_1_EDX[26]) data.features.push_back("SSE2");

        __cpuid(cpui.data(), 7);
        std::bitset<32> f_7_EBX(cpui[1]);

        if (f_7_EBX[5])  data.features.push_back("AVX2");
        if (f_7_EBX[16]) data.features.push_back("AVX512F");
        if (f_7_EBX[26]) data.features.push_back("AVX512PF");
        if (f_7_EBX[27]) data.features.push_back("AVX512ER");
        if (f_7_EBX[28]) data.features.push_back("AVX512CD");
    }

public:
    static void initialize() {
        detect_vendor();
        detect_brand();
        detect_extended_info();
        detect_architecture();
        detect_features();
    }

    static void print_cpu_info() {
        std::cout << "CPU Vendor: " << data.vendor << std::endl;
        std::cout << "CPU Brand: " << data.brand << std::endl;
        std::cout << "CPU Family: " << data.family << std::endl;
        std::cout << "CPU Model: " << data.model << std::endl;
        std::cout << "CPU Stepping: " << data.stepping << std::endl;
        std::cout << "CPU Architecture: " << data.architecture << std::endl;
        std::cout << "CPU Features: ";
        for (const auto& feature : data.features) {
            std::cout << feature << " ";
        }
        std::cout << std::endl;
    }

    static bool has_feature(const std::string& feature) {
        return std::find(data.features.begin(), data.features.end(), feature) != data.features.end();
    }

    static const std::string& get_vendor() { return data.vendor; }
    static const std::string& get_brand() { return data.brand; }
    static int get_family() { return data.family; }
    static int get_model() { return data.model; }
    static int get_stepping() { return data.stepping; }
    static const std::string& get_architecture() { return data.architecture; }
    static const std::vector<std::string>& get_features() { return data.features; }
};

// Static member definition
CPUInfo::CPUData CPUInfo::data;
