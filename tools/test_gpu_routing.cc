// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone check of gpu_routing_policy.h / gpu_routing_tables.h. Those headers
// depend only on gpu_profile.h (Profile enum) plus STL — no amdgpu-ep rebuild, no
// ORT, no HIP runtime.
//
// From the repo root (Developer PowerShell, cl on PATH):
//   cl /nologo /std:c++17 /EHsc /DUSE_HIP /I src/amdgpu tools/test_gpu_routing.cc ^
//      /Fe:tools/test_gpu_routing.exe && tools\test_gpu_routing.exe
//
// /DUSE_HIP is required so the HIP Auto-route branch is live.

#include "gpu_routing_policy.h"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>

using gpu_ep::fnv1a;
using gpu_ep::kNoModelArch;
using gpu_ep::model_arch_hash;
using gpu_ep::Profile;
using gpu_ep::select_backend;

namespace {

const char* profile_name(Profile p) {
    switch (p) {
        case Profile::MIGraphX:  return "MIGraphX";
        case Profile::DirectX:   return "DirectX";
        case Profile::Hip:       return "Hip";
        case Profile::Eager:     return "Eager";
        case Profile::Optimized: return "Optimized";
        case Profile::Auto:      return "Auto";
    }
    return "?";
}

int g_failures = 0;

void require(bool ok, std::string_view what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << '\n';
}

void require_eq(Profile got, Profile want, std::string_view what) {
    if (got == want) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << " got " << profile_name(got)
              << " want " << profile_name(want) << '\n';
}

Profile auto_route(std::string_view gfx, const std::optional<std::string>& model_arch,
                   bool webnn = false) {
    return select_backend(gfx, model_arch_hash(model_arch), webnn, Profile::Auto);
}

}  // namespace

int main() {
#ifndef USE_HIP
    std::cerr << "FAIL: compile with /DUSE_HIP so the HIP branch is tested\n";
    return 1;
#endif

    require(model_arch_hash(std::nullopt) == kNoModelArch, "omitted model_arch is kNoModelArch");
    require(model_arch_hash(std::optional<std::string>{""}) == kNoModelArch,
            "empty model_arch is kNoModelArch");
    require(model_arch_hash(std::optional<std::string>{"   "}) == kNoModelArch,
            "whitespace model_arch is kNoModelArch");
    require(model_arch_hash(std::optional<std::string>{"llama"}) == fnv1a("llama"),
            "llama hashes as normalized llama");
    require(model_arch_hash(std::optional<std::string>{" Llama "}) == fnv1a("llama"),
            "padded Llama normalizes then hashes");

    const std::optional<std::string> llama{"llama"};
    const std::optional<std::string> none{};
    const std::optional<std::string> spaces{"  "};

    require_eq(auto_route("gfx1150", llama), Profile::Hip, "Strix/GPT1 gfx1150 + llama");
    require_eq(auto_route("gfx1151", llama), Profile::Hip, "Halo gfx1151 + llama");
    require_eq(auto_route("gfx1152", llama), Profile::Hip, "Krackan/GPT2 gfx1152 + llama");
    require_eq(auto_route("gfx1153", llama), Profile::Hip, "GPT3/Krackan2e gfx1153 + llama");
    require_eq(auto_route("gfx1153:xnack-", llama), Profile::Hip, "gfx1153 suffix + llama");
    require_eq(auto_route("gfx1150:xnack-", llama), Profile::Hip, "gfx1150 suffix + llama");

    require_eq(auto_route("gfx1150", none), Profile::MIGraphX, "gfx1150 + omitted model_arch");
    require_eq(auto_route("gfx1152", none), Profile::MIGraphX, "gfx1152 + omitted model_arch");
    require_eq(auto_route("gfx1153", none), Profile::MIGraphX, "gfx1153 + omitted model_arch");
    require_eq(auto_route("gfx1150", spaces), Profile::MIGraphX, "gfx1150 + whitespace model_arch");

    require_eq(auto_route("gfx1170", llama), Profile::MIGraphX,
               "Medusa gfx117 must not match gfx115");
    require_eq(auto_route("gfx1201", llama), Profile::MIGraphX, "Navi48 + llama");
    require_eq(auto_route("gfx1100", llama), Profile::MIGraphX, "RDNA3 gfx1100 + llama");

    require_eq(auto_route("gfx1150", llama, /*webnn=*/true), Profile::DirectX,
               "WebNN on gfx1150 wins over HIP");

    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
