// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Profile enum only — no ORT. gpu_routing_policy.h / gpu_routing_tables.h include this
// instead of gpu_info.h so select_backend can be compiled in isolation (tools/test_gpu_routing.cc).

namespace gpu_ep {

enum class Profile {
    Auto,
    Eager,
    Optimized,
    MIGraphX,
    DirectX,
    Hip
};

}  // namespace gpu_ep
