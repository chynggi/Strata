// src/poly/device_pick.hpp - Which GPUs strata-poly uses by default on AMD (HIP).
// Some setups list the Ryzen iGPU (gfx1036, gfx1103, ...) as device 0 in front of the real card (WSL with ROCm 10.2),
// and the engine's default device is 0.  So by default an integrated device is left out when a discrete one exists.
// A choice made with HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES is kept exactly as it is.
// Pure logic: no GPU needed (device_pick_test.cpp).
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::poly::device_pick {

struct Dev {
    std::string arch;         // gcnArchName, "gfx1036:sramecc-:xnack-" or "gfx1036"
    bool integrated = false;  // hipDeviceProp_t.integrated
};

struct Pick {
    std::vector<int> ids;  // the devices to use, in order: ids[0] is the primary one
    std::string why;       // one line for the log
};

// The APUs (Ryzen iGPUs) whose flag may read 0 under some drivers.
inline bool apu_arch(const std::string& arch) {
    for (const char* a : {"gfx90c", "gfx1035", "gfx1036", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153"})
        if (arch.compare(0, std::strlen(a), a) == 0 && (arch.size() == std::strlen(a) || arch[std::strlen(a)] == ':'))
            return true;
    return false;
}

inline bool is_integrated(const Dev& d) { return d.integrated || apu_arch(d.arch); }

// `env_set`: one of the *_VISIBLE_DEVICES variables is set (non-empty).
inline Pick pick(const std::vector<Dev>& devs, bool env_set) {
    Pick p;
    const int n = (int) devs.size();
    for (int i = 0; i < n; ++i) p.ids.push_back(i);
    if (n == 0) {
        p.why = "no device";
        return p;
    }
    if (env_set) {
        p.why = "the *_VISIBLE_DEVICES variable is set, so all " + std::to_string(n) + " visible device(s) are kept";
        return p;
    }
    std::vector<int> discrete;
    for (int i = 0; i < n; ++i)
        if (!is_integrated(devs[(size_t) i])) discrete.push_back(i);
    if (discrete.empty()) {
        p.why = "only integrated device(s) found, using them";
    } else if ((int) discrete.size() == n) {
        p.why = "no integrated device, all " + std::to_string(n) + " device(s) are kept";
    } else {
        p.ids = discrete;
        p.why = "skipped " + std::to_string(n - (int) discrete.size()) +
                " integrated device(s); set HIP_VISIBLE_DEVICES to choose by hand";
    }
    return p;
}

inline bool env_selects_devices() {
    for (const char* v : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
        const char* s = std::getenv(v);
        if (s && *s) return true;
    }
    return false;
}

}  // namespace strata::poly::device_pick
