// No GPU needed: the default device choice over fake device lists.
#include "device_pick.hpp"
#include <cstdio>

using namespace strata::poly::device_pick;
int main() {
    int bad = 0;
    const auto check = [&](bool ok, const char* why) {
        if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); ++bad; }
    };
    const Dev igpu{"gfx1036", false}, igpu_flag{"gfx9999", true}, d9070{"gfx1201:sramecc-:xnack-", false},
        d7900{"gfx1100", false};
    Pick p = pick({igpu, d9070}, false);
    check(p.ids == std::vector<int>{1}, "iGPU first, discrete second: the discrete one");
    p = pick({igpu_flag, d9070}, false);
    check(p.ids == std::vector<int>{1}, "the integrated flag alone skips a device");
    p = pick({{"gfx1036:xnack-", false}, d9070}, false);
    check(p.ids == std::vector<int>{1}, "arch with features suffix is an APU");
    p = pick({d9070, igpu}, false);
    check(p.ids == std::vector<int>{0}, "discrete first, iGPU second");
    p = pick({igpu}, false);
    check(p.ids == std::vector<int>{0}, "only integrated: used");
    p = pick({d9070, d7900}, false);
    check(p.ids == std::vector<int>{0, 1}, "two discrete: both, in order");
    p = pick({igpu, d9070, d7900}, false);
    check(p.ids == std::vector<int>{1, 2}, "iGPU and two discrete");
    p = pick({igpu, d9070}, true);
    check(p.ids == std::vector<int>{0, 1}, "env set: nothing is skipped");
    check(!apu_arch("gfx10360") && !apu_arch("gfx11") && apu_arch("gfx1151"), "arch match is exact");
    check(pick({}, false).ids.empty(), "no devices");
    check(!p.why.empty(), "a reason is given");
    if (!bad) std::puts("poly_device_pick_test: ok");
    return bad ? 1 : 0;
}
