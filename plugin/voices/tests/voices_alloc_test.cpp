// No allocation after prepare(): global operator new/delete counted around trigger, setters and process of all
// four classes (the engine's allocation-hook idea).
#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "check.h"
#include "flybrain/voices.h"

static std::atomic<long> g_allocs{0}, g_frees{0};
static std::atomic<bool> g_count{false};
void* operator new(size_t n) { if (g_count.load()) ++g_allocs; void* p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void* operator new[](size_t n) { if (g_count.load()) ++g_allocs; void* p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void operator delete(void* p) noexcept { if (g_count.load()) ++g_frees; std::free(p); }
void operator delete[](void* p) noexcept { if (g_count.load()) ++g_frees; std::free(p); }
void operator delete(void* p, size_t) noexcept { if (g_count.load()) ++g_frees; std::free(p); }
void operator delete[](void* p, size_t) noexcept { if (g_count.load()) ++g_frees; std::free(p); }

using namespace flybrain::voices;

int main() {
    const int n = 512;
    std::vector<float> L(n), R(n);
    Kit kit; Sonification son; Sidechain sc; Limiter lim;
    kit.prepare(48000, n); son.prepare(48000, n); sc.prepare(48000, n); lim.prepare(48000, n);
    float z[8] = {1, 2, 3, 0, 5, 0.5f, 7, 8};
    g_allocs = 0; g_frees = 0; g_count = true;
    for (int block = 0; block < 200; ++block) {
        for (int i = 0; i < n; ++i) { L[i] = 0; R[i] = 0; }
        for (int d = 0; d < kDrums; ++d) kit.trigger(d, 0.9f, (block * 7 + d * 31) % n);
        if (block % 50 == 10) kit.setVoicing(block / 50);
        if (block % 30 == 0) { kit.setDrum(block % kDrums, -3.0f, false, 0.2f); kit.setVelocityCurve(0.3f); }
        if (block == 120) kit.reset();
        kit.process(L.data(), R.data(), n);
        son.setEnabled(true); son.setLevelDb(-12.0f); son.setGroupRates(z, 8); son.process(L.data(), R.data(), n);
        sc.setAmount(0.8f); sc.setMode(block % 2); sc.setGroupRate(z[block % 8]); sc.process(L.data(), R.data(), n);
        lim.process(L.data(), R.data(), n);
    }
    g_count = false;
    CHECK_MSG(g_allocs == 0 && g_frees == 0, "%ld allocations, %ld frees on the audio path", g_allocs.load(), g_frees.load());
    return finish();
}
