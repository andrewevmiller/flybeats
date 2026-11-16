// No allocation on the audio path: global operator new/delete are counted around process(), flush() and
// stepFeatures(). Also checks that the control swap and the runaway reset allocate nothing.
#include <atomic>
#include <cstdlib>
#include <new>

#include "data.h"

static std::atomic<long> g_allocs{0}, g_frees{0};
static std::atomic<bool> g_count{false};
void* operator new(size_t n) { if (g_count.load()) ++g_allocs; void* p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void* operator new[](size_t n) { if (g_count.load()) ++g_allocs; void* p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void operator delete(void* p) noexcept { if (g_count.load()) ++g_frees; std::free(p); }
void operator delete[](void* p) noexcept { if (g_count.load()) ++g_frees; std::free(p); }
void operator delete(void* p, size_t) noexcept { if (g_count.load()) ++g_frees; std::free(p); }
void operator delete[](void* p, size_t) noexcept { if (g_count.load()) ++g_frees; std::free(p); }

int main(int, char** argv) {
    std::string dir = argv[1];
    if (!exists(dir + "/small.flybrain")) { std::printf("SKIP: no test data\n"); return 0; }
    BrainFile b = BrainFile::load(dir + "/small.flybrain");
    b.runaway.set = true; b.runaway.rateCeiling = 1e3; b.runaway.meanRateLimit = 1.0; b.runaway.frames = 5;
    std::vector<float> audio = loadRaw<float>(dir + "/audio16k.f32");
    for (bool dbl : {false, true}) {
        for (int rate : {16000, 44100, 48000}) {
            Engine e(b); EngineConfig cfg; cfg.hostRate = rate; cfg.maxBlock = 512; cfg.doublePrecision = dbl; cfg.prerollFrames = 0; e.prepare(cfg);
            Controls c(b); c.gainScale = 50.0; Injection inj; inj.neuron = 3; inj.current = 1.0f; inj.frames = 10; c.addInjection(inj);
            e.setControls(c);                     // compiled off the audio thread; the swap happens inside process()
            std::vector<HitEvent> ev(256);
            std::vector<double> feats(16, 0.1);
            g_allocs = 0; g_frees = 0; g_count = true;
            for (size_t p = 0; p + 512 <= audio.size(); p += 512) { const float* ch[1] = {audio.data() + p}; e.process(ch, 1, 512, ev.data(), 256); }
            e.flush(ev.data(), 256);
            e.reset();
            for (int f = 0; f < 50; ++f) e.stepFeatures(feats.data(), ev.data(), 256);
            g_count = false;
            CHECK_MSG(g_allocs == 0 && g_frees == 0, "%s %d Hz: %ld allocations, %ld frees on the audio path (runaways %d)", dbl ? "double" : "float", rate, g_allocs.load(), g_frees.load(), e.runawayCount());
            CHECK_MSG(e.runawayCount() > 0, "the guard did not trigger during the allocation test");
        }
    }
    std::printf("no allocation in process/flush/reset/stepFeatures across float/double and 16k/44.1k/48k, with a control swap and runaway resets\n");
    return finish();
}
