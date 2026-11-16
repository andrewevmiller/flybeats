// flybrain_bench: microseconds per frame and percent of one core at 200 frames/s, float and double, silence and noise.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "flybrain/engine.h"

using namespace flybrain;

static double run(const BrainFile& brain, bool dbl, bool noise, int seconds, int rate, int block) {
    Engine e(brain);
    EngineConfig cfg; cfg.hostRate = rate; cfg.maxBlock = block; cfg.doublePrecision = dbl; cfg.prerollFrames = 0;
    e.prepare(cfg);
    std::vector<float> buf(static_cast<size_t>(block), 0.0f);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    const float* ch[1] = {buf.data()};
    std::vector<HitEvent> ev(256);
    long total = long(seconds) * rate;
    auto t0 = std::chrono::steady_clock::now();
    for (long s = 0; s < total; s += block) {
        for (auto& x : buf) x = noise ? nd(rng) : 0.0f;
        e.process(ch, 1, block, ev.data(), int(ev.size()));
    }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return us / double(e.framesProcessed());
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: flybrain_bench brain.flybrain [seconds] [rate] [block]\n"); return 2; }
    int seconds = argc > 2 ? std::atoi(argv[2]) : 20, rate = argc > 3 ? std::atoi(argv[3]) : 48000, block = argc > 4 ? std::atoi(argv[4]) : 512;
    BrainFile brain = BrainFile::load(argv[1]);
    std::printf("brain %s: %d neurons, %d connections, %d readout neurons; host %d Hz, block %d, %d s each\n",
                brain.name.c_str(), brain.n, brain.nnz, brain.nMn(), rate, block, seconds);
    std::printf("| precision | input | us per frame | %% of one core at 200 frames/s |\n|---|---|---|---|\n");
    for (bool dbl : {false, true})
        for (bool noise : {false, true}) {
            double us = run(brain, dbl, noise, seconds, rate, block);
            std::printf("| %s | %s | %.1f | %.2f |\n", dbl ? "double" : "float", noise ? "noise" : "silence", us, us * 200.0 / 1e6 * 100.0);
        }
    return 0;
}
