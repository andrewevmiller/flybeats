// Writes WAV files so the owner can listen: every drum at three velocities in each voicing, and the sonification
// driven by sine-modulated group rates. In-house 16-bit PCM WAV writer; no dependency.
//     voices_render <out_dir>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "flybrain/voices.h"

using namespace flybrain::voices;

static void writeWav(const std::string& path, const std::vector<float>& L, const std::vector<float>& R, int sr) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    uint32_t n = uint32_t(L.size()), dataBytes = n * 4, riff = 36 + dataBytes;
    auto u32 = [&](uint32_t v) { std::fputc(v & 255, f); std::fputc((v >> 8) & 255, f); std::fputc((v >> 16) & 255, f); std::fputc((v >> 24) & 255, f); };
    auto u16 = [&](uint16_t v) { std::fputc(v & 255, f); std::fputc((v >> 8) & 255, f); };
    std::fwrite("RIFF", 1, 4, f); u32(riff); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16(2); u32(uint32_t(sr)); u32(uint32_t(sr) * 4); u16(4); u16(16);
    std::fwrite("data", 1, 4, f); u32(dataBytes);
    for (uint32_t i = 0; i < n; ++i) {
        for (float x : {L[i], R[i]}) { x = x > 1 ? 1.0f : x < -1 ? -1.0f : x; u16(uint16_t(int16_t(std::lround(x * 32767.0f)))); }
    }
    std::fclose(f);
}

int main(int argc, char** argv) {
    std::string out = argc > 1 ? argv[1] : ".";
    const int sr = 48000;
    const char* names[kDrums] = {"kick", "hihat_pedal", "snare", "hihat_closed", "hihat_open", "tom", "crash", "ride"};
    const char* voicings[3] = {"studio", "tight", "room"};
    for (int v = 0; v < 3; ++v) {
        Kit kit; kit.prepare(sr, 512); kit.setVoicing(v);
        Limiter lim; lim.prepare(sr, 512);
        int per = int(0.6 * sr), n = per * 3 * kDrums + sr;
        std::vector<float> L(size_t(n), 0.0f), R(size_t(n), 0.0f);
        for (int p = 0; p < n; p += 512) {
            int len = std::min(512, n - p);
            for (int d = 0; d < kDrums; ++d) for (int k = 0; k < 3; ++k) { int at = (d * 3 + k) * per; if (at >= p && at < p + len) kit.trigger(d, k == 0 ? 0.3f : k == 1 ? 0.6f : 1.0f, at - p); }
            kit.process(L.data() + p, R.data() + p, len);
            lim.process(L.data() + p, R.data() + p, len);
        }
        writeWav(out + "/kit_" + voicings[v] + ".wav", L, R, sr);
        std::printf("wrote %s/kit_%s.wav: %s at velocities 0.3, 0.6, 1.0 each\n", out.c_str(), voicings[v], "every drum");
    }
    {
        Sonification son; son.prepare(sr, 512); son.setEnabled(true); son.setLevelDb(-6.0f);
        int n = 8 * sr; std::vector<float> L(size_t(n), 0.0f), R(size_t(n), 0.0f);
        for (int p = 0; p < n; p += 240) {                // one 5 ms frame per update
            float z[kDrums]; double t = double(p) / sr;
            for (int g = 0; g < kDrums; ++g) z[g] = float(2.0 * (1.0 + std::sin(2 * M_PI * (0.1 + 0.07 * g) * t + g)));
            son.setGroupRates(z, kDrums);
            son.process(L.data() + p, R.data() + p, std::min(240, n - p));
        }
        writeWav(out + "/sonification.wav", L, R, sr);
        std::printf("wrote %s/sonification.wav: eight voices driven by sine-modulated z in [0, 4]\n", out.c_str());
    }
    (void)names;
    return 0;
}
