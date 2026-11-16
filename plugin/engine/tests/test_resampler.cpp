#include "data.h"
#include "flybrain/dsp.h"

static std::vector<float> resampleAll(int rate, const std::vector<float>& x, int block) {
    Resampler r; r.prepare(rate, 16000);
    std::vector<float> out, buf(size_t(r.maxOutputs(block)) + 8);
    for (size_t p = 0; p < x.size(); p += size_t(block)) {
        int n = int(std::min(size_t(block), x.size() - p));
        int m = r.process(x.data() + p, n, buf.data());
        out.insert(out.end(), buf.begin(), buf.begin() + m);
    }
    return out;
}

int main(int, char** argv) {
    std::string dir = argv[1];
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (int rate : {44100, 48000, 88200, 96000, 176400, 192000}) {
        std::vector<float> x(size_t(rate) * 2);
        for (auto& v : x) v = nd(rng);
        std::vector<float> base = resampleAll(rate, x, 1);
        CHECK_MSG(std::fabs(double(base.size()) - 2.0 * 16000) < 200, "rate %d: %zu outputs", rate, base.size());
        for (int block : {7, 64, 480, 4096}) {
            std::vector<float> y = resampleAll(rate, x, block);
            CHECK_MSG(y == base, "rate %d block %d not bit-identical", rate, block);
        }
    }
    // pass-through at 16 kHz
    { Resampler r; r.prepare(16000, 16000); CHECK(r.isPassThrough() && r.latencyInputSamples() == 0); }
    // agreement with the numpy port
    for (int rate : {44100, 48000, 96000}) {
        std::string hp = dir + "/host" + std::to_string(rate) + ".f32";
        if (!exists(hp)) { std::printf("SKIP numpy agreement at %d: no data\n", rate); continue; }
        std::vector<float> x = loadRaw<float>(hp), want = loadRaw<float>(dir + "/host" + std::to_string(rate) + "_16k.f32");
        std::vector<float> got = resampleAll(rate, x, 512);
        CHECK_MSG(got.size() == want.size(), "rate %d: %zu vs %zu outputs", rate, got.size(), want.size());
        double m = 0;
        for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) m = std::max(m, double(std::fabs(got[i] - want[i])));
        CHECK_MSG(m <= 1e-6, "rate %d: max diff vs numpy %.3e", rate, m);
        std::printf("rate %d: numpy agreement max diff %.3e over %zu samples\n", rate, m, got.size());
    }
    // a 1 kHz tone passes at unit gain; a 9 kHz tone (above the 16 kHz Nyquist) aliases to 7 kHz and must be < -80 dB
    for (int rate : {44100, 48000, 96000}) {
        std::vector<float> x(size_t(rate) * 3);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(std::sin(2 * M_PI * 1000.0 * double(i) / rate) + std::sin(2 * M_PI * 9000.0 * double(i) / rate));
        std::vector<float> y = resampleAll(rate, x, 256);
        auto level = [&](double hz) {
            double re = 0, im = 0;
            size_t start = 16000, len = y.size() - 16000 - 100;   // skip the start-up
            for (size_t k = 0; k < len; ++k) { double ph = 2 * M_PI * hz * double(k) / 16000.0; re += y[start + k] * std::cos(ph); im += y[start + k] * std::sin(ph); }
            return 2.0 * std::sqrt(re * re + im * im) / double(len);
        };
        double a1 = level(1000.0), a7 = level(7000.0);
        CHECK_MSG(std::fabs(20 * std::log10(a1)) < 0.05, "rate %d: 1 kHz gain %.4f dB", rate, 20 * std::log10(a1));
        CHECK_MSG(20 * std::log10(a7 / a1) < -80, "rate %d: alias at 7 kHz %.1f dB", rate, 20 * std::log10(a7 / a1));
        std::printf("rate %d: 1 kHz gain %+.4f dB, 9 kHz alias %.1f dB\n", rate, 20 * std::log10(a1), 20 * std::log10(a7 / a1));
    }
    // time alignment and production delay: an impulse at input 1000 at 48 kHz peaks at output 333 and is produced
    // only once input 1000 + H has arrived
    {
        Resampler r; r.prepare(48000, 16000);
        CHECK(r.latencyInputSamples() == Resampler::kHalfTaps && r.upFactor() == 1 && r.downFactor() == 3);
        std::vector<float> x(3000, 0.0f); x[1000] = 1.0f;
        std::vector<float> out, buf(8);
        size_t producedAt333 = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            int m = r.process(&x[i], 1, buf.data());
            for (int k = 0; k < m; ++k) { out.push_back(buf[k]); if (out.size() == 334) producedAt333 = i; }
        }
        size_t peak = size_t(std::max_element(out.begin(), out.end()) - out.begin());
        CHECK_MSG(peak == 333, "peak at %zu", peak);
        CHECK_MSG(producedAt333 == 1000 + size_t(Resampler::kHalfTaps) - 1 || producedAt333 == 1000 + size_t(Resampler::kHalfTaps), "output 333 produced at input %zu", producedAt333);
    }
    return finish();
}
