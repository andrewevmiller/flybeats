#include "data.h"
#include "flybrain/dsp.h"

int main(int, char** argv) {
    std::string dir = argv[1];
    if (!exists(dir + "/audio16k.f32")) { std::printf("SKIP: no test data\n"); return 0; }
    BrainFile b = BrainFile::load(dir + "/small.flybrain");
    std::vector<float> audio = loadRaw<float>(dir + "/audio16k.f32");
    std::vector<double> want = loadRaw<double>(dir + "/ref64_bands.f64");    // normalised features
    Ear ear; ear.prepare(b);
    std::vector<double> raw(16), got;
    for (float s : audio)
        if (ear.push(s, raw.data()))
            for (int k = 0; k < 16; ++k) got.push_back((raw[k] - b.bandMean[k]) / b.bandStd[k]);
    CHECK_MSG(got.size() == want.size(), "%zu vs %zu values", got.size(), want.size());
    double m = maxAbsDiff(got, want, std::min(got.size(), want.size()));
    CHECK_MSG(m < 1e-9, "max diff %.3e", m);
    std::printf("ear vs numpy: %zu frames, max abs diff %.3e (values up to %.2f)\n", got.size() / 16, m, maxAbs(want));
    // the first frame sees only zeros before the audio: log1p(0)=0 -> (0 - mean)/std; reset gives the same again
    ear.reset(); ear.clearPending();
    std::vector<double> again;
    for (size_t i = 0; i < 160; ++i) if (ear.push(audio[i], raw.data())) for (int k = 0; k < 16; ++k) again.push_back((raw[k] - b.bandMean[k]) / b.bandStd[k]);
    CHECK(again.size() == 32 && std::equal(again.begin(), again.end(), got.begin()));
    // FFT against a direct DFT
    RealFFT fft; fft.prepare(512);
    std::vector<double> x(512), re(257), im(257);
    std::mt19937 rng(1); std::normal_distribution<double> nd;
    for (auto& v : x) v = nd(rng);
    fft.forward(x.data(), re.data(), im.data());
    double err = 0;
    for (int k = 0; k <= 256; ++k) {
        double r = 0, i = 0;
        for (int t = 0; t < 512; ++t) { r += x[t] * std::cos(-2 * M_PI * k * t / 512.0); i += x[t] * std::sin(-2 * M_PI * k * t / 512.0); }
        err = std::max(err, std::max(std::fabs(r - re[k]), std::fabs(i - im[k])));
    }
    CHECK_MSG(err < 1e-9, "fft vs dft %.3e", err);
    return finish();
}
