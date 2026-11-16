#include "flybrain/engine.h"

#include <cmath>
#include <cstring>

#include "flybrain/dsp.h"
#include "flybrain/network.h"

#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__)
#include <xmmintrin.h>
#include <pmmintrin.h>
#define FLYBRAIN_X86 1
#endif

namespace flybrain {

void setFlushDenormals(bool on) {
#if defined(FLYBRAIN_X86)
    unsigned csr = _mm_getcsr();
    if (on) csr |= 0x8040; else csr &= ~0x8040u;   // FTZ (bit 15) and DAZ (bit 6)
    _mm_setcsr(csr);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    if (on) fpcr |= (1u << 24); else fpcr &= ~(uint64_t(1) << 24);
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
#else
    (void)on;
#endif
}

// ---------------------------------------------------------------- the precision-generic implementation
struct EngineBase {
    virtual ~EngineBase() = default;
    virtual void prepare(const EngineConfig& cfg) = 0;
    virtual void reset() = 0;
    virtual int process(const float* const* in, int nch, int ns, HitEvent* out, int maxEvents) = 0;
    virtual int flush(HitEvent* out, int maxEvents) = 0;
    virtual int stepFeatures(const double* ear, HitEvent* out, int maxEvents) = 0;
    virtual uint64_t setControls(const Controls& c, const std::string& description) = 0;
    virtual bool readSnapshot(Snapshot& out) = 0;
    virtual int latencyHostSamples() const = 0;
    virtual int64_t framesProcessed() const = 0;
    virtual int runawayCount() const = 0;
    Recorder recorder;
    std::atomic<double> trimDb{0.0};
    std::atomic<int> preroll{400};
};

template <class T>
struct EngineImpl : EngineBase {
    const BrainFile& b;
    EngineConfig cfg;
    Resampler resampler;
    Ear ear;
    Core<T> core;
    Readout<T> readout;
    PeakDecoder<T> decoder;
    RunawayGuard<T> guard;
    TripleBuffer<CompiledControls<T>> controlBus;
    const CompiledControls<T>* current = nullptr;
    CompiledControls<T> neutralControls;             // used until the first swap
    TripleBuffer<Snapshot> snapshots;
    Snapshot staging;                                // the frame waiting for its hit decision
    std::vector<float> mono, resampled;
    std::vector<double> rawBands;
    std::vector<T> feats, earScaled, earIn, bandMean, bandStd, bandGain, bandOffset;
    int64_t frame = 0, firstFrameAfterReset = 0;
    bool resetFlag = true;
    uint64_t serial = 0;
    std::atomic<uint64_t> lastSerialApplied{0};
    std::mutex controlMutex;                        // serialises UI-side setControls callers only
    int runaways = 0;
    double trimGain = 1.0;

    explicit EngineImpl(const BrainFile& brain) : b(brain) {}

    void prepare(const EngineConfig& c) override {
        cfg = c;
        resampler.prepare(cfg.hostRate, 16000);
        ear.prepare(b);
        core.prepare(b);
        readout.prepare(b);
        decoder.prepare(b.nDrums);
        guard.prepare(b);
        for (auto& slot : controlBus.slots) slot.allocate(b);
        neutralControls.allocate(b);
        Controls neutral(b);
        neutralControls.compile(b, neutral);
        current = &neutralControls;
        for (auto& s : snapshots.slots) s.allocate(b.nBands, b.n, b.nMn(), b.nDrums);
        staging.allocate(b.nBands, b.n, b.nMn(), b.nDrums);
        mono.assign(size_t(cfg.maxBlock), 0.0f);
        resampled.assign(size_t(resampler.maxOutputs(1)) + 8, 0.0f);
        rawBands.assign(size_t(b.nBands), 0.0);
        feats.assign(size_t(b.nBands), T(0)); earScaled.assign(size_t(b.nBands), T(0)); earIn.assign(size_t(b.nEars()), T(0));
        bandMean.assign(b.bandMean.begin(), b.bandMean.end()); bandStd.assign(b.bandStd.begin(), b.bandStd.end());
        bandGain.assign(b.bandGain.begin(), b.bandGain.end()); bandOffset.assign(b.bandOffset.begin(), b.bandOffset.end());
        recorder.allocate(b.nBands, b.n, b.nMn(), b.nDrums, cfg.recorderCapacityFrames);
        preroll.store(cfg.prerollFrames);
        trimDb.store(cfg.inputTrimDb);
        core.setControls(current);
        frame = 0;
        reset();
    }

    void reset() override {
        resampler.reset();
        ear.reset();
        ear.clearPending();
        core.reset();
        readout.reset();
        decoder.reset();
        guard.reset();
        firstFrameAfterReset = frame;
        resetFlag = true;
    }

    void swapControlsIfAny() {
        if (controlBus.acquire()) {
            current = &controlBus.readSlot();
            core.setControls(current);
            lastSerialApplied.store(current->serial, std::memory_order_relaxed);
        }
    }

    // One frame from raw log1p band powers (double). Returns the number of hit events written.
    int stepRaw(const double* raw, int sampleOffset, HitEvent* out, int maxEvents) {
        for (int k = 0; k < b.nBands; ++k) feats[k] = (T(raw[k]) - bandMean[k]) / bandStd[k];
        return stepNormalised(sampleOffset, out, maxEvents);
    }

    int stepNormalised(int sampleOffset, HitEvent* out, int maxEvents) {
        for (int k = 0; k < b.nBands; ++k) earScaled[k] = feats[k] * bandGain[k] + bandOffset[k];
        if (!current->neutral) for (int k = 0; k < b.nBands; ++k) earScaled[k] = earScaled[k] + current->bandOffset[k];
        return stepEar(sampleOffset, out, maxEvents);
    }

    int stepEar(int sampleOffset, HitEvent* out, int maxEvents) {
        swapControlsIfAny();
        for (int e = 0; e < b.nEars(); ++e) earIn[e] = earScaled[size_t(b.earBand[e])];
        core.step(earIn.data(), frame);
        bool runaway = guard.check(core.v, core.r);
        readout.step(core.r.data());
        int nEvents = 0;
        // decide the previous frame, publish it with its hits
        if (decoder.push(readout.prob.data(), readout.vel.data(), frame, current->threshold)) {
            nEvents += publishDecided(sampleOffset, out, maxEvents);
        }
        // stage this frame
        staging.frame = frame;
        staging.resetFlag = resetFlag;
        staging.runaway = runaway;
        staging.neutral = current->neutral;
        staging.controlsSerial = current->serial;
        for (int k = 0; k < b.nBands; ++k) { staging.bands[k] = double(feats[k]); staging.ear[k] = double(earScaled[k]); }
        for (int i = 0; i < b.n; ++i) staging.rates[i] = double(core.r[i]);
        for (int i = 0; i < b.nMn(); ++i) staging.mn[i] = double(readout.mn[i]);
        for (int d = 0; d < b.nDrums; ++d) { staging.prob[d] = double(readout.prob[d]); staging.vel[d] = double(readout.vel[d]); }
        resetFlag = false;
        ++frame;
        if (runaway) {
            // soft mute and reset: state to zero, the staged frame goes out with no hits
            int64_t f;
            if (decoder.dropPending(f)) nEvents += publishDecided(sampleOffset, out, maxEvents, true);
            core.reset(); readout.reset(); ear.reset(); guard.reset();
            firstFrameAfterReset = frame;
            resetFlag = true;
            ++runaways;
        }
        return nEvents;
    }

    int publishDecided(int sampleOffset, HitEvent* out, int maxEvents, bool forceNoHits = false) {
        Snapshot& s = snapshots.writeSlot();
        s.copyFrom(staging);
        const bool muted = (s.frame - firstFrameAfterReset) < int64_t(preroll.load(std::memory_order_relaxed));
        s.muted = muted;
        int nEvents = 0;
        for (int d = 0; d < b.nDrums; ++d) {
            uint8_t h = (muted || forceNoHits) ? 0 : decoder.hits[size_t(d)];
            s.hits[size_t(d)] = h;
            if (h && nEvents < maxEvents) {
                out[nEvents++] = HitEvent{s.frame, d, float(decoder.decidedVel[size_t(d)]), sampleOffset};
            }
        }
        recorder.push(s);
        snapshots.publish();
        return nEvents;
    }

    int process(const float* const* in, int nch, int ns, HitEvent* out, int maxEvents) override {
        if (cfg.flushDenormals) setFlushDenormals(true);
        trimGain = dbToGain(trimDb.load(std::memory_order_relaxed));
        int nEvents = 0;
        const float inv = 1.0f / float(nch > 0 ? nch : 1);
        for (int i = 0; i < ns; ++i) {
            float m = 0.0f;
            for (int c = 0; c < nch; ++c) m += in[c][i];
            m = m * inv;                                   // the mean of the channels (sampler.py's convention)
            if (trimGain != 1.0) m = float(double(m) * trimGain);
            int produced = resampler.process(&m, 1, resampled.data());
            for (int k = 0; k < produced; ++k) {
                if (ear.push(resampled[size_t(k)], rawBands.data()))
                    nEvents += stepRaw(rawBands.data(), i, out + nEvents, maxEvents - nEvents);
            }
        }
        return nEvents;
    }

    int flush(HitEvent* out, int maxEvents) override {
        if (decoder.flush(current->threshold)) return publishDecided(-1, out, maxEvents);
        return 0;
    }

    int stepFeatures(const double* earFeatures, HitEvent* out, int maxEvents) override {
        if (cfg.flushDenormals) setFlushDenormals(true);
        for (int k = 0; k < b.nBands; ++k) { earScaled[k] = T(earFeatures[k]); feats[k] = T(0); }
        if (!current->neutral) for (int k = 0; k < b.nBands; ++k) earScaled[k] = earScaled[k] + current->bandOffset[k];
        return stepEar(0, out, maxEvents);
    }

    uint64_t setControls(const Controls& c, const std::string& description) override {
        std::lock_guard<std::mutex> g(controlMutex);
        CompiledControls<T>& slot = controlBus.writeSlot();
        slot.compile(b, c);
        slot.serial = ++serial;
        controlBus.publish();
        if (recorder.running()) recorder.logControl(frame, slot.serial, description);
        return slot.serial;
    }

    bool readSnapshot(Snapshot& out) override {
        if (!snapshots.acquire()) return false;
        out.copyFrom(snapshots.readSlot());
        return true;
    }

    int latencyHostSamples() const override {
        // resampler delay, plus the frame being assembled (hop) and the decision frame (hop), at the host rate
        double hopHost = double(b.hop) * cfg.hostRate / 16000.0;
        return resampler.latencyInputSamples() + int(std::lround(2.0 * hopHost));
    }
    int64_t framesProcessed() const override { return frame; }
    int runawayCount() const override { return runaways; }
};

struct Engine::Impl {
    std::unique_ptr<EngineBase> e;
};

Engine::Engine(const BrainFile& brain) : brainRef(brain), impl(new Impl) {}
Engine::~Engine() = default;

void Engine::prepare(const EngineConfig& cfg) {
    cfgCopy = cfg;
    if (cfg.doublePrecision) impl->e.reset(new EngineImpl<double>(brainRef));
    else impl->e.reset(new EngineImpl<float>(brainRef));
    impl->e->prepare(cfg);
}
void Engine::reset() { impl->e->reset(); }
int Engine::process(const float* const* in, int nch, int ns, HitEvent* out, int maxEvents) { return impl->e->process(in, nch, ns, out, maxEvents); }
int Engine::flush(HitEvent* out, int maxEvents) { return impl->e->flush(out, maxEvents); }
int Engine::stepFeatures(const double* ear, HitEvent* out, int maxEvents) { return impl->e->stepFeatures(ear, out, maxEvents); }
void Engine::setInputTrimDb(double db) { impl->e->trimDb.store(db); }
void Engine::setPrerollFrames(int frames) { impl->e->preroll.store(frames); }
uint64_t Engine::setControls(const Controls& c, const std::string& d) { return impl->e->setControls(c, d); }
bool Engine::readSnapshot(Snapshot& out) { return impl->e->readSnapshot(out); }
void Engine::allocateSnapshot(Snapshot& s) const { s.allocate(brainRef.nBands, brainRef.n, brainRef.nMn(), brainRef.nDrums); }
Recorder& Engine::recorder() { return impl->e->recorder; }
int Engine::latencyHostSamples() const { return impl->e->latencyHostSamples(); }
int64_t Engine::framesProcessed() const { return impl->e->framesProcessed(); }
int Engine::runawayCount() const { return impl->e->runawayCount(); }

}  // namespace flybrain
