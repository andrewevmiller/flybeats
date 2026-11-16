// The engine facade: host audio in, hit events out, with the snapshot tap, control bus and recorder tap.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "flybrain/brainfile.h"
#include "flybrain/bus.h"
#include "flybrain/controls.h"
#include "flybrain/recorder.h"

namespace flybrain {

struct HitEvent {
    int64_t frame = 0;       // the 5 ms frame the hit belongs to
    int drum = 0;
    float velocity = 0;
    int sampleOffset = 0;    // host-rate sample within the block at which the hit became known (-1: at flush)
};

struct EngineConfig {
    int hostRate = 48000;
    int maxBlock = 4096;
    bool doublePrecision = false;
    bool flushDenormals = true;
    int prerollFrames = 400;           // 2 s: hits muted after a reset, the network keeps running
    double inputTrimDb = 0.0;
    size_t recorderCapacityFrames = 1024;
};

class Engine {
public:
    explicit Engine(const BrainFile& brain);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Allocates everything. Not real-time safe. Resets.
    void prepare(const EngineConfig& cfg);
    // Real-time safe after prepare(): no allocation, lock, I/O or exception.
    void reset();
    int process(const float* const* inputs, int numChannels, int numSamples, HitEvent* out, int maxEvents);
    // End of stream (print mode): decides the last frame. Real-time safe.
    int flush(HitEvent* out, int maxEvents);
    // Steps one frame from ear features already normalised and scaled (nBands values, the reference engine's
    // `ear` array), bypassing resampler and ear: for the bit-exact core tests. Real-time safe.
    int stepFeatures(const double* earFeatures, HitEvent* out, int maxEvents);

    void setInputTrimDb(double db);                 // any thread, takes effect at the next block
    void setPrerollFrames(int frames);
    // Control bus: compile off the audio thread, swapped in at the next frame boundary. Returns the serial.
    uint64_t setControls(const Controls& c, const std::string& description = "");
    // Snapshot tap: copies the newest frame into `out` (sized by prepare's allocateSnapshot). False if none newer.
    bool readSnapshot(Snapshot& out);
    void allocateSnapshot(Snapshot& s) const;
    Recorder& recorder();

    int latencyHostSamples() const;
    int64_t framesProcessed() const;
    int runawayCount() const;
    const BrainFile& brain() const { return brainRef; }
    const EngineConfig& config() const { return cfgCopy; }

    struct Impl;

private:
    const BrainFile& brainRef;
    EngineConfig cfgCopy;
    std::unique_ptr<Impl> impl;
};

void setFlushDenormals(bool on);   // this thread's FTZ/DAZ (x86) or FZ (arm64)

}  // namespace flybrain
