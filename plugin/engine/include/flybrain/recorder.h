// The recorder tap: the audio thread pushes frames into an SPSC ring; a worker thread writes them to disk
// with a JSON sidecar. plugin/export/flybrain/recording.py loads the pair.
#pragma once
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "flybrain/bus.h"
#include "flybrain/util.h"

namespace flybrain {

class Recorder {
public:
    ~Recorder();
    void allocate(int nBands, int n, int nMn, int nDrums, size_t capacityFrames);
    // Off the audio thread. `info` is written into the sidecar as-is (brain hash, host, sample rate, block size).
    void start(const std::string& binPath, const std::string& jsonPath, const Json& info);
    void stop();                                  // drains the ring, writes the sidecar
    bool running() const { return active.load(); }
    // Audio thread: never blocks. Returns false if the frame was dropped (ring full) or the recorder is off.
    bool push(const Snapshot& s);
    // Any thread: a control change to log (frame it takes effect, serial, description).
    void logControl(int64_t frame, uint64_t serial, const std::string& description);
    uint64_t framesWritten() const { return written.load(); }
    uint64_t framesDropped() const { return ring.droppedCount(); }

private:
    void worker();
    void writeFrame(const Snapshot& s);
    SpscRing<Snapshot> ring;
    std::atomic<bool> active{false}, stopping{false};
    std::atomic<uint64_t> written{0};
    std::thread thread;
    FILE* file = nullptr;
    std::string binPath, jsonPath;
    Json info;
    std::mutex logMutex;
    std::vector<Json> controlLog;
    int nBands = 0, n = 0, nMn = 0, nDrums = 0;
};

}  // namespace flybrain
