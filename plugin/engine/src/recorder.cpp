#include "flybrain/recorder.h"

#include <chrono>
#include <cstring>

namespace flybrain {

Recorder::~Recorder() { stop(); }

void Recorder::allocate(int nb, int nn, int nm, int nd, size_t capacityFrames) {
    nBands = nb; n = nn; nMn = nm; nDrums = nd;
    ring.allocate(capacityFrames);
    for (auto& s : ring.storage()) s.allocate(nb, nn, nm, nd);
}

void Recorder::start(const std::string& bin, const std::string& json, const Json& inf) {
    stop();
    binPath = bin; jsonPath = json; info = inf;
    file = std::fopen(binPath.c_str(), "wb");
    if (!file) throw std::runtime_error("recorder: cannot open " + binPath);
    const char magic[8] = {'F', 'L', 'Y', 'R', 'E', 'C', '0', '1'};
    std::fwrite(magic, 1, 8, file);
    uint32_t dims[4] = {uint32_t(nBands), uint32_t(n), uint32_t(nMn), uint32_t(nDrums)};
    std::fwrite(dims, 4, 4, file);
    written.store(0);
    { std::lock_guard<std::mutex> g(logMutex); controlLog.clear(); }
    stopping.store(false);
    active.store(true);
    thread = std::thread([this] { worker(); });
}

void Recorder::stop() {
    if (!active.load()) return;
    stopping.store(true);
    if (thread.joinable()) thread.join();
    active.store(false);
    std::fclose(file);
    file = nullptr;
    Json side = info;
    side.type = Json::Object;
    Json j; j.type = Json::Number; j.num = double(written.load()); side.obj["frames_written"] = j;
    j.num = double(ring.droppedCount()); side.obj["frames_dropped"] = j;
    j.num = nBands; side.obj["n_bands"] = j; j.num = n; side.obj["n_neurons"] = j; j.num = nMn; side.obj["n_mn"] = j; j.num = nDrums; side.obj["n_drums"] = j;
    Json log; log.type = Json::Array;
    { std::lock_guard<std::mutex> g(logMutex); log.arr = controlLog; }
    side.obj["control_log"] = log;
    Json fmt; fmt.type = Json::String;
    fmt.str = "FLYREC01: magic[8], uint32 n_bands,n,n_mn,n_drums; per frame: int64 frame, uint8 reset, uint8 runaway, uint8 neutral, uint8 muted, uint64 serial, float64 bands[n_bands], ear[n_bands], rates[n], mn[n_mn], prob[n_drums], vel[n_drums], uint8 hits[n_drums]";
    side.obj["format"] = fmt;
    FILE* f = std::fopen(jsonPath.c_str(), "wb");
    if (f) { std::string text = side.dump(); std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); }
}

bool Recorder::push(const Snapshot& s) {
    if (!active.load(std::memory_order_relaxed)) return false;
    Snapshot* slot = ring.beginPush();
    if (!slot) return false;
    slot->copyFrom(s);
    ring.endPush();
    return true;
}

void Recorder::logControl(int64_t frame, uint64_t serial, const std::string& description) {
    Json e; e.type = Json::Object;
    Json v; v.type = Json::Number; v.num = double(frame); e.obj["frame"] = v;
    v.num = double(serial); e.obj["serial"] = v;
    Json d; d.type = Json::String; d.str = description; e.obj["controls"] = d;
    std::lock_guard<std::mutex> g(logMutex);
    controlLog.push_back(e);
}

void Recorder::writeFrame(const Snapshot& s) {
    std::fwrite(&s.frame, 8, 1, file);
    uint8_t flags[4] = {uint8_t(s.resetFlag), uint8_t(s.runaway), uint8_t(s.neutral), uint8_t(s.muted)};
    std::fwrite(flags, 1, 4, file);
    std::fwrite(&s.controlsSerial, 8, 1, file);
    std::fwrite(s.bands.data(), 8, s.bands.size(), file);
    std::fwrite(s.ear.data(), 8, s.ear.size(), file);
    std::fwrite(s.rates.data(), 8, s.rates.size(), file);
    std::fwrite(s.mn.data(), 8, s.mn.size(), file);
    std::fwrite(s.prob.data(), 8, s.prob.size(), file);
    std::fwrite(s.vel.data(), 8, s.vel.size(), file);
    std::fwrite(s.hits.data(), 1, s.hits.size(), file);
    written.fetch_add(1);
}

void Recorder::worker() {
    for (;;) {
        Snapshot* s = ring.front();
        if (s) { writeFrame(*s); ring.pop(); continue; }
        if (stopping.load()) {
            if (!ring.front()) break;
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace flybrain
