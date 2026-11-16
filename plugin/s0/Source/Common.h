// Shared bits of the two S0 test plugins: a synthesized click, a 20 ms RMS window, and a meter bar.
#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <atomic>
#include <cmath>
#include <vector>

namespace flys0
{

// A 5 ms decaying 2 kHz burst. Synthesized, so no sample files and no licence question for S0.
struct Click
{
    void prepare (double sampleRate)
    {
        sr = sampleRate;
        length = (int) (0.005 * sr);
        pos = length;   // silent until triggered
    }
    void trigger (float amplitude) { pos = 0; amp = amplitude; }
    float next()
    {
        if (pos >= length) return 0.0f;
        const float t = (float) pos / (float) sr;
        const float env = 1.0f - (float) pos / (float) length;
        ++pos;
        return amp * env * std::sin (2.0f * juce::MathConstants<float>::pi * 2000.0f * t);
    }
    double sr = 48000.0;
    int length = 0, pos = 0;
    float amp = 0.5f;
};

// Running RMS over a rectangular window (20 ms): the crude onset detector S0 needs, nothing more.
struct RmsWindow
{
    void prepare (double sampleRate, double seconds)
    {
        n = juce::jmax (1, (int) (sampleRate * seconds));
        buf.assign ((size_t) n, 0.0f);
        sum = 0.0; i = 0;
    }
    float push (float x)
    {
        sum -= buf[(size_t) i];
        buf[(size_t) i] = x * x;
        sum += x * x;
        if (++i >= n) i = 0;
        if (sum < 0.0) sum = 0.0;                 // float drift guard
        return (float) std::sqrt (sum / (double) n);
    }
    std::vector<float> buf;
    double sum = 0.0;
    int n = 1, i = 0;
};

// Peak level of a bus over the last block, for the editor's meters. Written on the audio thread, read by a timer.
struct Meter
{
    void update (const juce::AudioBuffer<float>& b)
    {
        float m = 0.0f;
        for (int ch = 0; ch < b.getNumChannels(); ++ch)
            m = juce::jmax (m, b.getMagnitude (ch, 0, b.getNumSamples()));
        peak.store (m, std::memory_order_relaxed);
    }
    std::atomic<float> peak { 0.0f };
};

// A labelled horizontal bar in dB, so observations (1) and (2) are readable on screen.
class MeterBar : public juce::Component
{
public:
    explicit MeterBar (juce::String name) : label (std::move (name)) {}
    void set (float peak) { level = peak; repaint(); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds();
        g.setColour (juce::Colours::white);
        g.drawText (label, r.removeFromLeft (150), juce::Justification::centredLeft);
        auto bar = r.reduced (2);
        g.setColour (juce::Colours::darkgrey);
        g.fillRect (bar);
        const float db = juce::Decibels::gainToDecibels (level, -60.0f);
        const float frac = juce::jlimit (0.0f, 1.0f, (db + 60.0f) / 60.0f);
        g.setColour (level > 0.0f ? juce::Colours::limegreen : juce::Colours::red);
        g.fillRect (bar.removeFromLeft ((int) (frac * (float) bar.getWidth())));
        g.setColour (juce::Colours::white);
        g.drawText (level > 0.0f ? juce::String (db, 1) + " dB" : "silent (no audio)", getLocalBounds().removeFromRight (120),
                    juce::Justification::centredRight);
    }
private:
    juce::String label;
    float level = 0.0f;
};

} // namespace flys0
