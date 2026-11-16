// Fly S0 Instrument: a VST3 *instrument* (an FL Studio generator channel) whose only purpose is to answer:
//   (2) does a generator receive any mixer audio on an input bus? (forum reports say no; S0 looks)
//   (3) does FL route MIDI out from an instrument (the wrapper's MIDI output port)?
// It clicks and sends a MIDI note once per beat from the host transport, and meters its optional input bus.
#include "Common.h"

namespace flys0
{

class InstrumentProcessor : public juce::AudioProcessor
{
public:
    InstrumentProcessor()
        // A synth with a stereo output and an *optional* stereo input bus. If the host connects audio to it, the
        // meter shows it; if a generator gets none (as reported for FL Studio), the meter reads silent.
        : AudioProcessor (BusesProperties()
                              .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                              .withInput ("Input", juce::AudioChannelSet::stereo(), false))
    {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo()) return false;
        const auto in = layouts.getMainInputChannelSet();
        return in.isDisabled() || in == juce::AudioChannelSet::stereo() || in == juce::AudioChannelSet::mono();
    }

    void prepareToPlay (double sampleRate, int) override { sr = sampleRate; click.prepare (sampleRate); }
    void releaseResources() override {}

    using juce::AudioProcessor::processBlock;   // the double-precision overload stays JUCE's default
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override
    {
        juce::ScopedNoDenormals noDenormals;
        const int n = buffer.getNumSamples();
        // Input bus meter (observation 2). The bus may be disabled by the host; then it stays silent.
        const bool haveIn = getBus (true, 0) != nullptr && getBus (true, 0)->isEnabled() && getTotalNumInputChannels() > 0;
        inputEnabled.store (haveIn);
        if (haveIn) inMeter.update (getBusBuffer (buffer, true, 0)); else inMeter.peak.store (0.0f);

        // One click and one MIDI note (observation 3) on every beat of the host transport.
        midi.clear();
        bool playing = false; double bpm = 0.0, ppq = 0.0;
        if (auto* ph = getPlayHead())
            if (auto pos = ph->getPosition())
            {
                playing = pos->getIsPlaying();
                bpm = pos->getBpm().orFallback (0.0);
                ppq = pos->getPpqPosition().orFallback (-1.0);
            }
        transportPlaying.store (playing); transportBpm.store (bpm); transportPpq.store (ppq);
        if (playing && bpm > 0.0 && ppq >= 0.0)
        {
            const double ppqPerSample = bpm / 60.0 / sr;
            const double ppqEnd = ppq + n * ppqPerSample;
            for (double beat = std::ceil (ppq - 1e-9); beat < ppqEnd; beat += 1.0)
            {
                const int offset = juce::jlimit (0, n - 1, (int) ((beat - ppq) / ppqPerSample));
                if ((long long) beat == lastBeat) continue;       // a block boundary can revisit the same beat
                lastBeat = (long long) beat;
                midi.addEvent (juce::MidiMessage::noteOn (1, 36, (juce::uint8) 100), offset);
                pendingOffs.push_back (offset + (int) (0.010 * sr));
                clickAt.push_back (offset);
                beats.fetch_add (1);
            }
        }
        else lastBeat = -1;

        auto out = getBusBuffer (buffer, false, 0);
        out.clear();
        for (int i = 0; i < n; ++i)
        {
            for (size_t k = 0; k < clickAt.size(); ++k)
                if (clickAt[k] == i) click.trigger (0.5f);
            const float c = click.next();
            for (int ch = 0; ch < out.getNumChannels(); ++ch) out.setSample (ch, i, c);
        }
        for (size_t k = 0; k < pendingOffs.size();)
        {
            if (pendingOffs[k] < n) { midi.addEvent (juce::MidiMessage::noteOff (1, 36), pendingOffs[k]); pendingOffs.erase (pendingOffs.begin() + (long) k); }
            else { pendingOffs[k] -= n; ++k; }
        }
        clickAt.clear();
    }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Fly S0 Instrument"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

    Meter inMeter;
    std::atomic<bool> inputEnabled { false }, transportPlaying { false };
    std::atomic<double> transportBpm { 0.0 }, transportPpq { 0.0 };
    std::atomic<int> beats { 0 };

private:
    double sr = 48000.0;
    Click click;
    long long lastBeat = -1;
    std::vector<int> pendingOffs, clickAt;   // tiny, reserved below; S0 does not police allocation
};

class InstrumentEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit InstrumentEditor (InstrumentProcessor& p) : AudioProcessorEditor (p), proc (p)
    {
        addAndMakeVisible (inBar);
        addAndMakeVisible (info);
        info.setColour (juce::Label::textColourId, juce::Colours::white);
        info.setJustificationType (juce::Justification::topLeft);
        setSize (520, 140);
        startTimerHz (20);
    }
    void paint (juce::Graphics& g) override { g.fillAll (juce::Colour (0xff202020)); }
    void resized() override
    {
        auto r = getLocalBounds().reduced (8);
        inBar.setBounds (r.removeFromTop (24));
        info.setBounds (r);
    }
private:
    void timerCallback() override
    {
        inBar.set (proc.inMeter.peak.load());
        info.setText (juce::String ("input bus: ") + (proc.inputEnabled.load() ? "enabled by host" : "disabled by host")
                      + "\ntransport: " + (proc.transportPlaying.load() ? "playing" : "stopped")
                      + "  bpm " + juce::String (proc.transportBpm.load(), 1)
                      + "  ppq " + juce::String (proc.transportPpq.load(), 2)
                      + "\nbeats sent as MIDI note 36: " + juce::String (proc.beats.load()),
                      juce::dontSendNotification);
    }
    InstrumentProcessor& proc;
    MeterBar inBar { "input bus (audio in)" };
    juce::Label info;
};

juce::AudioProcessorEditor* InstrumentProcessor::createEditor() { return new InstrumentEditor (*this); }

} // namespace flys0

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new flys0::InstrumentProcessor(); }
