# talkytalk

> **talkytalk, sesi metne çevirmek için değil; konuşmayı doğru anda, doğru şekilde bölmek ve yönlendirmek için tasarlanmış, gerçek zamanlı bir konuşma sinyal motorudur.**

talkytalk produces probabilistic intent signals from streaming audio. It does not speak, decide, or respond. It only signals.

## Philosophy

> "İnsanlar konuşurken beklenmek istemez. Sistemler ise hâlâ cümle bitmesini bekler."

talkytalk solves this by:
- Processing audio in real-time (20-40ms frames)
- Predicting intent *before* sentence completion
- Analyzing *how* something is said, not just *what*
- Producing probabilistic, retractable signals

**Core principle**: Early wrong > Late right

## Installation

```bash
pip install talkytalk
```

For audio device support:
```bash
pip install talkytalk[audio]
```

## Quick Start

```python
from talkytalk import Pipeline, PipelineConfig
from talkytalk.analyzers import VADAnalyzer, ProsodyAnalyzer, EmotionAnalyzer
from talkytalk.predictors import IntentPredictor, TimingPredictor

# Create pipeline
pipeline = (
    Pipeline()
    .add_analyzer(VADAnalyzer())
    .add_analyzer(ProsodyAnalyzer())
    .add_analyzer(EmotionAnalyzer())
    .add_predictor(IntentPredictor())
    .add_predictor(TimingPredictor())
)

# Process audio source
for packet in pipeline.run_sync(audio_source):
    if packet.timing.interrupt_safe:
        # System can safely respond now
        handle_intent(packet)
```

## The IntentPacket

talkytalk's sole output is the `IntentPacket`:

```python
IntentPacket(
    intent: Intent,              # query | command | conversation | unknown
    confidence: float,           # 0.0 – 1.0
    language: str,               # detected language
    target_language: str | None, # for translation intent
    emotion: Emotion(
        arousal: float,          # calm (0) ↔ excited (1)
        valence: float,          # negative (0) ↔ positive (1)
    ),
    timing: Timing(
        user_paused: bool,       # meaningful pause detected
        interrupt_safe: bool,    # system can respond now
        speech_likelihood: float,# probability of continued speech
        silence_duration_ms: int,
    ),
    frame_id: int,
    timestamp_ms: int,
)
```

**This is not a decision. This is not a command. This is a signal.**

## Architecture

```
Audio Stream → [Frame Buffer] → [Analyzers] → [Predictors] → IntentPacket
                    ↓               ↓              ↓
               20-40ms frames   Features      Predictions
```

### Analyzers (Feature Extraction)
- **VADAnalyzer**: Voice activity detection (energy + zero-crossing)
- **ProsodyAnalyzer**: Pitch, tempo, intonation patterns
- **EmotionAnalyzer**: Arousal/valence from acoustic features
- **LanguageAnalyzer**: Language identification (placeholder)

### Predictors (Intent Estimation)
- **IntentPredictor**: Early intent from behavioral cues
- **TimingPredictor**: Turn-taking and interrupt safety

### Adapters (Output Transformation)
- **DictAdapter**: Convert to dictionary/JSON
- **CallbackAdapter**: Event-driven notifications
- Custom adapters for LLMs, game engines, etc.

## Early Intent Heuristics

| Pattern | Acoustic Cues |
|---------|---------------|
| **Command** | High arousal + falling intonation + fast tempo |
| **Query** | Rising intonation + moderate arousal + hesitation |
| **Conversation** | Neutral arousal + mixed intonation |

## Timing Signals

```python
# User paused for 300ms+ after speech
packet.timing.user_paused == True

# Safe to interrupt/respond
packet.timing.interrupt_safe == True

# Probability user will continue speaking
packet.timing.speech_likelihood  # 0.0 - 1.0
```

**Interrupt safety requires:**
- Pause detected (≥300ms)
- Not rising intonation (question forming)
- Low speech likelihood (<0.7)
- OR: Long pause (≥700ms)
- OR: Falling intonation + pause + high intent confidence

## Extensibility

### Custom Analyzer

```python
from talkytalk.analyzers.base import Analyzer, AnalysisResult

class MyAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "my_analyzer"
    
    def analyze(self, frame, buffer, state) -> AnalysisResult:
        # Extract features from frame
        return AnalysisResult(
            analyzer_name=self.name,
            frame_id=frame.frame_id,
            timestamp_ms=frame.timestamp_ms,
            data={"my_feature": computed_value},
        )
```

### Custom Predictor

```python
from talkytalk.predictors.base import Predictor, PredictionContext

class MyPredictor(Predictor):
    @property
    def name(self) -> str:
        return "my_predictor"
    
    def predict(self, context: PredictionContext, state) -> None:
        # Update state based on analysis results
        my_result = context.analysis_results.get("my_analyzer")
        if my_result:
            state.intent_confidence = compute_confidence(my_result)
```

### Custom Adapter

```python
from talkytalk.adapters.base import Adapter

class LLMPromptAdapter(Adapter[str]):
    @property
    def name(self) -> str:
        return "llm_prompt"
    
    def transform(self, packet) -> str:
        return f"[User emotion: {packet.emotion.quadrant}] "
```

## What talkytalk is NOT

- ❌ A chatbot
- ❌ An LLM
- ❌ A voice assistant
- ❌ A decision engine
- ❌ An ASR system

talkytalk only produces signals. Downstream systems make decisions.

## Design Principles (Non-Negotiable)

### Core Philosophy
| Principle | Meaning |
|-----------|----------|
| **Audio-first, Text-later** | System listens to produce behavior, not to transcribe |
| **Intent ≠ Meaning** | Intent prediction is independent of semantic meaning |
| **Partial truth > Full sentence** | 40% correct early signal beats 100% correct late sentence |
| **Every frame is a vote** | No single decision; frames accumulate hypotheses |
| **No blocking ever** | No component can block the main pipeline flow |
| **Realtime > Accuracy** | Live interaction wins over offline correctness |
| **Human pacing matters** | 200ms early response beats 2s late correct answer |
| **Turn-taking is first-class** | Speaking turn is as critical as content |
| **Interrupt is a feature** | Interruptibility is designed, not accidental |
| **Emotion is modulation** | Emotion guides, but doesn't decide |

## Roadmap

### ✅ Completed (v0.1.0)
- [x] Core pipeline architecture
- [x] IntentPacket data structure
- [x] VAD, Prosody, Emotion analyzers
- [x] Intent, Timing predictors
- [x] **Early intent prediction** (thinking while listening)
- [x] **Turn-taking & overlap detection**
- [x] **Behavior mapping layer** (Emotion → Action)
- [x] **Comprehensive benchmark suite**
- [x] Test infrastructure

### 🔜 Near Future (3-5 steps)
- [ ] **Adaptive thresholds** – User-learning VAD / interrupt thresholds
- [ ] **Session memory (non-text)** – Last 10-30s acoustic behavior memory
- [ ] **User speaking style fingerprint** – Tempo, pause, arousal profile
- [ ] **Dynamic emit rate** – More frequent during speech, less during silence
- [ ] **Confidence decay** – Old intents naturally fade in extended silence

### 🚀 Vision (5-10 steps)
- [ ] **LLM-as-reactor, not brain** – LLM only executes, pipeline is the brain
- [ ] **Cross-modal hooks** – Eye, face, gesture can plug in (not required)
- [ ] **Predict-before-speech** – Intent probability before user speaks
- [ ] **Multi-agent readiness** – Same pipeline scales to multiple speakers
- [ ] **Hardware-aware pipelines** – Edge / mobile / embedded variants
- [ ] **Conversation physics** – Speech as forces, friction, momentum

### 🔧 Technical Backlog
- [ ] WebRTC VAD integration (production-grade)
- [x] Live microphone source (sounddevice)
- [ ] Lightweight LID (CPU-only, low latency)
- [ ] Heavy LID (wav2vec2, optional)
- [ ] ASR integration (optional, downstream only)
- [ ] LLM adapter examples (OpenAI, Anthropic)
- [ ] Game adapter examples (Unity, Unreal)
- [ ] WebSocket streaming

## New in v0.1.0

### 🧠 Early Intent Prediction
```python
from talkytalk.predictors import EarlyIntentPredictor

# Produces hypotheses DURING speech
# Tracks hypothesis stability
# Computes interruptibility score
```

### 🗣️ Advanced Turn-Taking
```python
from talkytalk.predictors import TurnTakingPredictor

# States: USER_SPEAKING, USER_PAUSING, TURN_YIELDED
# Overlap probability estimation
# Suggested wait time calculation
```

### 🎭 Behavior Mapping
```python
from talkytalk.behavior import BehaviorMapper, BehaviorMode

mapper = BehaviorMapper(mode=BehaviorMode.TEACHER)
signal = mapper.map(packet)
# → response_strategy, suggested_tone, empathy_level
```

### ⚡ Benchmark Suite
```python
from talkytalk.benchmark import BenchmarkSuite

suite = BenchmarkSuite(latency_budget_ms=10.0)
result = suite.run("my_test", pipeline, source)
# → latency stats, jitter, spike rate, realtime factor
```

## 🎤 Live Microphone Demo

```bash
# Install audio support
pip install sounddevice

# Run live demo
python examples/microphone_demo.py
```

Speak into your microphone and see real-time intent signals!

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run simple test (no microphone needed)
python examples/simple_test.py

# Type check
mypy talkytalk/

# Run microphone demo
python examples/microphone_demo.py
```

## Documentation

See [DOCS.md](DOCS.md) for comprehensive technical documentation in Turkish.

## License

MIT
