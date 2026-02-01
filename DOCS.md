# talkytalk - Kapsamlı Teknik Dokümantasyon

## 🎯 Proje Tanımı

**talkytalk**, gerçek zamanlı ses akışından davranışsal sinyaller üreten bir Python kütüphanesidir.

### Temel Felsefe

> "İnsanlar konuşurken beklenmek istemez. Sistemler ise hâlâ cümle bitmesini bekler."

talkytalk bu sorunu çözer:
- **Cümle bitmeden** niyet tahmini yapar
- **Ne söylendiğinden önce** nasıl söylendiğini analiz eder
- **Olasılıksal ve geri alınabilir** sinyaller üretir
- **Asla karar vermez** - sadece sinyal üretir

### Ne Değildir

| ❌ Değil | ✅ Olan |
|----------|---------|
| Chatbot | Sinyal işleyici |
| LLM | Davranış analizcisi |
| Sesli asistan | Çekirdek kütüphane |
| Karar motoru | Olasılık üreticisi |
| ASR sistemi | Prozodi analizcisi |

---

## 🏗️ Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                               │
│                    (20-40ms frames)                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRAME BUFFER                                │
│                  (sliding window ~1000ms)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │   VAD    │    │ PROSODY  │    │ EMOTION  │
    │ Analyzer │    │ Analyzer │    │ Analyzer │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   ANALYSIS RESULTS  │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  INTENT  │   │  TIMING  │   │  EARLY   │
    │ Predictor│   │ Predictor│   │ INTENT   │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │    INTENT PACKET    │
              │  (Tek ve Nihai)     │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ BEHAVIOR │   │   LLM    │   │   GAME   │
    │  MAPPER  │   │ ADAPTER  │   │ ADAPTER  │
    └──────────┘   └──────────┘   └──────────┘
```

---

## 📦 Modül Yapısı

```
talkytalk/
├── core/                    # Çekirdek veri yapıları
│   ├── packet.py           # IntentPacket, Emotion, Timing
│   ├── stream.py           # AudioFrame, FrameBuffer
│   └── pipeline.py         # Ana işleme pipeline'ı
│
├── analyzers/               # Özellik çıkarıcılar
│   ├── base.py             # Analyzer protokolü
│   ├── vad.py              # Voice Activity Detection
│   ├── prosody.py          # Pitch, tempo, intonation
│   ├── emotion.py          # Arousal/Valence
│   └── language.py         # Dil algılama (placeholder)
│
├── predictors/              # Tahmin ediciler
│   ├── base.py             # Predictor protokolü
│   ├── intent.py           # Temel niyet tahmini
│   ├── timing.py           # Turn-taking sinyalleri
│   ├── early_intent.py     # Erken/progresif niyet
│   └── overlap.py          # Gelişmiş turn-taking
│
├── behavior/                # Davranış eşleme
│   └── mapper.py           # Emotion → Action
│
├── adapters/                # Çıktı dönüştürücüler
│   └── base.py             # Adapter protokolü
│
├── benchmark/               # Performans ölçümü
│   └── metrics.py          # Latency tracking
│
└── sources.py               # Test ses kaynakları
```

---

## 🎛️ IntentPacket - Tek ve Nihai Çıktı

```python
@dataclass(frozen=True)
class IntentPacket:
    # Temel niyet
    intent: Intent           # query | command | conversation | unknown
    confidence: float        # 0.0 – 1.0
    
    # Dil bilgisi
    language: str            # algılanan dil
    target_language: str     # çeviri hedefi (opsiyonel)
    
    # Duygusal durum
    emotion: Emotion
        arousal: float       # sakin (0) ↔ gergin (1)
        valence: float       # negatif (0) ↔ pozitif (1)
    
    # Zamanlama sinyalleri
    timing: Timing
        user_paused: bool         # anlamlı duraklama
        interrupt_safe: bool      # sistem cevap verebilir
        speech_likelihood: float  # konuşma devam edecek mi
        silence_duration_ms: int  # sessizlik süresi
    
    # Meta
    frame_id: int
    timestamp_ms: int
```

**Bu bir karar değildir. Bu bir komut değildir. Bu bir sinyaldir.**

---

## 🔬 Analyzers (Özellik Çıkarıcılar)

### VADAnalyzer - Voice Activity Detection

```python
from talkytalk.analyzers import VADAnalyzer

vad = VADAnalyzer(
    energy_threshold_db=-40.0,  # Minimum ses seviyesi
    hangover_frames=5,          # Konuşma sonrası bekleme
    adaptive=True,              # Gürültü tabanına uyum
)
```

**Çıktılar:**
- `is_speech`: Konuşma var mı
- `speech_probability`: Konuşma olasılığı (0-1)
- `energy_db`: Enerji seviyesi (dB)
- `zero_crossing_rate`: Sıfır geçiş oranı

**Algoritma:**
1. RMS enerji hesaplama
2. Zero-crossing rate (gürültü vs konuşma ayrımı)
3. Adaptif gürültü tabanı güncelleme
4. Hangover (kesintisiz konuşma için)

---

### ProsodyAnalyzer - Prozodik Özellikler

```python
from talkytalk.analyzers import ProsodyAnalyzer

prosody = ProsodyAnalyzer(
    min_pitch_hz=50.0,       # Minimum F0
    max_pitch_hz=500.0,      # Maximum F0
    pause_threshold_ms=200,  # Duraklama eşiği
)
```

**Çıktılar:**
- `pitch_hz`: Temel frekans
- `pitch_variance`: Pitch varyansı
- `tempo`: Konuşma hızı (hece/sn)
- `is_rising_intonation`: Soru mu?
- `is_falling_intonation`: Komut/tamamlanma mı?
- `pause_duration_ms`: Duraklama süresi

**Niyet İpuçları:**
| Pattern | Prosodic Cue |
|---------|--------------|
| Soru | Rising intonation |
| Komut | Falling + yüksek tempo |
| Düşünme | Uzun duraklama |
| Belirsizlik | Yüksek pitch variance |

---

### EmotionAnalyzer - Duygu Analizi

```python
from talkytalk.analyzers import EmotionAnalyzer

emotion = EmotionAnalyzer(
    smoothing_alpha=0.3,  # Temporal smoothing
)
```

**Model: Russell's Circumplex**

```
        High Arousal
             │
   Tense ────┼──── Excited
   Negative  │     Positive
             │
  ───────────┼───────────
             │
   Sad ──────┼──── Calm
   Negative  │     Positive
             │
        Low Arousal
```

**Arousal korelasyonları:**
- Yüksek enerji → Yüksek arousal
- Hızlı konuşma → Yüksek arousal
- Yüksek pitch variance → Yüksek arousal

**Valence korelasyonları (zayıf):**
- Yüksek pitch → Daha pozitif (genel)
- Rising intonation → Daha pozitif

---

## 🎯 Predictors (Tahmin Ediciler)

### IntentPredictor - Temel Niyet Tahmini

```python
from talkytalk.predictors import IntentPredictor

intent_pred = IntentPredictor(
    confidence_threshold=0.3,
    decay_rate=0.95,  # Temporal decay
)
```

**Heuristikler:**

| Intent | Koşullar |
|--------|----------|
| COMMAND | arousal > 0.7 + falling + tempo > 4 |
| QUERY | rising intonation |
| CONVERSATION | 0.3 < arousal < 0.7 + karışık intonation |

---

### EarlyIntentPredictor - Erken Niyet (YENİ!)

```python
from talkytalk.predictors import EarlyIntentPredictor

early = EarlyIntentPredictor(
    stability_threshold=5,       # Frame sayısı
    confidence_momentum=0.8,     # Smoothing
    hypothesis_timeout_ms=2000,  # Hipotez ömrü
)
```

**Farkı:**
- Konuşma **sürerken** hipotez üretir
- Hipotez stabilitesini takip eder
- **Interruptibility score** hesaplar

**Interruptibility Kullanımı:**
```python
# LLM entegrasyonu
if packet.interruptibility > 0.3:
    start_prefetching()  # Hazırlanmaya başla

if packet.interruptibility > 0.6:
    begin_generating()   # Spekülatif üretim

if packet.interruptibility > 0.8 and packet.timing.interrupt_safe:
    deliver_response()   # Cevabı ver
```

---

### TimingPredictor - Zamanlama

```python
from talkytalk.predictors import TimingPredictor

timing = TimingPredictor(
    pause_threshold_ms=300,      # Duraklama eşiği
    turn_end_threshold_ms=700,   # Turn sonu eşiği
    interrupt_confidence=0.6,    # Interrupt için min confidence
)
```

**Interrupt Safety Algoritması:**
```
interrupt_safe = (
    NOT rising_intonation AND
    speech_likelihood < 0.7 AND
    (
        silence >= 700ms OR
        (falling_intonation AND silence >= 300ms) OR
        (intent_confidence >= 0.6 AND user_paused)
    )
)
```

---

### TurnTakingPredictor - Gelişmiş Turn-Taking (YENİ!)

```python
from talkytalk.predictors import TurnTakingPredictor

turn = TurnTakingPredictor(
    min_turn_gap_ms=200,
    safe_interrupt_gap_ms=500,
    max_wait_ms=2000,
)
```

**Çıktılar:**
- `state`: USER_SPEAKING | USER_PAUSING | TURN_YIELDED | SYSTEM_CAN_SPEAK
- `can_interrupt`: Teknik olarak mümkün mü
- `should_wait`: Stratejik öneri
- `overlap_probability`: Üst üste konuşma riski
- `suggested_wait_ms`: Önerilen bekleme süresi

---

## 🎭 Behavior Mapping (YENİ!)

### BehaviorMapper - Duygu → Davranış

```python
from talkytalk.behavior import BehaviorMapper, BehaviorMode

mapper = BehaviorMapper(mode=BehaviorMode.ASSISTANT)
signal = mapper.map(packet)

# LLM prompt'una ekle
prompt_prefix = signal.to_prompt_prefix()
# → "Respond gently and with understanding. Tone: empathetic."
```

**Modlar:**
| Mode | Base Delay | Empathy | Energy Matching |
|------|------------|---------|-----------------|
| ASSISTANT | 100ms | 0.5 | 0.3 |
| TEACHER | 300ms | 0.8 | 0.2 |
| GAME_NPC | 50ms | 0.4 | 0.9 |
| COMPANION | 200ms | 0.9 | 0.7 |
| CUSTOMER_SERVICE | 150ms | 0.7 | 0.2 |

**Response Strategies:**
- `IMMEDIATE`: Hemen cevap ver
- `GENTLE`: Yumuşak, anlayışlı
- `WAIT`: Bekle
- `MIRROR`: Enerjiyi yansıt
- `CALM_DOWN`: Sakinleştir
- `ENERGIZE`: Canlandır

---

## ⚡ Benchmark Suite (YENİ!)

```python
from talkytalk.benchmark import BenchmarkSuite, LatencyTracker

# Latency tracking
tracker = LatencyTracker(budget_ms=10.0)

with tracker.measure("frame_process"):
    pipeline.process_frame(frame)

stats = tracker.get_stats("frame_process")
# {
#     "mean_ms": 2.3,
#     "p95_ms": 4.1,
#     "p99_ms": 6.8,
#     "max_ms": 12.1,
#     "jitter_ms": 0.8,
#     "over_budget_rate": 0.02,
# }

# Full benchmark
suite = BenchmarkSuite(latency_budget_ms=10.0)
result = suite.run("test_pipeline", pipeline, source, ground_truth={
    "intents": ["query"],
    "safe_interrupt_windows": [(500, 700), (1200, 1500)],
    "speech_windows": [(0, 400), (800, 1100)],
    "final_intent": "query",
})

print(result.summary())
```

**Ölçülen Metrikler:**
- ⏱️ End-to-end latency (mean, p95, p99, max)
- 📈 Jitter (latency variance)
- 🔥 Spike detection
- 🎯 Intent accuracy
- 🔄 Interrupt success rate
- 😶 False silence rate
- 🧠 Early intent precision
- ⚡ Realtime factor (>1 = faster than realtime)

---

## 🚀 Kurulum ve Çalıştırma

### Kurulum

```bash
# Temel kurulum
pip install talkytalk

# Geliştirici kurulumu
git clone https://github.com/serhatskywalker/TalkyTalk
cd TalkyTalk
pip install -e ".[dev]"

# Audio desteği ile
pip install talkytalk[audio]
```

### Bağımlılıklar

```
numpy>=1.24.0          # Zorunlu
sounddevice>=0.4.6     # Opsiyonel (mikrofon)
webrtcvad>=2.0.10      # Opsiyonel (production VAD)
pytest>=7.0.0          # Geliştirme
pytest-asyncio>=0.21.0 # Geliştirme
mypy>=1.0.0            # Geliştirme
```

### Temel Kullanım

```python
from talkytalk import Pipeline, PipelineConfig, AudioConfig
from talkytalk.analyzers import VADAnalyzer, ProsodyAnalyzer, EmotionAnalyzer
from talkytalk.predictors import IntentPredictor, TimingPredictor, EarlyIntentPredictor
from talkytalk.sources import SineSource

# Pipeline oluştur
config = PipelineConfig(
    audio=AudioConfig(sample_rate=16000, frame_duration_ms=20),
    emit_interval_ms=100,
)

pipeline = (
    Pipeline(config)
    .add_analyzer(VADAnalyzer())
    .add_analyzer(ProsodyAnalyzer())
    .add_analyzer(EmotionAnalyzer())
    .add_predictor(IntentPredictor())
    .add_predictor(TimingPredictor())
    .add_predictor(EarlyIntentPredictor())
)

# Ses kaynağı
source = SineSource(frequency_hz=200, duration_ms=1000)

# Senkron işleme
for packet in pipeline.run_sync(source):
    print(f"Intent: {packet.intent.value}")
    print(f"Confidence: {packet.confidence:.2f}")
    print(f"Interrupt safe: {packet.timing.interrupt_safe}")
```

### Async Kullanım

```python
import asyncio

async def process_audio():
    pipeline = create_pipeline()
    source = get_audio_source()
    
    async for packet in pipeline.run(source):
        if packet.timing.interrupt_safe:
            await handle_response(packet)

asyncio.run(process_audio())
```

### Callback Kullanım

```python
def on_packet(packet):
    if packet.is_actionable:
        trigger_llm(packet)

pipeline.on_packet(on_packet)
pipeline.run_sync(source)
```

---

## 🧪 Test Çalıştırma

```bash
# Tüm testler
pytest tests/

# Belirli test
pytest tests/test_pipeline.py -v

# Coverage ile
pytest tests/ --cov=talkytalk

# Type checking
mypy talkytalk/
```

---

## 📋 Roadmap

### ✅ Tamamlanan
- [x] Çekirdek pipeline mimarisi
- [x] IntentPacket veri yapısı
- [x] VAD, Prosody, Emotion analyzers
- [x] Intent, Timing predictors
- [x] Early intent prediction
- [x] Turn-taking & overlap detection
- [x] Behavior mapping layer
- [x] Benchmark suite
- [x] Test altyapısı

### 🔜 Planlanan
- [ ] WebRTC VAD entegrasyonu
- [ ] Gerçek mikrofon kaynağı (sounddevice)
- [ ] Lightweight LID (CPU-only)
- [ ] Heavy LID (wav2vec2, opsiyonel)
- [ ] ASR entegrasyonu (opsiyonel, downstream)
- [ ] Örnek LLM adapter (OpenAI)
- [ ] Örnek Game adapter (Unity)
- [ ] WebSocket streaming
- [ ] Comprehensive benchmarks

---

## 🔑 Tasarım İlkeleri (Non-Negotiable)

### Temel Prensipler
1. **Early wrong > Late right** - Erken hatalı tahmin, geç doğru tahminden iyidir
2. **Silence is signal** - Sessizlik boşluk değil, sinyaldir
3. **Probabilistic, not deterministic** - Her çıktı olasılıksaldır
4. **Model agnostic** - Herhangi bir modelle çalışabilir
5. **Minimal dependencies** - Sadece numpy zorunlu
6. **Modular & swappable** - Her bileşen değiştirilebilir
7. **ASR optional** - Metin olmadan da çalışır

### Güçlü Tasarım Prensipleri
| Prensip | Anlam |
|---------|-------|
| **Audio-first, Text-later** | Sistem sesi "metne çevirmek" için değil, davranış üretmek için dinler |
| **Intent ≠ Meaning** | Niyet tahmini semantik anlamdan bağımsızdır |
| **Partial truth > Full sentence** | %40 doğru erken sinyal, %100 doğru geç cümleden değerlidir |
| **Every frame is a vote** | Tek karar yok, frame'ler hipotez biriktirir |
| **No blocking ever** | Pipeline'da hiçbir bileşen ana akışı durduramaz |
| **Realtime > Accuracy** | Offline doğruluk değil, canlı etkileşim kazanır |
| **Human pacing matters** | 200ms erken cevap, 2s geç doğru cevaptan iyidir |
| **Turn-taking is first-class** | Konuşma sırası, içerik kadar kritiktir |
| **Interrupt is a feature** | Bölünebilirlik bilinçli tasarlanır |
| **Emotion is modulation** | Duygu yönlendirir, karar vermez |

---

## 🛣️ Gelecek Yol Haritası

### 🔜 Yakın Gelecek (3-5 Adım)
- **Adaptive thresholds** – Kullanıcıya göre öğrenen VAD / interrupt eşikleri
- **Session memory (non-text)** – Son 10-30 saniyenin akustik davranış hafızası
- **User speaking style fingerprint** – Tempo, pause, arousal profili
- **Dynamic emit rate** – Yoğun konuşmada daha sık, sessizlikte daha seyrek emit
- **Confidence decay** – Uzayan sessizlikte eski intent'lerin doğal ölmesi

### 🚀 Vizyon (5-10 Adım)
- **LLM-as-reactor, not brain** – LLM sadece karar uygulayıcı, beyin pipeline
- **Cross-modal hooks** – Göz, yüz, gesture eklenebilir (zorunlu değil)
- **Predict-before-speech** – Kullanıcı konuşmadan niyet ihtimali üretimi
- **Multi-agent readiness** – Aynı pipeline birden fazla konuşmacıya ölçeklenir
- **Hardware-aware pipelines** – Edge / mobile / embedded varyantlar
- **Conversation physics** – Konuşma = kuvvetler, sürtünme, momentum

---
    
## 📜 Lisans

MIT License
