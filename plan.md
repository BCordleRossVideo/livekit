# LiveKit Transcription VST Plugin — Implementation Plan

## Overview

Build a **JUCE-based VST3 plugin** ("LiveKit Transcriber") that runs inside **LAMA Mix** or **LAMA Connect** (both support VST plugins). The plugin captures audio from the host's audio bus, publishes it into a LiveKit room as an audio track, and displays real-time transcription text received back from a server-side LiveKit STT agent.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAMA Mix / LAMA Connect (VST Host)                         │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  LiveKit Transcriber VST3 Plugin                       │ │
│  │                                                        │ │
│  │  ┌──────────┐   ring    ┌───────────┐   WebRTC   ┌──┐ │ │
│  │  │  JUCE    │──buffer──▶│ LiveKit   │───────────▶│LK│ │ │
│  │  │processBlk│           │ C++ SDK   │            │  │ │ │
│  │  └──────────┘           │ publish   │◀───────────│Rm│ │ │
│  │                         └───────────┘  transcr.  └──┘ │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Plugin Editor (GUI)                             │  │ │
│  │  │  ┌─────────────────────────────┐                 │  │ │
│  │  │  │ Settings Panel              │                 │  │ │
│  │  │  │  • Server URL               │                 │  │ │
│  │  │  │  • API Key / API Secret     │                 │  │ │
│  │  │  │  • Room Name / Identity     │                 │  │ │
│  │  │  │  • [Connect] [Disconnect]   │                 │  │ │
│  │  │  └─────────────────────────────┘                 │  │ │
│  │  │  ┌─────────────────────────────┐                 │  │ │
│  │  │  │ Transcription Display       │                 │  │ │
│  │  │  │  (scrolling text area)      │                 │  │ │
│  │  │  └─────────────────────────────┘                 │  │ │
│  │  │  Status: Connected ● | Latency: 42ms             │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Plan

### Step 1: Project Scaffolding & Build System

Create a new CMake-based JUCE project at `/home/user/livekit/livekit-transcription-vst/`.

**Files to create:**
- `CMakeLists.txt` — top-level build file that:
  - Fetches JUCE via `FetchContent` (or git submodule)
  - Integrates the LiveKit C++ SDK (`livekit/client-sdk-cpp`) as a dependency
  - Adds a C++ JWT library ([firebase/jwt-cpp](https://github.com/Thalhammer/jwt-cpp)) for token generation
  - Defines the VST3 plugin target using `juce_add_plugin()`
- `src/` directory with the source files listed below

**Build dependencies:**
- CMake >= 4.0
- Rust/Cargo (stable) — required by LiveKit C++ SDK's Rust submodule
- Protobuf compiler (`protoc`)
- JUCE 7+ (fetched via CMake)
- jwt-cpp (header-only, fetched via CMake)

### Step 2: Plugin Processor (Audio Engine)

**File:** `src/PluginProcessor.h` / `src/PluginProcessor.cpp`

This is the JUCE `AudioProcessor` subclass — the core of the VST plugin.

**Responsibilities:**
- `prepareToPlay()` — initialize the lock-free ring buffer sized for the host sample rate and block size
- `processBlock()` — on every audio callback from the host:
  1. Copy incoming audio samples into a **lock-free SPSC ring buffer** (Single Producer, Single Consumer). This is critical — the JUCE audio thread is real-time and must never block or allocate.
  2. Pass audio through unchanged (the plugin is an insert effect that doesn't modify audio)
- Store plugin parameters as `AudioProcessorValueTreeState` properties:
  - `serverUrl` (String) — LiveKit server WebSocket URL (e.g., `wss://myserver.livekit.cloud`)
  - `apiKey` (String) — LiveKit API Key
  - `apiSecret` (String) — LiveKit API Secret
  - `roomName` (String) — target room name
  - `participantIdentity` (String) — identity for this plugin instance
- `getStateInformation()` / `setStateInformation()` — persist settings with the host session (XML serialization). **Note:** API Secret will be stored; warn the user in the UI.

### Step 3: LiveKit Connection Manager

**File:** `src/LiveKitConnection.h` / `src/LiveKitConnection.cpp`

A dedicated class managing the LiveKit room lifecycle on a **background thread** (never on the audio thread).

**Responsibilities:**

1. **Token Generation:**
   - Use jwt-cpp to create a signed JWT with claims:
     - `iss` = API Key
     - `sub` = room name
     - `iat` / `exp` = current time / expiry (e.g., 6 hours)
     - `video.room` = room name
     - `video.roomJoin` = true
     - `video.canPublish` = true
     - `video.canSubscribe` = true (to receive transcription events)
     - `jti` = participant identity
   - Sign with HS256 using the API Secret

2. **Room Connection:**
   ```cpp
   livekit::Room room;
   livekit::RoomOptions opts;
   room.connect(serverUrl, token, opts);
   ```

3. **Audio Publishing:**
   - Create an `AudioSource` with the host's sample rate and channel count (mono recommended for transcription)
   - Create a `LocalAudioTrack` from the source
   - Publish via `room.localParticipant()->publishTrack(track)`
   - Run a background thread that:
     - Reads audio from the ring buffer
     - Resamples to 16kHz mono if needed (optimal for STT)
     - Constructs `AudioFrame` objects (int16 PCM, 20ms chunks)
     - Calls `audioSource->captureFrame(frame)`

4. **Transcription Reception:**
   - Register for `RoomEvent::TranscriptionReceived` events
   - Parse `TranscriptionSegment` objects (id, text, language, final flag)
   - Forward transcription text to the UI via a thread-safe callback / message queue

5. **Connection State Management:**
   - Track states: Disconnected → Connecting → Connected → Disconnecting
   - Expose state to UI for status display
   - Handle reconnection on network drops (LiveKit SDK has built-in reconnect)

### Step 4: Ring Buffer (Lock-Free Audio Transport)

**File:** `src/RingBuffer.h`

A **lock-free single-producer single-consumer (SPSC) ring buffer** to safely pass audio from the real-time JUCE audio thread to the LiveKit publishing thread.

- Template-based, works with `float` samples
- Uses `std::atomic` for read/write positions
- Sized to hold ~200ms of audio (provides headroom for scheduling jitter)
- No allocations, no locks, no syscalls — safe for real-time use

JUCE provides `juce::AbstractFifo` which can also be used for this purpose.

### Step 5: Audio Resampler

**File:** `src/AudioResampler.h` / `src/AudioResampler.cpp`

The host may run at 44.1kHz, 48kHz, or 96kHz, but LiveKit STT agents typically expect **16kHz mono** audio.

- Use JUCE's built-in `juce::LagrangeInterpolator` or `juce::WindowedSincInterpolator` for resampling
- Convert stereo to mono (average L+R channels)
- Convert float32 → int16 PCM (required by `AudioFrame`)
- Process on the LiveKit publishing thread (not the audio thread)

### Step 6: Plugin Editor (GUI)

**File:** `src/PluginEditor.h` / `src/PluginEditor.cpp`

A JUCE `AudioProcessorEditor` subclass providing the plugin UI.

**Layout (approx 500×600 px):**

```
┌──────────────────────────────────────┐
│  LiveKit Transcriber            v1.0 │
├──────────────────────────────────────┤
│  Server URL:  [wss://...          ]  │
│  API Key:     [___________________]  │
│  API Secret:  [●●●●●●●●●●●●●●●●●]  │
│  Room:        [___________________]  │
│  Identity:    [___________________]  │
│                                      │
│  [  Connect  ]    [  Disconnect  ]   │
│  Status: ● Connected (48kHz mono)    │
├──────────────────────────────────────┤
│  Transcription:                      │
│ ┌──────────────────────────────────┐ │
│ │ Speaker 1: Hello, welcome to    │ │
│ │ the show today.                  │ │
│ │ Speaker 1: We're going to be    │ │
│ │ talking about audio production.  │ │
│ │ Speaker 2: Thanks for having me │ │
│ │ ...                              │ │
│ └──────────────────────────────────┘ │
│  [ Clear ]               [ Copy ]   │
├──────────────────────────────────────┤
│  Latency: 42ms  │  Frames: 12,450   │
└──────────────────────────────────────┘
```

**Components:**
- `juce::TextEditor` fields for server URL, API key, room name, identity
- A password-masked `TextEditor` for API secret
- Connect/Disconnect `juce::TextButton` pair
- A `juce::Label` for connection status with color indicator
- A read-only, scrollable `juce::TextEditor` for transcription output
- Clear/Copy buttons for the transcription area
- A status bar showing latency and frame count

**Behavior:**
- Settings fields disabled while connected
- Connect button triggers `LiveKitConnection::connect()`
- Transcription text appended via `MessageManager::callAsync()` (thread-safe GUI update)
- All settings persisted with the plugin state

### Step 7: LAMA Mix / LAMA Connect Compatibility

Both LAMA products support **VST3 plugins** on Windows. Key considerations:

- Build as **VST3** format (primary target for LAMA)
- Also build **AU** for macOS compatibility
- The plugin acts as an **audio insert effect** (receives audio on its input bus, passes it through unchanged to output)
- LAMA Connect can route any audio source through VST plugins in its signal chain
- LAMA Mix supports VST plugins on each channel strip
- Use standard JUCE plugin categories: `pluginIsSynth = false`, `pluginWantsMidiInput = false`
- Test with common sample rates: 44.1kHz, 48kHz, 96kHz
- Support mono and stereo input configurations

### Step 8: Build & Package

- CMake presets for Windows (MSVC), macOS (Xcode/clang), Linux (GCC/clang)
- Build produces:
  - `.vst3` bundle (all platforms)
  - `.component` AU bundle (macOS only)
- Installer scripts or instructions for copying to standard VST3 directories:
  - Windows: `C:\Program Files\Common Files\VST3\`
  - macOS: `~/Library/Audio/Plug-Ins/VST3/`
  - Linux: `~/.vst3/`

---

## File Structure

```
livekit-transcription-vst/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── PluginProcessor.h
│   ├── PluginProcessor.cpp
│   ├── PluginEditor.h
│   ├── PluginEditor.cpp
│   ├── LiveKitConnection.h
│   ├── LiveKitConnection.cpp
│   ├── RingBuffer.h
│   ├── AudioResampler.h
│   ├── AudioResampler.cpp
│   └── TokenGenerator.h        // JWT token generation using jwt-cpp
└── resources/
    └── logo.png                 // Optional branding
```

---

## Key Technical Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LiveKit C++ SDK build complexity (Rust toolchain required) | Document prerequisites clearly; consider pre-building the SDK as a static library |
| Real-time audio thread safety | Strict lock-free ring buffer; all LiveKit calls on background thread only |
| Sample rate mismatch | Resampler converts any host rate to 16kHz for STT |
| API Secret stored in plugin state | Warn user in UI; consider optional file-based secret storage |
| LiveKit C++ SDK maturity | SDK is official but less documented than JS/Python; may need to reference Rust SDK source for edge cases |
| LAMA-specific quirks | Test with LAMA's trial version; the plugin uses standard VST3 APIs so should be compatible |

---

## Server-Side Requirement (Not part of plugin, but needed)

For transcription to work, you need a **LiveKit STT Agent** running server-side that:
1. Subscribes to the audio track published by this plugin
2. Runs speech-to-text (e.g., Deepgram, Whisper, Google STT)
3. Publishes transcription segments back to the room via `TranscriptionReceived` events

LiveKit provides pre-built agent templates for this at [docs.livekit.io/agents](https://docs.livekit.io/agents/build/text/).

---

## Implementation Order

1. **Step 1** — Project scaffolding, CMakeLists.txt, verify JUCE + LiveKit SDK compile together
2. **Step 4** — Ring buffer (independent, testable in isolation)
3. **Step 5** — Audio resampler (independent, testable)
4. **Step 2** — Plugin processor with ring buffer integration
5. **Step 3** — LiveKit connection manager (token gen + connect + publish + receive transcription)
6. **Step 6** — Plugin editor GUI
7. **Step 7** — LAMA compatibility testing
8. **Step 8** — Build & packaging
