# User Guide

This guide assumes ADR Studio is already installed. It walks through the full workflow: signing in, choosing a console, creating a voice profile, training, synthesis, external integration, settings, and backup.

If you just want to hear something quickly, read [Quick Start](quickstart.md) first.

---

## 1. First launch and sign-in

### What happens after installation

The Windows installers come in two variants: **CPU** and **GPU (CUDA)**. They share the same AppId, so you can upgrade or switch variants in place without losing voices or base models.

On the very first launch you will see:

1. **Terms of use** — you must accept before the engine starts.
2. **Setup wizard (3 steps)** — local account, device/performance, storage and privacy. “Use defaults” skips the details.
3. **Prewarm page** — downloads the ~2 GB base model and loads the engine. Progress and stage are shown; the launcher appears only when both are done.
4. **Launcher** — console cards, model library, settings, and about page.

Prewarm normally happens once. After that, ADR runs entirely offline.

### The local account

The account belongs to this computer only. It does not connect to any server:

- Default username: `ADR`
- Default password: `Adr12345@`

You can change it during setup or later in **Launcher → Settings → Local account** (password at least 6 characters). Only a PBKDF2-derived value is stored, in `.local_account` inside your data folder. The plaintext password is never written to disk.

Sign-in only unlocks the local UI. It is unrelated to any cloud account. If you forget the password, deleting `.local_account` resets it; voice profiles are unaffected.

> Source installs, `server.bat`, and Docker deployments have no sign-in page — they expose the HTTP API directly.

### Other ways to accept the terms

| Deployment | How to accept |
| --- | --- |
| Desktop app | Click “Accept” on the terms page |
| CLI / source | Add `--agree-terms` to the command |
| Server / Docker | Set `ADR_AGREE_TERMS=1` |

Without acceptance the service exits with code 2. The record is stored locally only.

---

## 2. Which console should I use?

All three consoles share the same engine and data. They differ in how much control they expose:

| Console | Path | Best for | What it does |
| --- | --- | --- | --- |
| **Easy** | `/easy` | First-time users, quick output | Pick a voice, type, listen. Three steps, no parameters |
| **Pro** | `/pro` | Training, tuning, monitoring | Overview, voice cloning, synthesis, model library |
| **Call** | `/call` | Integrating with other software | Two API endpoints, parameter docs, generated cURL / Python / JS, N.E.K.O reference table |

Rules of thumb:

- **Just want speech:** use Easy.
- **Want to train a voice, control epochs / VRAM recipe / speed / splitting:** use Pro.
- **Already comfortable, connecting N.E.K.O or your own app:** use Call.
- The index page (`/`) is only a hub. Every console has a **Docs** button in the header, plus language and theme switches.

---

## 3. Voice profiles

A voice profile is the central object in ADR: a reference recording, metadata, and optional fine-tuned weights.

### What is inside

```
data/voices/<profile-name>/
  ref.wav     clean single-speaker reference (3–10 s works; 1 min+ is better for creation)
  meta.json   prompt_text / t2s_weights / vits_weights / style / created_at, etc.
```

- Similarity depends mainly on `ref.wav` and the fine-tuned weights.
- Weight paths may point to ADR training output or to GPT-SoVITS weights you place yourself.
- The folder name is the profile name, and also the model ID seen by OpenAI-compatible clients and N.E.K.O.

### Creating a profile

From **Easy → Create a new voice**, or by importing a package in **Pro → Speech synthesis**:

1. Prepare a recording with **one speaker, a quiet room, and no background music**. Supported formats: `wav / mp3 / flac / m4a / ogg`. A minute or more is ideal; a few seconds is enough for a usable starting profile.
2. Choose the language spoken in the recording (Chinese / English / Japanese / Korean / Cantonese). This becomes the profile's default synthesis language.
3. Name it (up to 20 characters, unique).
4. Start learning. ADR slices, transcribes, extracts features, and builds the profile in the background. Failures show a reason and a retry button.

A new profile is zero-shot. To get closer to the original voice, train or continue training.

### Binding trained weights

There are three equivalent routes:

- **Easy:** select an existing voice → Create a new voice → tick **Improve an existing voice** → upload a new recording. It continues from the old weights and binds the new version back to the same profile. Previous versions are kept and can be rolled back.
- **Pro → Voice cloning:** fill in “Bind to voice profile after training”.
- **Manual:** point `t2s_weights` / `vits_weights` in the profile metadata at your own output.

### Export, import, rollback

- **Export package** creates a `.adrvoice` file containing the reference audio, bound weights, and metadata. Use it for backup or moving to another computer.
- **Import package** accepts `.adrvoice` or `.zip`. It asks before overwriting an existing profile, and asks again if the weights are shared with another profile.
- **Roll back** restores the previous weight version; the current version is pushed into history first, so you can step back repeatedly.
- A voice package contains a voice model. **Only share it with people you trust.**

---

## 4. Training

Training is in **Pro → Voice cloning**.

### Before you start

- Training **requires an NVIDIA CUDA GPU**. AMD, Intel, and CPU-only machines can synthesize but cannot train. You can train on another NVIDIA machine or cloud GPU and copy the `.adrvoice` package back.
- 4 GB of VRAM is the minimum; 8 GB is more comfortable. Close games, hardware-accelerated browsers, and other TTS services first.
- Upload a clean recording. ADR slices and transcribes it automatically.

### Parameters

| Parameter | Meaning |
| --- | --- |
| Experiment name | Identifier for this run and its output folder |
| Epochs | 1–50, default 8. More is not always better on small datasets |
| VRAM recipe | `8GB+` (batch 4, faster) or `4GB` (batch 1, clips ≤10 s) |
| Bind after training | Auto-create a profile (recommended), do not bind, or bind to an existing profile |
| Quality gate | On by default; stops when similarity reaches 0.80 to avoid overfitting |

### What to watch

- **State / best similarity / zero-shot baseline** — tells you whether training actually beat the untrained voice.
- **Similarity curve** — one score per epoch, with the 0.80 target line.
- **Training log** — failures are usually explained in plain text.

Stop cancels both training and the quality gate. Outputs go to `output/`; logs go to the data folder's `logs/`.

Common failure causes: out of VRAM, recording too short or silent, disk full, a second training run already active (HTTP 409), or missing training scripts in an incomplete installation. See [Troubleshooting](troubleshooting.md).

---

## 5. Synthesis

### Pro console

1. Select a voice profile (refresh, export, import, roll back as needed).
2. Enter text.
3. Set common options: speed `0.5–2.0`, seed (`-1` for random), format `wav / mp3 / raw`.
4. Expand **Advanced parameters** to change text language, prompt language, sentence splitting (`cut0` none / `cut1` group of four / `cut3` by sentence / `cut5` by punctuation), `top_k`, and other sampling options.
5. Click **Synthesize**. The result can be played and downloaded.

The first request loads weights and may take tens of seconds; after that the model stays resident. Synthesis requests are serialized inside the engine, and the UI shows queue and busy state.

### Easy console

Pick a voice → type (language defaults to auto, so mixed Chinese/English works) → generate → play or save. Professional parameters are hidden.

### If it does not sound right

- Make sure the reference is clean, single-speaker, and music-free.
- Confirm the profile actually has trained weights bound (the Pro training page shows similarity).
- Do not send empty text or text made only of punctuation.
- See [Troubleshooting](troubleshooting.md) for more.

---

## 6. The Call console and external integration

### Where the service listens

By default ADR listens on `127.0.0.1:9881` (local only). To let other LAN devices call it, you must do **both**:

```powershell
$env:ADR_TTS_API_KEY = "<your-key>"     # comma-separated for multiple keys
python -m adr.server --addr 0.0.0.0
```

Otherwise any device that can reach the port can call it without authentication.

### API surfaces

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v3/tts/stream-input` | GPT-SoVITS api_v3 duplex streaming |
| `GET /api/v3/voices` | Voice list (ADR profiles) |
| `POST /v1/audio/speech` | OpenAI-compatible TTS |
| `GET /v1/models` | OpenAI-compatible model list (profiles) |
| `POST /api/v2/tts` | GSV api_v2 compatibility (N.E.K.O default) |
| `POST /api/adr/v1/tts` | ADR native API, synthesis by profile name |
| `GET /api/adr/v1/health` | Health and engine stage |

Full fields, error codes, and streaming byte format are in the [TTS API specification](tts-api-spec.md).

### Using the Call console

- Switch between **Native `/api/adr/v1/tts`** and **GSV-compatible `/api/v2/tts`** at the top.
- Fill in parameters on the left; generated cURL / Python / JS appears on the right.
- **Synthesize and preview** sends a real request to your local service.
- The N.E.K.O reference table mirrors the currently selected profile.

### N.E.K.O in four steps

1. Start ADR and confirm the port in the banner (default 9881).
2. In N.E.K.O, choose **GPT-SoVITS** as the TTS provider (OpenAI-compatible also works).
3. Set the API address to `http://127.0.0.1:9881`.
4. Refresh the voice list; ADR profiles appear.

The first synthesis is slow (cold start); subsequent requests stream quickly.

---

## 7. Settings: global defaults vs temporary values

The difference is simple: **will this change still apply after the next engine start, or only to the current request?**

### Global defaults (Launcher → Settings)

Desktop settings are injected into the local engine as **process environment variables**. They do not modify system environment variables, and they take effect **the next time the engine starts** after you click Save.

| Setting | Effect |
| --- | --- |
| Expose to LAN | Listening address and auth linkage |
| API Key | Comma-separated; empty means no auth |
| Voice profile folder | `ADR_DATA_DIR`; changing it does not move existing profiles |
| Default voice profile | `ADR_TTS_DEFAULT_PROFILE`; used when a request omits a profile |
| Model download source | HuggingFace or hf-mirror mirror |
| Autostart, download limit | Shell behavior and base-model download |
| Safety audit log, AI-generated WAV tag | Records events only; no text or voiceprints |
| Advanced: port, NUMBA cache, device, CPU threads, log level, dataloader workers, CUDA Graph, CPU compatibility | Troubleshooting and performance |

These values are **shared by every console and every request**. For example, setting a default voice means callers that omit `profile` get that voice.

### Temporary values (console parameters)

Parameters in Easy, Pro, and Call — speed, seed, format, language, splitting, `top_k`, the profile selected for this request — apply **only to that synthesis or training run**. They are not written back to settings and do not affect later requests.

Profile metadata sits in between: it is a **profile-level default**. Explicit request fields override profile values, which override the global default profile.

### Environment variables for source/server deployments

Without the desktop shell, environment variables provide the same global behavior:

```powershell
$env:ADR_DATA_DIR = "D:\ADR_data"
$env:ADR_TTS_DEFAULT_PROFILE = "my_voice"
$env:ADR_TTS_DEVICE = "cpu"            # auto / cuda / cpu
$env:ADR_TTS_API_KEY = "key-a,key-b"
$env:ADR_LOG_LEVEL = "DEBUG"
python -m adr.server -p 9881
```

---

## 8. Data folder, backup, and migration

### Where data lives

Priority order:

1. `ADR_DATA_DIR`
2. `F:/ADR_data` (legacy path, used if it exists)
3. `%LOCALAPPDATA%\ADR\data` (Windows default)

Typical contents:

```
data/voices/        voice profiles (ref.wav + meta.json + optional weights)
output/uploads/     raw recordings uploaded from the console
output/             training outputs and logs
logs/server.log     service log
tts_cache/          synthesis cache
.local_account      local account (desktop only)
```

**Base models are not in the data folder.** They live in the installation directory's model folder, managed by prewarm. The setup wizard's third page can show the location, check integrity, and open the folder.

### Three backup strategies

| Method | What it preserves | Best for |
| --- | --- | --- |
| Export `.adrvoice` | One profile's reference audio, weights, metadata | Moving to another computer, sharing with yourself |
| Copy the whole data folder | All profiles, uploads, logs, cache, account | Full migration or machine backup |
| Copy only `data/voices/` | All voice profiles | Keeping voices without logs/cache |

Close ADR before copying; training or synthesis may be writing files.

### Moving to a new computer

1. Install the same version (CPU or GPU as appropriate).
2. Launch once, accept the terms, complete the account step, and let prewarm finish.
3. Close ADR. Copy `data/voices/` from the old machine into the new data folder, or import each `.adrvoice` package.
4. Start ADR again. Profiles appear in the voice list and in `/api/adr/v1/profiles`.

Notes:

- Upgrading or reinstalling does not delete profiles in your data folder.
- Changing the profile folder in Settings does **not** migrate existing profiles; copy them manually.
- Move weight files and `ref.wav` together. `meta.json` alone is not enough.

---

## 9. Documentation center

- Open it from the **Docs** button in any console, or from the Docs card on the index page.
- All user-facing documents are bundled and readable offline; when online, **Check for updates** compares them with the official repository.
- Documents with an English version can switch to it. Documents without one **fall back to Chinese with a clear notice** instead of returning an error.
- Search is fuzzy and accepts everyday phrasing (“no sound”, “won't install”, “OOM”, “401”) and maps it to the right document.
- If the installation package is missing `docs/`, the center explicitly reports that no bundled documents were found rather than pretending everything is up to date.

---

## 10. The one-line workflow

**Install → sign in → create a voice in Easy → train and bind it in Pro → synthesize → integrate from Call when needed → export voice packages for backup.**

If something breaks, start with [Troubleshooting](troubleshooting.md).
