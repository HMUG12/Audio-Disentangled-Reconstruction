# Quick Start

ADR Studio is a local voice-cloning and text-to-speech workbench. You record or import a short voice sample, optionally fine-tune it, then synthesize speech — all on your own computer. Audio, text, and voice profiles never leave the machine.

## 1. Install

Two Windows installers are available:

| Package | Use when |
| --- | --- |
| **CPU** | No NVIDIA GPU, or you only need inference |
| **GPU (CUDA)** | NVIDIA GPU; faster inference and required for training |

Both packages share the same AppId, so you can switch between them in place. Your voice profiles and downloaded base models are preserved.

1. Download the installer from the Releases page and verify its SHA256 against the published value.
2. Run it. Installation is per-user and does not require administrator rights.
3. On first launch, accept the terms, complete the three-step setup wizard, then wait for the prewarm page.

> Silent install: `ADR-Studio-*-x64-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="D:\ADR Studio"`

## 2. First launch and sign-in

The first run does three things:

1. Shows the terms. You must accept before the engine will start.
2. Asks for a **local account**. Defaults: username `ADR`, password `Adr12345@`. The password is stored only as a PBKDF2 hash in your data folder — no cloud account, no telemetry.
3. Prewarms the engine: downloads the ~2 GB base model, then loads it. Progress is shown on the page. Only after this finishes does the launcher appear.

After that, ADR works fully offline. You can change the account later in **Launcher → Settings → Local account**.

Command-line and server deployments do not use this sign-in page. They require either `--agree-terms` or `ADR_AGREE_TERMS=1` before the service will start.

## 3. Choose a console

| Console | Path | Best for |
| --- | --- | --- |
| **Easy** | `/easy` | First-time users: pick a voice, type, listen |
| **Pro** | `/pro` | Training, parameter tuning, curves, model library |
| **Call** | `/call` | Connecting ADR to N.E.K.O or your own code |

Start with **Easy**. Switch to **Pro** when you want to train or fine-tune. Use **Call** when you are ready to integrate.

Every console has a **Docs** button in the header. Language and theme switches are there too.

## 4. Create your first voice

1. Open **Easy** and click **Create a new voice**.
2. Upload a recording: one speaker, quiet room, no music. `wav / mp3 / flac / m4a / ogg` are supported. One minute or more gives the best result.
3. Choose the language spoken in the recording. This sets the default synthesis language for the profile.
4. Name it and start learning. ADR slices, transcribes, and builds the profile in the background.
5. When it finishes, go back and type something to synthesize.

This gives you a zero-shot voice. To make it sound closer to the original, continue with training or use **Improve an existing voice**.

## 5. Train (optional but recommended)

Training lives in **Pro → Voice cloning**.

- Training requires an NVIDIA CUDA GPU. AMD, Intel, and CPU-only machines can synthesize but cannot train.
- 4 GB of VRAM is enough; 8 GB is more comfortable.
- Choose 8 GB (batch 4) or 4 GB (batch 1, clips ≤10 s) depending on your card.
- Keep the quality gate on. It stops training automatically when similarity reaches 0.80.
- Bind the result to a voice profile so it is used for synthesis immediately.

You can stop at any time. The previous weight version is kept, and you can roll back from the Pro console.

## 6. Synthesize

In **Easy**, select the voice, type text, and click **Generate speech**.

In **Pro → Speech synthesis**, you can also control speed, seed, output format, language, sentence splitting, and sampling parameters. The first request loads the model and may take tens of seconds; later requests are fast.

## 7. Connect N.E.K.O

1. Start ADR and note the port in the startup banner (default `9881`).
2. In N.E.K.O, open speech synthesis and choose **GPT-SoVITS** as the provider (OpenAI-compatible also works).
3. Set the API address to `http://127.0.0.1:9881`.
4. Refresh the voice list. Your ADR profiles appear automatically.

For access from other devices on your LAN, set `ADR_TTS_API_KEY` **and** listen on `0.0.0.0`. Otherwise the service is reachable without authentication.

## 8. Where your data lives

In priority order:

1. `ADR_DATA_DIR` environment variable
2. `F:/ADR_data` (legacy path, used if it exists)
3. `%LOCALAPPDATA%\ADR\data`

Voice profiles are under `data/voices/<name>/` (`ref.wav` + `meta.json`). Logs are under `logs/`.

Back up a single voice by exporting a `.adrvoice` package, or copy the whole data folder while ADR is closed.

## 9. Documentation

Open the **Docs** drawer from any console. It contains all user-facing guides, works offline, and can check the official repository for updates when online.

Some documents are available in English. For those that are not, ADR shows the Chinese original with a clear notice instead of failing — and search still works with English phrases such as “no sound” or “OOM”.

If something goes wrong, read [Troubleshooting](troubleshooting.md). For the full workflow, read the [User Guide](guide.md).

## 10. Licensing and responsible use

ADR is licensed under AGPL-3.0-only. Commercial use is allowed under its terms. You must have explicit permission from the owner of any voice you clone, and you may not use ADR for fraud, impersonation, defamation, or evidence fabrication.
