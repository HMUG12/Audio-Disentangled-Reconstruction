# Troubleshooting

Each entry follows **Symptom → Cause → Fix**. Before diving in, do these three things — they solve most cases:

1. Open **Pro → Overview** and click **Run diagnostics**. If fixable items appear, click **Fix issues**.
2. Read the tail of `logs/server.log` in your data folder, plus the training or synthesis log in the Pro console.
3. Open `http://127.0.0.1:9881/api/adr/v1/health` and check `engine_ready`, `engine_stage`, and `engine_error`.

> When filing an issue, include: ADR version, installer variant (CPU / GPU), Windows version, GPU model, the last 50 lines of `server.log`, and the diagnostic report. Never post API keys or private audio publicly.

---

## 1. Installation fails or is blocked by antivirus

**Symptom**

- Double-clicking the installer does nothing, or Windows shows “Windows protected your PC”.
- 360, Huorong, or Defender flags the installer, `adr-desktop.exe`, or files in `runtime`.
- Installation stops halfway with a permission or path error.

**Cause**

- The installer is not commercially code-signed, so SmartScreen treats it as an unknown publisher; security tools often false-positive on bundled Python runtimes and Tauri shells.
- Installation is per-user and needs no administrator rights, but installing into protected folders or folders locked by security software fails.
- Extremely long paths or insufficient disk space.

**Fix**

1. Compare the installer SHA256 with the value on the Releases page; re-download if it differs.
2. SmartScreen: click **More info → Run anyway**.
3. Add the installer and install folder (default `%LOCALAPPDATA%\Programs\ADR Studio`) to your security tool's allow list, then reinstall. Do not disable system protection permanently.
4. Install to a normal folder, or use silent install with an explicit directory:

   ```powershell
   .\ADR-Studio-*-x64-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR="D:\ADR Studio"
   ```

5. Ensure enough free space (about 2 GB CPU, 4 GB GPU, excluding base models and profiles).
6. If it still fails, check `%TEMP%` and the installer log for the Inno Setup error and attach it to your issue.

---

## 2. The engine stays in “loading” forever

**Symptom**

- The prewarm progress bar does not move, or the page stays on “loading engine”.
- The launcher opens, but synthesis queues forever and `/api/adr/v1/health` stays at `importing` / `loading` / `kernel`.
- Logs mention numba cache, CUDA initialization, or the watchdog.

**Cause**

- First launch downloads a ~2 GB base model; a slow or blocked network makes this take a long time.
- Base model incomplete or the model folder was moved.
- numba cache points to a read-only or unwritable path.
- A driver problem makes CUDA initialization retry repeatedly.
- 13th/14th-gen Intel CPUs can crash during compute.

**Fix**

1. Wait. First prewarm includes download plus engine load; the page shows both stages.
2. Network: in **Launcher → Settings → Model download source**, switch to **hf-mirror**, or set a download limit; save and restart the engine.
3. Model integrity: use **Check integrity** in the setup wizard or settings; missing parts are fetched on the next prewarm.
4. numba: set **Advanced → NUMBA cache directory** to a writable path (for example `D:\adr_numba_cache`), then restart.
5. Device: temporarily set **Inference device** to `cpu`. If it starts, the problem is CUDA/driver related.
6. 13th/14th-gen Intel: enable **CPU compatibility mode** (slightly slower, avoids the crash).
7. Read the last 50 lines of `logs/server.log`; `engine_error` usually summarizes the failing stage.
8. If it is truly stuck, exit ADR from the tray and start again. The process-level watchdog restarts the engine after a permanent degradation.

---

## 3. CUDA unavailable or out of VRAM (OOM)

**Symptom**

- Overview shows CPU, or logs say it fell back to CPU.
- Training fails immediately with `CUDA out of memory`, `OOM`, or `Allocation on device`.
- Synthesis returns 503 with a VRAM-related message midway through.
- `nvidia-smi` sees the GPU, but ADR does not use it.

**Cause**

- The **CPU installer** is installed; it has no CUDA runtime.
- The NVIDIA driver is too old, or the GPU is unsupported by the installed CUDA build.
- The GPU is full because of a game, hardware-accelerated browser, or another GPT-SoVITS service.
- The 8 GB recipe is too large for a 4 GB card, or the input text/audio is too long.
- `ADR_GPU_MEM_FRACTION` is set too low.

**Fix**

1. Check `runtime\BUILD_VARIANT.txt` in the install directory. It should say `gpu`; if not, install the GPU package.
2. Run `nvidia-smi`. It should work and show a reasonably recent CUDA version; update the NVIDIA driver if needed.
3. Close other GPU-heavy programs and retry.
4. Switch training to the **4 GB recipe** (batch 1, clips ≤10 s), or reduce epochs.
5. For inference, set the device to `auto` or `cpu`. CPU works, just slower.
6. If `ADR_GPU_MEM_FRACTION` was set during setup or via environment variable, clear it or raise it. Reaching the cap intentionally raises OOM instead of freezing the whole machine.
7. Split long text into shorter sentences, or change the splitting method (`cut3` / `cut5`).
8. Without an NVIDIA GPU, this machine can only do CPU inference. Train on an NVIDIA machine or cloud GPU, then export a `.adrvoice` package and import it here.

---

## 4. Synthesis produces no sound

**Symptom**

- The request succeeds (200), but playback is silent, empty, or extremely short.
- Easy stays on “generating” and eventually fails.
- A caller receives zero bytes or an undecodable file.

**Cause**

- Text is empty, whitespace only, or punctuation only.
- The profile is missing, `ref.wav` is corrupt, or no usable weights are bound.
- The first request is still cold-starting and timed out.
- Wrong output format: `raw` is headerless PCM and cannot be played as WAV.
- Browser autoplay policy blocked playback, or the system volume/output device is wrong.
- The server returned 401 / 500 / 503 and the client saved the error body as audio.

**Fix**

1. Try a normal sentence to rule out text issues.
2. Refresh the profile list in Pro and preview the profile's reference audio.
3. Check `/api/adr/v1/health`. If `engine_ready` is false, wait for prewarm.
4. Switch the format back to `wav`; the downloaded file should start with `RIFF`.
5. Check volume, output device, and mute state; click the page once before pressing play to satisfy autoplay rules.
6. Send one request from the Call console and inspect the HTTP status and error JSON. For 401 see section 7; for 500/503 inspect `server.log`.
7. If only one profile is silent, compare with another profile. If another works, the reference audio or weights for that profile are the problem.

---

## 5. Training stops or fails

**Symptom**

- Training goes from “running” to “failed” or “stopped”.
- The last log line is OOM, `Killed`, file-not-found, timeout, or a traceback.
- **Start training** reports that training is already running.
- Training completes but similarity does not improve, or the voice gets worse.

**Cause**

- Out of VRAM, or two training runs at once (HTTP 409).
- Recording too short, silent, multi-speaker, or full of background music.
- Disk full or output directory not writable.
- Training scripts missing because the installation is incomplete.
- Training timeout (`ADR_TRAIN_TIMEOUT`) or the user pressed stop.
- Too many epochs causing overfitting, or the quality gate was disabled.

**Fix**

1. Press **Stop**, wait for the old process to actually exit, then start again. Do not double-click start.
2. Use the 4 GB recipe and fewer epochs; close other GPU-heavy programs.
3. Use a clean, single-speaker recording of at least one minute; confirm it plays correctly.
4. Check free disk space and output folder permissions.
5. Confirm `scripts/gsv_finetune.py` exists in the install directory; reinstall the full package if not.
6. Keep the **quality gate** on (auto-stop at similarity 0.80) instead of piling on epochs.
7. A failed run does not destroy existing profiles. Retry, or roll back to the previous weight version.
8. Attach the tail of the training log to your issue — it usually names the failing stage.

---

## 6. Models or voice profiles “disappeared”

**Symptom**

- The voice list is empty, or one profile is gone.
- The default profile setting reports that it does not exist.
- Calls return `profile not found`.
- The model library says a model is not downloaded even though it was before.

**Cause**

- The data folder changed: `ADR_DATA_DIR` was set, the profile folder was changed in Settings, or the legacy `F:/ADR_data` no longer exists.
- `data/voices/<name>` was renamed or moved in Explorer without updating `meta.json`.
- `meta.json` is corrupt or has an encoding problem, or weight files were deleted separately.
- Models live in the install directory; reinstalling elsewhere does not bring the old models along.
- ADR is running under a different Windows user, so `%LOCALAPPDATA%` differs.

**Fix**

1. Check the **current** data folder path in Settings and confirm where the profiles actually are.
2. Use **Open** to inspect `data/voices/`.
3. If you changed folders, copy the old `data/voices/` back or point `ADR_DATA_DIR` at the old location.
4. The folder name must match the profile name in `meta.json`; if you rename the folder manually, update the metadata too.
5. Restore a corrupt `meta.json` from backup; if there is no backup, re-import the `.adrvoice` package or recreate the profile.
6. For missing models, run **Check integrity** so prewarm can fetch them, or place the weights back into the model folder using the GPT-SoVITS layout.
7. Make sure multiple ADR processes are not using different data folders at the same time.

---

## 7. 401 / 413 / 429

### 401 Unauthorized

**Symptom:** the API returns `401 unauthorized`, or the console reports an authorization error.

**Cause:** `ADR_TTS_API_KEY` is set, but the request has no credential, the wrong credential, or uses the disabled `?api_key=` query parameter.

**Fix**

- Use one of: `Authorization: Bearer <key>`, `X-API-Key: <key>`, or `?api_key=<key>` (query auth is off by default).
- If the client can only append query parameters, set `ADR_ALLOW_QUERY_API_KEY=1` and accept that the key may appear in logs/history.
- Keys are case-sensitive and must match exactly; separate multiple keys with commas.
- `GET /api/adr/v1/health` is unauthenticated and is a good first check that the service is alive.

### 413 Payload Too Large

**Symptom:** HTTP 413, or a package upload/import fails.

**Cause:** a non-multipart body exceeds `ADR_MAX_BODY_MB` (default 16 MB), text exceeds the limit, a profile package exceeds the import limit, or a single document exceeds 800 KB.

**Fix**

- Shorten the text or synthesize in batches.
- Raise `ADR_MAX_BODY_MB` (server) and `ADR_PROFILE_IMPORT_MAX_MB` (package import), then restart.
- A document 413 means the file exceeds the docs center's 800 KB protection limit; that is expected, not a bug.

### 429 Too Many Requests

**Symptom:** HTTP 429.

**Cause:** **ADR itself does not return 429 in the current version.** A 429 almost certainly comes from a reverse proxy, gateway, or CDN (Nginx, Cloudflare, corporate gateway) applying rate limits.

**Fix**

- Inspect the proxy's rate-limit configuration and response headers (`Retry-After`, `X-RateLimit-*`).
- ADR's own overload signal is **503 `busy: synthesis queue full`**, not 429. For 503, reduce concurrency or retry later; do not treat it as a bad request and retry in a tight loop.

---

## 8. Port already in use

**Symptom**

- The startup banner says 9881 is busy, or the desktop app offers another port.
- `OSError: [WinError 10048] Only one usage of each socket address is normally permitted`.
- `127.0.0.1:9881` shows some other service.

**Cause:** another GPT-SoVITS / TTS service, an old ADR process, or an unrelated program holds port 9881.

**Fix**

1. Find the process:

   ```powershell
   netstat -ano | findstr :9881
   tasklist /FI "PID eq <PID-from-previous-command>"
   ```

2. If it is an old ADR instance, exit from the tray or end the leftover Python process.
3. If it is another GSV service, stop it or give ADR a different port:

   ```powershell
   python -m adr.server -p 9882
   ```

   Desktop users can change **Settings → Advanced → Service port** (`0` means the default 9881 with automatic fallback), then restart the engine.
4. Update N.E.K.O and any callers to the new port.
5. Cannot reach it from the LAN? Confirm it listens on `0.0.0.0`, Windows Firewall allows the port, and an API key is configured.

---

## 9. Documents will not open, or the docs center is empty

**Symptom**

- The Docs drawer spins forever or says the docs API is unavailable.
- It reports that no documents were found in the installation.
- A specific document returns 404 / 502 / 413.
- Opening an HTML file directly shows an empty page.

**Cause**

- The docs API is same-origin (`/api/adr/v1/docs`) and must be served by ADR. Opening files via `file://` has no backend.
- The installer did not copy `docs/`, or only copied part of it.
- The “online” source was selected but the network or repository path is unavailable.
- A single document exceeds the 800 KB limit and is rejected on purpose.
- The desktop shell's prewarm/loading pages do not enable the docs drawer.

**Fix**

1. Open the console through `http://127.0.0.1:9881/` (or the desktop shell), not as a local file.
2. If it reports missing documents, reinstall the complete package; do not copy only the `adr/` folder.
3. Switch back to the **local** source. Online failures degrade to local results and never block the console.
4. A 404 means the slug is not in the whitelist or the upstream repository does not have it yet. Not every internal document is public.
5. A 502 means the official repository is unreachable (offline, proxy, DNS); local documents still work.
6. Missing English versions intentionally fall back to Chinese with a notice.

---

## 10. Language or theme switching misbehaves

**Symptom**

- Clicking **EN / 中文** does nothing, or only half the page changes.
- Colors break or text becomes unreadable after switching themes.
- A refresh reverts to the previous language or theme.

**Cause**

- Browser LocalStorage is disabled, full, or contains invalid values.
- Page scripts are blocked or rewritten by CSP, extensions, or a translation add-on.
- Full-page browser translation runs at the same time as ADR's i18n, producing mixed text.
- The desktop WebView cached an old page.

**Fix**

1. Click the switch again, or force refresh (`Ctrl + F5`).
2. Language is stored in `localStorage` under `adr_lang`; theme under `adr.theme`. Delete both in DevTools → Application → Local Storage, then reload.
3. Turn off full-page translation extensions/features and retry.
4. Make sure JavaScript is enabled and `/i18n*.js` and `/theme.js` are not blocked.
5. Desktop app: fully exit from the tray and restart. If it persists, clear the WebView2 cache and restart.
6. Themes only affect UI colors, never synthesis output. Switching back to **Default** restores the normal palette.

---

## 11. Still stuck?

Collect the following — it covers most cases:

1. Pro → Overview → **Run diagnostics** → copy the report.
2. The last 50 lines of `logs/server.log` (remove API keys and usernames from paths first).
3. The full JSON from `/api/adr/v1/health`.
4. The exact action, time, and whether it reproduces reliably.
5. Installer variant (CPU / GPU), version, and the contents of `runtime\BUILD_VARIANT.txt`.

Then search [GitHub Issues](https://github.com/HMUG12/Audio-Disentangled-Reconstruction/issues) for an existing report. If none matches, open a new one with the information above.
