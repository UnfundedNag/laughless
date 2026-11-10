# laughless

Add a **"No laughs" audio track** to sitcom episodes: the same episode, with the studio audience / laugh
track removed by an AI separation model. Your original audio stays untouched and remains the default track;
pick "No laughs" in Jellyfin, Plex, VLC, mpv or whatever you watch with.

> **Status: early (v0.3).** It works well on the episodes it has been tried on, but the defaults were tuned
> by ear on a handful of clips. Feedback and comparison clips are welcome.

## How it works

The laughter isn't a separate track. It's mixed into the same audio as the dialogue, music and effects.
So this is **source separation, not deletion**: a neural network trained to pull crowd noise (applause,
cheering, laughter) out of recordings estimates "everything except the crowd", and that becomes the new track.

The default recipe (`--mode surround`) makes use of how sitcoms are mixed in 5.1:

1. The **centre channel** (mostly dialogue) and a **stereo mix of the other channels** (where most of the
   audience sits) are extracted separately.
2. The crowd model runs on each part on its own.
3. The two cleaned parts are mixed back into stereo (centre at 0.8, the rest at 0.7).

Files without a centre channel (stereo, mono) automatically use `--mode stereo` (the model on the plain mix).

The model is **Mel-Band RoFormer "crowd" by aufr33 and viperx**, run through
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator).

### How the default was chosen

On a 3-minute clip from a Blu-ray Friends episode, 15 variants were compared by ear:

| Variant | Verdict |
|---|---|
| Original mix | reference |
| Centre channel only, no AI | laughs still there, sounds thin |
| UVR-MDX-NET Crowd HQ (stereo) | not acceptable |
| **Mel-Band RoFormer crowd (stereo)** | good: `--mode stereo` |
| RoFormer on the centre channel only | `--mode center` |
| UVR-MDX-NET on the centre channel | not acceptable |
| **RoFormer on centre + rest separately, remixed** | **best: the default** |
| Blend of the MDX and RoFormer results | not acceptable |
| RoFormer + 10 % / 25 % of the original | `--blend 10` / `--blend 25` |
| RoFormer "extra strength" | `--strength 1.25` |
| RoFormer run twice | about the same as the default, twice the time: `--passes 2` |
| Remix with the centre louder | `--center-gain` / `--sides-gain` |

Every variant is still a flag, so you can try them on your own shows.

## Laugh-track detection (skips shows that don't need it)

Before processing, laughless samples 10 × 30 s spread over the episode, runs the crowd model on just
that, and measures two things:

- **crowd share**: how much of the audio energy the model calls "crowd", and
- **laugh bursts per minute**: short (0.5–6 s) crowd-dominated bursts that start after a calm moment.
  Laughter comes in bursts after lines; music and ambience are long and steady.

A file is processed only when **both** are high (defaults `--min-laugh 8` % and `--min-events 2.5`/min).
Otherwise it's skipped with a message, so a drama in the same folder is left alone.
`--detect-only` prints the numbers without changing anything; `--no-detect` processes regardless.

Calibration so far (more shows welcome):

| Show | Crowd share | Bursts/min | Verdict |
|---|---|---|---|
| Friends S03E12 (studio audience) | 21.3 % | 5.8 | laugh track |
| The Mentalist S01E01–E08 (drama) | 0.2–3.3 % | 0.2–1.6 | no laugh track |

A crowd share alone is fragile: with only 6 samples, one drama episode with loud crowd scenes reached 7.9 %,
so laughless takes 10 samples and requires the burst count too. An earlier "frames with any crowd" metric scored dramas up to 29 % and was dropped.

## GPUs: NVIDIA, AMD, CPU

laughless itself has no GPU-specific code. The RoFormer model runs in **PyTorch**, and audio-separator uses
whatever PyTorch reports (`torch.cuda.is_available()`), so NVIDIA (CUDA builds of PyTorch) and **AMD (ROCm
builds)** both work, and CPU works too (slowly). The default `--precision auto` uses **mixed precision on a GPU**
and full precision on CPU.

### Speed: RTX 3090 vs Radeon AI PRO R9700

Model separation time for 60.7 s of audio, one pass, measured with `bench.py` in this repo (model load excluded):

| Precision | RTX 3090 (CUDA 13.0) | R9700 (ROCm 7.1 wheels, gfx1201) | Difference vs fp32 output |
|---|---|---|---|
| fp32 (`--precision fp32`) | 18.9 s, 3.2× real time | 82.0 s, **0.74× real time** | reference |
| mixed / autocast (**default on GPU**) | 10.0 s, 6.1× | 13.0 s, 4.7× | −71 to −72 dB |
| native fp16 (`--precision fp16`) | 9.6 s, 6.4× | 11.5 s, 5.3× | −68 to −69 dB |

- **Half precision is the big lever**: ~2× on the 3090 and **~7× on the R9700**. In fp32 the R9700 is slower
  than real time; with half precision it's within ~20 % of the 3090. RDNA4's matrix units only speed up
  16-bit maths. Differences of ~70 dB down are inaudible.
- `--precision fp16` also halves GPU memory (about 1 GB instead of 1.7–2 GB), handy on a shared card.
- With the default precision, the NVIDIA and AMD outputs are the same to well below audibility (fp32 outputs
  matched with correlation 1.0, −96 dB).
- Real-world on the 3090: a full 22-minute 5.1 episode (two passes, surround mode) took ~10 min in fp32,
  so roughly 5–6 min with the default. Model load is 3–8 s per run, once per batch.

## Limits (be honest with yourself)

- **The pauses stay.** Actors wait for the laugh, so scenes can feel oddly quiet. `--blend 10` softens this.
- **Leftovers.** Big or overlapping laughs can leave traces, and the model sometimes trims a little of a
  voice that overlaps a laugh.
- **Speed.** RoFormer is heavy. On a recent NVIDIA or AMD GPU an episode takes minutes (see the speed table).
  **On CPU it's very slow**: roughly 10× the episode length per pass, and surround mode does two passes.
  Use `--start/--duration` to preview a short clip first.
- The new track is stereo, even when the original is 5.1.

## Install

Requirements: Python 3.10+, `ffmpeg`/`ffprobe` on your PATH. NVIDIA (CUDA), AMD (ROCm) or CPU.

```bash
python3 -m venv ~/.venvs/laughless
~/.venvs/laughless/bin/pip install "audio-separator[gpu]"   # NVIDIA GPU (CUDA)
# or: ~/.venvs/laughless/bin/pip install "audio-separator[cpu]"
curl -LO https://raw.githubusercontent.com/UnfundedNag/laughless/main/laughless.py
```

**AMD (ROCm): what you must do** (tested on a Radeon AI PRO R9700, Ubuntu 24.04, ROCm 7.2.1):

1. Install **PyTorch for ROCm first**, before audio-separator, so pip doesn't pull a CUDA or CPU build:
   ```bash
   ~/.venvs/laughless/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
   ```
2. Install audio-separator **without** the `[gpu]` extra (that one is CUDA), **plus `onnxruntime`**. audio-separator
   always imports it, even though RoFormer runs in PyTorch on the GPU:
   ```bash
   ~/.venvs/laughless/bin/pip install audio-separator onnxruntime audioread
   ```
3. Have a C compiler and Python headers for its dependencies (`sudo apt install build-essential python3-dev`).
4. Make sure the **model folder is writable** (`--model-dir`). audio-separator writes small index files next to the
   model, and a read-only folder fails with "Permission denied: …download_checks.json".
5. Check: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should
   print `True` and your Radeon. laughless prints the device and backend (`ROCm …`) when it starts.
6. Keep the default `--precision auto` (or `fp16`). **fp32 is about 7× slower on RDNA4.**

On Python 3.13 you may also need `pip install audioread audioop-lts standard-aifc standard-sunau`.
For CPU-only installs, install `torch torchvision torchaudio` together from
`--index-url https://download.pytorch.org/whl/cpu`, so their versions match.

The model (~900 MB) is downloaded on first run to `~/.cache/laughless/models`.

## Use

```bash
# Preview 3 minutes first (writes "Episode.laughless.mkv" next to the file)
laughless.py "Friends S03E12.mkv" --start 240 --duration 180

# Whole episode, new file next to the original
laughless.py "Friends S03E12.mkv"

# Whole season folder, replacing each file in place (keeps every stream, adds the track)
laughless.py "/tv/Friends/Season 3" --in-place --backup

# Other recipes
laughless.py ep.mkv --mode stereo           # model on the stereo mix
laughless.py ep.mkv --blend 10              # keep 10 % of the original to soften pauses
laughless.py ep.mkv --passes 2              # run the model twice
laughless.py ep.mkv --audio-only out.wav    # just the processed audio
```

Run `laughless.py --help` for every option.

### Files it handles

- **Any audio codec ffmpeg can decode**: AC3, E-AC3, DTS, TrueHD, AAC, FLAC, Opus, PCM, MP3, …
- **Channel layouts**: 5.1 and 7.1 (side or back surrounds) use surround mode. Stereo and mono fall back to
  stereo mode.
- **Containers**: MKV stays MKV (all video, audio, subtitle and attachment streams kept). MP4 stays MP4 unless
  it can't hold every stream (e.g. Blu-ray PGS subtitles), in which case it writes MKV. AVI, TS, M2TS, WMV and
  others are written as MKV.
- The new track gets the **source track's language** and the title "No laughs" (`--title`), and is **not**
  the default track. Files that already have a track with that title are skipped (`--force` to redo).
- By default the default-flagged audio stream is used. Pick another with `--audio-stream N`.

## Credits and licenses

- This script: MIT (see `LICENSE`).
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT), which also
  downloads and runs the model.
- **Model: Mel-Band RoFormer crowd, by aufr33 and viperx**, from the
  [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) community.
  This repo **does not include or redistribute the model**. I could not find a license statement from its
  authors; check its terms before any commercial use.
- Mel-Band RoFormer architecture: Wang et al., *Mel-Band RoFormer for Music Source Separation* (ISMIR 2023 LBD).
- An earlier take on the same itch: [laughr](https://github.com/jeffgreenca/laughr) (2018), which mutes the
  laugh sections instead of separating them.
