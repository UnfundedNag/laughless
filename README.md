# laughless

Add a **"No laughs" audio track** to sitcom episodes: the same episode, with the studio audience / laugh
track removed by an AI separation model. Your original audio stays untouched and remains the default track;
pick "No laughs" in Jellyfin, Plex, VLC, mpv or whatever you watch with.

> **Status: v0.1, early.** It works well on the episodes it has been tried on, but the defaults were tuned
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

## Limits (be honest with yourself)

- **The pauses stay.** Actors wait for the laugh, so scenes can feel oddly quiet. `--blend 10` softens this.
- **Leftovers.** Big or overlapping laughs can leave traces, and the model sometimes trims a little of a
  voice that overlaps a laugh.
- **Speed.** RoFormer is heavy. On a modern NVIDIA GPU, an episode takes minutes. **On CPU it's very slow**:
  roughly 10× the episode length per pass, and surround mode does two passes. Use `--start/--duration`
  to preview a short clip first.
- The new track is stereo, even when the original is 5.1.

## Install

Requirements: Python 3.10+, `ffmpeg`/`ffprobe` on your PATH.

```bash
python3 -m venv ~/.venvs/laughless
~/.venvs/laughless/bin/pip install "audio-separator[gpu]"   # NVIDIA GPU (CUDA)
# or: ~/.venvs/laughless/bin/pip install "audio-separator[cpu]"
curl -LO https://raw.githubusercontent.com/UnfundedNag/laughless/main/laughless.py
```

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
