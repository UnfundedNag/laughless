#!/usr/bin/env python3
"""laughless: add a "no laughs" audio track to sitcom episodes.

It runs an AI crowd-separation model (Mel-Band Roformer "crowd", by aufr33 and viperx, via
python-audio-separator) over the episode's audio and adds the result as an extra audio track.
The original track is kept and stays the default, so you pick "no laughs" in your player.

Default recipe ("surround"): on 5.1/7.1 audio the model is run separately on the centre channel
(dialogue) and on a stereo mix of the other channels, then the two are mixed back together.
In blind listening tests this sounded better than running the model on a plain stereo downmix.

Limits: the laughter is mixed into the same audio as everything else, so this is separation, not
deletion. Expect occasional leftovers, and the pauses the actors leave for laughs stay in.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

__version__ = "0.2.0"

DEFAULT_MODEL = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".ts", ".m2ts", ".mts", ".wmv", ".webm", ".mpg", ".mpeg"}
DEFAULT_TITLE = "No laughs"
DEFAULT_MIN_LAUGH = 8.0    # crowd share of audio energy (%); calibrated: sitcom ~21, drama <= 3.3 (README)
DEFAULT_MIN_EVENTS = 2.5  # laugh-shaped bursts per minute; calibrated: sitcom ~5.6, drama <= 1.0


def log(msg: str) -> None:
    print(f"[laughless] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd[:6])} ...\n{r.stderr[-2000:]}")


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def pick_audio_stream(info: dict, wanted: int | None) -> tuple[int, dict]:
    """Return (index among audio streams, stream dict). Default: the default-flagged audio stream, else the first."""
    audio = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    if not audio:
        raise RuntimeError("no audio stream in file")
    if wanted is not None:
        if wanted >= len(audio):
            raise RuntimeError(f"--audio-stream {wanted}: file has only {len(audio)} audio stream(s)")
        return wanted, audio[wanted]
    for i, s in enumerate(audio):
        if s.get("disposition", {}).get("default"):
            return i, s
    return 0, audio[0]


def channel_names(stream: dict) -> list[str]:
    """Channel names from the layout ('5.1(side)' -> FL FR FC LFE SL SR), via ffmpeg's own layout table."""
    layout = stream.get("channel_layout") or ""
    n = int(stream.get("channels") or 0)
    known = {
        "mono": ["FC"], "stereo": ["FL", "FR"], "2.1": ["FL", "FR", "LFE"], "3.0": ["FL", "FR", "FC"],
        "3.0(back)": ["FL", "FR", "BC"], "4.0": ["FL", "FR", "FC", "BC"], "quad": ["FL", "FR", "BL", "BR"],
        "quad(side)": ["FL", "FR", "SL", "SR"], "3.1": ["FL", "FR", "FC", "LFE"], "5.0": ["FL", "FR", "FC", "BL", "BR"],
        "5.0(side)": ["FL", "FR", "FC", "SL", "SR"], "4.1": ["FL", "FR", "FC", "LFE", "BC"],
        "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"], "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
        "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"], "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
        "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"], "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
        "7.1(wide-side)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
    }
    if layout in known:
        return known[layout]
    # Unknown/empty layout: guess from the channel count the way most encoders do.
    return {1: ["FC"], 2: ["FL", "FR"], 6: ["FL", "FR", "FC", "LFE", "SL", "SR"],
            8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"]}.get(n, [])


def sides_pan(ch: list[str]) -> str:
    """Stereo mix of every channel except the centre (and LFE folded in lightly)."""
    left, right = ["0.7*FL"], ["0.7*FR"]
    surround_l = [c for c in ("SL", "BL") if c in ch]
    surround_r = [c for c in ("SR", "BR") if c in ch]
    w = 0.5 / max(1, len(surround_l))
    left += [f"{w:g}*{c}" for c in surround_l]
    right += [f"{w:g}*{c}" for c in surround_r]
    if "BC" in ch:
        left.append("0.35*BC"); right.append("0.35*BC")
    if "LFE" in ch:
        left.append("0.5*LFE"); right.append("0.5*LFE")
    return f"pan=stereo|c0={'+'.join(left)}|c1={'+'.join(right)}"


class Remover:
    """Loads the separation model once and reuses it for every pass and file."""

    def __init__(self, model: str, model_dir: str | None, overlap: int | None, work: Path):
        from audio_separator.separator import Separator  # imported late so --help works without it
        params = {}
        if overlap:
            params["mdxc_params"] = {"segment_size": 256, "override_model_segment_size": False,
                                     "batch_size": 1, "overlap": overlap, "pitch_shift": 0}
        self.work = work
        self.sep = Separator(output_dir=str(work), output_format="WAV",
                             model_file_dir=model_dir or os.path.expanduser("~/.cache/laughless/models"),
                             log_level=40, **params)
        self.sep.load_model(model_filename=model)

    def strip(self, wav: Path, passes: int) -> Path:
        """Return a wav with the crowd removed (the model's non-crowd stem), `passes` times."""
        cur = wav
        for p in range(passes):
            outs = [self.work / Path(o).name for o in self.sep.separate(str(cur))]
            keep = [o for o in outs if "(other)" in o.name.lower() or "(no crowd)" in o.name.lower()]
            if not keep:
                raise RuntimeError(f"model gave no non-crowd stem: {[o.name for o in outs]}")
            for o in outs:
                if o not in keep:
                    o.unlink(missing_ok=True)
            nxt = self.work / f"{wav.stem}.pass{p + 1}.wav"
            keep[0].rename(nxt)
            cur = nxt
        return cur

    def laugh_score(self, wav: Path) -> tuple[float, float]:
        """Share (percent) of the sample's audio energy that the model assigns to the crowd.

        Calibrated on 5-minute samples (README: Detection): a multi-camera sitcom with an audience
        (Friends) scored 21 %, a drama with music and crowd scenes (The Mentalist, 4 episodes)
        0.2-3.3 %. An earlier frame-count metric could not tell them apart (drama scored up to 29)."""
        import numpy as np
        import soundfile as sf
        outs = [self.work / Path(o).name for o in self.sep.separate(str(wav))]
        crowd = [o for o in outs if "(crowd)" in o.name.lower()]
        if not crowd:
            raise RuntimeError(f"model gave no crowd stem: {[o.name for o in outs]}")
        mix, sr = sf.read(str(wav), always_2d=True)
        cr, _ = sf.read(str(crowd[0]), always_2d=True)
        for o in outs:
            o.unlink(missing_ok=True)
        n = min(len(mix), len(cr))
        m = mix[:n].mean(axis=1)
        c = cr[:n].mean(axis=1)
        total = float((m ** 2).sum())
        if total <= 1e-9:
            return 0.0, 0.0
        share = 100.0 * float((c ** 2).sum()) / total
        # Laugh-shaped bursts: 0.5-6 s stretches where the crowd carries >40 % of the level,
        # starting after a calm second (laughs follow lines; music and ambience are long and steady).
        hop = int(sr * 0.1)
        f = n // hop
        rm = np.sqrt((m[: f * hop].reshape(f, hop) ** 2).mean(axis=1) + 1e-12)
        rc = np.sqrt((c[: f * hop].reshape(f, hop) ** 2).mean(axis=1) + 1e-12)
        act = (rc > 0.4 * rm) & (rm > 10 ** (-50 / 20))
        events, i = 0, 0
        while i < f:
            if act[i]:
                j = i
                while j < f and act[j]:
                    j += 1
                if 5 <= j - i <= 60 and (i == 0 or act[max(0, i - 10):i].mean() < 0.3):
                    events += 1
                i = j
            else:
                i += 1
        return share, events / max(f / 600.0, 1e-9)


def sample_audio(src: Path, idx: int, duration: float, work: Path, windows: int = 10, length: float = 30.0) -> Path:
    """Stereo wav of `windows` slices of `length` s spread over the episode (skipping the first/last 8 %)."""
    out = work / "detect.wav"
    if duration <= windows * length * 1.5:
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", f"0:a:{idx}",
             "-ac", "2", "-ar", "44100", str(out)])
        return out
    lo, hi = duration * 0.08, duration * 0.92 - length
    starts = [lo + (hi - lo) * i / (windows - 1) for i in range(windows)]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for s in starts:
        cmd += ["-ss", f"{s:.1f}", "-t", str(length), "-i", str(src)]
    chains = "".join(f"[{i}:a:{idx}]aresample=44100,aformat=channel_layouts=stereo[a{i}];" for i in range(windows))
    cmd += ["-filter_complex", chains + "".join(f"[a{i}]" for i in range(windows)) + f"concat=n={windows}:v=0:a=1[o]",
            "-map", "[o]", str(out)]
    run(cmd)
    return out


def build_track(src: Path, idx: int, stream: dict, args, rm: Remover, work: Path) -> Path:
    ch = channel_names(stream)
    mode = args.mode
    if mode in ("surround", "center") and "FC" not in ch:
        log(f"audio is {stream.get('channel_layout') or str(stream.get('channels')) + ' ch'} (no centre channel): using --mode stereo")
        mode = "stereo"
    cut = []
    if args.start:
        cut += ["-ss", str(args.start)]
    if args.duration:
        cut += ["-t", str(args.duration)]
    base = ["ffmpeg", "-nostdin", "-v", "error", "-y", *cut, "-i", str(src), "-map", f"0:a:{idx}", "-ar", "44100"]

    def extract(name: str, af: str | None) -> Path:
        out = work / f"{name}.wav"
        run(base + (["-af", af] if af else ["-ac", "2"]) + [str(out)])
        return out

    orig = extract("orig", None)
    if mode == "stereo":
        mixed = rm.strip(orig, args.passes)
    elif mode == "center":
        mixed = rm.strip(extract("center", "pan=stereo|c0=FC|c1=FC"), args.passes)
    else:  # surround
        c = rm.strip(extract("center", "pan=stereo|c0=FC|c1=FC"), args.passes)
        s = rm.strip(extract("sides", sides_pan(ch)), args.passes)
        mixed = work / "surround.wav"
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(c), "-i", str(s), "-filter_complex",
             f"[0:a]volume={args.center_gain}[c];[1:a]volume={args.sides_gain}[s];[c][s]amix=inputs=2:normalize=0",
             str(mixed)])
    # Optional: push harder (over-subtract the estimated crowd) and/or blend some original back in.
    w_m = args.strength
    w_o = (1.0 - args.strength) + args.blend / 100.0
    final = work / "final.wav"
    if abs(w_m - 1.0) > 1e-9 or abs(w_o) > 1e-9:
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(mixed), "-i", str(orig), "-filter_complex",
             f"amix=inputs=2:weights={w_m:g} {w_o:g}:normalize=0", str(final)])
    else:
        shutil.copy(mixed, final)
    return final


def out_path(src: Path, args) -> tuple[Path, str]:
    """Where to write, and which container. mp4/mkv keep their container; everything else becomes mkv."""
    ext = src.suffix.lower()
    fmt = "mp4" if ext in (".mp4", ".m4v", ".mov") else "matroska"
    new_ext = ".mp4" if fmt == "mp4" else ".mkv"
    if args.in_place:
        return src.with_suffix(new_ext), fmt
    as_dir = args.output and (args.output.endswith(os.sep) or Path(args.output).is_dir())
    if args.output and not as_dir and len(args.inputs) == 1 and not Path(args.inputs[0]).is_dir():
        o = Path(args.output)
        return o, ("mp4" if o.suffix.lower() in (".mp4", ".m4v", ".mov") else "matroska")
    d = Path(args.output) if args.output else src.parent
    return d / f"{src.stem}.laughless{new_ext}", fmt


def mux(src: Path, track: Path, stream: dict, info: dict, args, dst: Path, fmt: str, work: Path) -> Path:
    n_audio = sum(1 for s in info["streams"] if s.get("codec_type") == "audio")
    lang = (stream.get("tags") or {}).get("language", "und")
    title = args.title
    tmp = work / ("muxed" + (".mp4" if fmt == "mp4" else ".mkv"))
    codec = {"aac": ["-c:a:" + str(n_audio), "aac", "-b:a:" + str(n_audio), args.bitrate],
             "ac3": ["-c:a:" + str(n_audio), "ac3", "-b:a:" + str(n_audio), args.bitrate],
             "opus": ["-c:a:" + str(n_audio), "libopus", "-b:a:" + str(n_audio), args.bitrate],
             "flac": ["-c:a:" + str(n_audio), "flac"]}[args.codec]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-i", str(track),
           "-map", "0", "-map", "1:a", "-c", "copy", *codec,
           f"-metadata:s:a:{n_audio}", f"title={title}", f"-metadata:s:a:{n_audio}", f"handler_name={title}",
           f"-metadata:s:a:{n_audio}", f"language={lang}",
           f"-disposition:a:{n_audio}", "0", "-f", fmt, str(tmp)]
    if fmt == "mp4":
        cmd[cmd.index("-f"):cmd.index("-f")] = ["-movflags", "+faststart"]
    try:
        run(cmd)
    except RuntimeError as e:
        if fmt != "mp4":
            raise
        log(f"mp4 can't hold every stream of this file ({str(e).splitlines()[-1][:120]}); writing mkv instead")
        dst = dst.with_suffix(".mkv")
        return mux(src, track, stream, info, args, dst, "matroska", work)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    shutil.move(str(tmp), part)
    if args.in_place and dst == src and args.backup:
        shutil.copy2(src, src.with_name(src.name + ".orig"))
    os.replace(part, dst)
    if args.in_place and dst != src:  # container changed (e.g. avi -> mkv): old file goes away
        if args.backup:
            src.rename(src.with_name(src.name + ".orig"))
        else:
            src.unlink()
    return dst


def already_done(info: dict, title: str) -> bool:
    return any(s.get("codec_type") == "audio" and title in ((s.get("tags") or {}).get("title"),
                                                          (s.get("tags") or {}).get("handler_name"))
               for s in info["streams"])


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for i in inputs:
        p = Path(i)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS and ".laughless." not in f.name)
        elif p.exists():
            files.append(p)
        else:
            log(f"not found: {p}")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="laughless", description="Add a 'no laughs' audio track to sitcom episodes.")
    ap.add_argument("inputs", nargs="+", help="video files and/or folders (searched recursively)")
    ap.add_argument("--mode", choices=["surround", "stereo", "center"], default="surround",
                    help="surround (default): separate centre and the rest of a 5.1/7.1 mix, then remix; "
                         "stereo: run on a stereo downmix; center: centre channel only. "
                         "Files without a centre channel always use stereo.")
    ap.add_argument("--passes", type=int, default=1, help="run the model N times (2 catches more, can thin voices)")
    ap.add_argument("--blend", type=float, default=0.0, metavar="PCT",
                    help="mix PCT%% of the original back in to soften the silent pauses (e.g. 10)")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="1.0 = as the model gives it; >1 subtracts the laughter harder (e.g. 1.25)")
    ap.add_argument("--center-gain", type=float, default=0.8, help="surround mode: level of the centre/dialogue part")
    ap.add_argument("--sides-gain", type=float, default=0.7, help="surround mode: level of the rest of the mix")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="any separation model audio-separator knows")
    ap.add_argument("--model-dir", help="where models are cached (default ~/.cache/laughless/models)")
    ap.add_argument("--overlap", type=int, help="Roformer overlap (default: model's own; 16 is slower, smoother)")
    ap.add_argument("--audio-stream", type=int, help="which audio stream to use (0-based); default = the default one")
    ap.add_argument("--start", type=float, help="preview: start at this many seconds")
    ap.add_argument("--duration", type=float, help="preview: only this many seconds")
    ap.add_argument("-o", "--output", help="output file (one input) or folder; default: <name>.laughless.<ext> next to it")
    ap.add_argument("--in-place", action="store_true", help="replace the input file (keeps all its streams)")
    ap.add_argument("--backup", action="store_true", help="with --in-place: keep the original as <name>.orig")
    ap.add_argument("--title", default=DEFAULT_TITLE, help=f"title of the new track (default '{DEFAULT_TITLE}')")
    ap.add_argument("--codec", choices=["aac", "ac3", "opus", "flac"], default="aac", help="codec of the new track")
    ap.add_argument("--bitrate", default="192k", help="bitrate of the new track (ignored for flac)")
    ap.add_argument("--audio-only", metavar="FILE.wav", help="don't mux: just write the processed audio to this wav")
    ap.add_argument("--force", action="store_true", help="process even if a track with --title already exists")
    ap.add_argument("--no-detect", action="store_true",
                    help="skip the laugh-track check (by default files without a laugh track are left alone)")
    ap.add_argument("--min-laugh", type=float, default=DEFAULT_MIN_LAUGH, metavar="PCT",
                    help=f"min. crowd share of the audio energy for a laugh track (default {DEFAULT_MIN_LAUGH:g})")
    ap.add_argument("--min-events", type=float, default=DEFAULT_MIN_EVENTS, metavar="N",
                    help=f"min. laugh-shaped bursts per minute for a laugh track (default {DEFAULT_MIN_EVENTS:g})")
    ap.add_argument("--detect-only", action="store_true", help="only print each file's laugh score, change nothing")
    ap.add_argument("--keep-temp", action="store_true", help="keep the working folder (for debugging)")
    ap.add_argument("--version", action="version", version=f"laughless {__version__}")
    args = ap.parse_args(argv)
    if args.in_place and (args.start or args.duration):
        ap.error("--in-place with --start/--duration would cut the episode; preview to a separate file instead")

    files = collect(args.inputs)
    if not files:
        log("nothing to do")
        return 1
    rm = None
    failures = 0
    for src in files:
        try:
            info = ffprobe(src)
            if already_done(info, args.title) and not args.force:
                log(f"skip (already has a '{args.title}' track): {src}")
                continue
            idx, stream = pick_audio_stream(info, args.audio_stream)
            log(f"{src.name}: audio #{idx} {stream.get('codec_name')} {stream.get('channel_layout') or stream.get('channels')} "
                f"-> mode {args.mode}, passes {args.passes}")
            work = Path(tempfile.mkdtemp(prefix="laughless-"))
            try:
                if rm is None:
                    rm = Remover(args.model, args.model_dir, args.overlap, work)
                rm.work = work
                rm.sep.output_dir = str(work)
                if getattr(rm.sep, "model_instance", None) is not None:
                    rm.sep.model_instance.output_dir = str(work)
                if args.detect_only or not args.no_detect:
                    dur = float(info.get("format", {}).get("duration") or stream.get("duration") or 0)
                    share, epm = rm.laugh_score(sample_audio(src, idx, dur, work))
                    laughs = share >= args.min_laugh and epm >= args.min_events
                    verdict = "laugh track" if laughs else "no laugh track"
                    if args.detect_only:
                        print(f"crowd {share:5.1f}%  bursts/min {epm:4.1f}  {verdict:15s}  {src}")
                        continue
                    if not laughs:
                        log(f"skip (no laugh track: crowd {share:.1f}% / {epm:.1f} bursts per min, need >= "
                            f"{args.min_laugh:g}% and {args.min_events:g}; --no-detect to process anyway): {src}")
                        continue
                    log(f"laugh track detected (crowd {share:.1f}%, {epm:.1f} bursts/min): processing")
                track = build_track(src, idx, stream, args, rm, work)
                if args.audio_only:
                    shutil.copy(track, args.audio_only)
                    log(f"wrote {args.audio_only}")
                    continue
                dst, fmt = out_path(src, args)
                if args.start or args.duration:  # preview: cut the video to match the processed audio
                    pv = work / ("preview" + src.suffix)
                    cut = (["-ss", str(args.start)] if args.start else []) + (["-t", str(args.duration)] if args.duration else [])
                    run(["ffmpeg", "-nostdin", "-v", "error", "-y", *cut, "-i", str(src), "-map", "0", "-c", "copy", str(pv)])
                    info, src_for_mux = ffprobe(pv), pv
                else:
                    src_for_mux = src
                out = mux(src_for_mux, track, stream, info, args, dst, fmt, work)
                log(f"wrote {out}")
            finally:
                if args.keep_temp:
                    log(f"temp kept: {work}")
                else:
                    shutil.rmtree(work, ignore_errors=True)
        except Exception as e:  # keep going with the next file in batch mode
            failures += 1
            log(f"FAILED {src}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
