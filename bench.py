#!/usr/bin/env python3
# Usage: bench.py <video-or-audio file> <model dir>   (times model load + separation for each precision)
"""laughless speed benchmark: model load vs separation, precision/batch variants, quality drift vs fp32."""
import sys, time, os, json, shutil, tempfile, subprocess, numpy as np, soundfile as sf, torch
from audio_separator.separator import Separator
clip, model_dir = sys.argv[1], sys.argv[2]
MODEL = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"
work = tempfile.mkdtemp(prefix="lbench-")
wav = os.path.join(work, "in.wav")
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", clip, "-map", "0:a:0", "-ac", "2", "-ar", "44100", wav], check=True)
dur = sf.info(wav).duration
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"device: {dev} | torch {torch.__version__} | hip {torch.version.hip} | cuda {torch.version.cuda} | audio {dur:.1f}s", flush=True)
configs = [("fp32, batch 1", {}, 1), ("fp32, batch 4", {}, 4), ("autocast, batch 1", {"use_autocast": True}, 1),
           ("autocast, batch 4", {"use_autocast": True}, 4), ("native fp16, batch 1", {"use_native_fp16": True}, 1)]
ref = None; rows = []
for name, kw, bs in configs:
    out = os.path.join(work, name.replace(", ", "_").replace(" ", "")); os.makedirs(out, exist_ok=True)
    try:
        t0 = time.time()
        s = Separator(output_dir=out, output_format="WAV", model_file_dir=model_dir, log_level=40,
                      mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": bs, "overlap": 8, "pitch_shift": 0}, **kw)
        s.load_model(model_filename=MODEL); torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_load = time.time() - t0
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t1 = time.time(); files = s.separate(wav)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_sep = time.time() - t1
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        o = [f for f in files if "(other)" in f][0]; a, _ = sf.read(os.path.join(out, os.path.basename(o)))
        a = a.mean(axis=1)
        if ref is None: ref = a; drift = "reference"
        else:
            n = min(len(a), len(ref)); d = a[:n] - ref[:n]
            drift = f"{10*np.log10((d**2).sum()/(ref[:n]**2).sum()+1e-20):.1f} dB"
        rows.append((name, t_load, t_sep, dur / t_sep, peak, drift))
        print(f"{name:22s} load {t_load:5.1f}s  separate {t_sep:6.1f}s  = {dur/t_sep:5.2f}x realtime  peak VRAM {peak:4.1f} GB  drift {drift}", flush=True)
        del s
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name:22s} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
shutil.rmtree(work, ignore_errors=True)
