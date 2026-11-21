# The Open Data benchmark without the launcher (which needs opendata.blender.org even to run): Blender with the
# launcher's own arguments and benchmark script, a warm-up process then the timed one per scene, and the launcher's
# stats parsed from Cycles' "Rendered N samples in T seconds" line. Prints the launcher's --json shape.
# direct_bench.py <blender> <scenes dir> <benchmark main.py> <scene> ...
# $SCENE_ENV ("<scene>:K=V,K=V;<scene>:K=V") adds environment variables for one scene's processes (a per-scene cubin).
import json, os, re, subprocess, sys
blender, scenes_dir, main_py, scenes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
import glob
scene_env = {}
for part in filter(None, os.environ.get("SCENE_ENV", "").split(";")):
    name, kvs = part.split(":", 1)
    scene_env[name] = dict(kv.split("=", 1) for kv in kvs.split(",") if kv)
out = []
for s in scenes:
    env = {**os.environ, **scene_env.get(s, {})}
    if s in scene_env:
        print(f"scene {s} env {scene_env[s]}", file=sys.stderr, flush=True)
    blend = glob.glob(f"{scenes_dir}/*/{s}/main.blend")[0]
    base = [blender, "--background", "--factory-startup", "-noaudio", "--debug-cycles", "--enable-autoexec",
            "--engine", "CYCLES", blend] + (["--python-expr", os.environ["PRE_PY"]] if os.environ.get("PRE_PY") else []) + [
            "--python", main_py, "--", "--device-type", os.environ.get("BENCH_DEVICE", "CUDA")]
    w = subprocess.run(base + ["--warm-up"], capture_output=True, text=True, env=env)
    if w.returncode:
        sys.exit(f"direct_bench: warm-up of {s} failed ({w.returncode}):\n{(w.stdout + w.stderr)[-3000:]}")
    reps = int(os.environ.get("BENCH_REPEATS", "1"))   # ours: warm repeats in one process (screening only)
    timed = base
    if reps > 1:
        i = base.index("--python")
        timed = base[:i] + ["--python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_repeat.py")] + base[i + 2:]
        env = {**env, "BENCH_MAIN": main_py}
    r = subprocess.run(timed, capture_output=True, text=True, env=env)
    log = r.stdout + r.stderr
    if os.environ.get("BENCH_LOG"):   # the full Blender log, e.g. for the KDF/KDW lines of a CYCLES_KDATA_DUMP cubin
        open(os.environ["BENCH_LOG"], "a").write(f"=== {s}\n{log}")
    if r.returncode:
        sys.exit(f"direct_bench: {s} failed ({r.returncode}):\n{log[-3000:]}")
    for line in log.splitlines():   # anything the SVM path says about itself goes to the item log
        if "SVM" in line or "CYCLES_CUDA_CUBIN" in line or "sort partitions" in line or "L2 persist" in line:
            print(f"{s}: {line}", file=sys.stderr, flush=True)
    # 4.x: "Rendered N samples in T seconds" at line start; 5.x prefixes log lines ("00:05.9  cycles | Rendered ...")
    # and its per-batch lines go on with " (x seconds per sample), ...": take the summary lines that end in "seconds".
    m = re.findall(r"(?:^|\| )Rendered (\d+) samples in ([\d.]+) seconds\s*$", log, re.M)
    if not m:
        sys.exit(f"direct_bench: no 'Rendered N samples in T seconds' line for {s}")
    tot = re.findall(r"Total render time: ([\d.]+)", log)
    nosync = re.findall(r"Render time \(without synchronization\): ([\d.]+)", log)
    peak = re.findall(r"Peak: ([\d.]+)M", log) or re.findall(r"Peak: [\d,]+ \(([\d.]+)M\)", log)
    dev = json.loads(re.findall(r"Benchmark JSON Data: (.*)", log)[-1])["device_info"]
    for k, (ns, ts) in enumerate(m[-reps:] if reps > 1 else m[-1:]):
        n, t = int(ns), float(ts)
        out.append({"scene": {"label": s}, "device_info": dev, "stats": {
            "number_of_samples": n, "time_for_samples": t, "samples_per_minute": n / t * 60,
            "total_render_time": float(tot[-1]), "render_time_no_sync": float(nosync[-1]),
            "device_peak_memory": float(peak[-1]) if peak else None, "time_limit": 30,
            **({"repeat": k} if reps > 1 else {})}})
print(json.dumps(out))
