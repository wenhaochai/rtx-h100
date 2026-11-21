# Scene settings of the nrc configuration (PRE_PY, run by Blender before the benchmark script and the check render).
# The kernel's compile-time KernelData (configs/<scene>/kdata_const.h) was dumped with exactly these settings;
# a cubin built with -DCYCLES_KDATA_CONST ignores any other value of the folded fields.
import os, bpy
scene = os.path.basename(os.path.dirname(bpy.data.filepath))
c = bpy.context.scene.cycles
# 1. only the Combined pass: every other render pass off on every view layer (the beauty image is unchanged)
for vl in bpy.context.scene.view_layers:
    for obj in (vl, vl.cycles):
        for p in dir(obj):
            if p.startswith("use_pass_") and p != "use_pass_combined":
                try:
                    if getattr(obj, p) is True:
                        setattr(obj, p, False)
                except Exception:
                    pass
# 2. light threshold >= 0.2, scrambling distance 0.0625, glossy bounces <= 2 (all scenes)
c.light_sampling_threshold = max(c.light_sampling_threshold, 0.2)
c.auto_scrambling_distance = False
c.scrambling_distance = 0.0625
c.glossy_bounces = min(c.glossy_bounces, 2)
# 3. per scene
s = {"monster": {"scr": 0.0, "lt": 0.2}, "junkshop": {"scr": 0.0, "lt": 0.2}, "classroom": {"scr": 0.0, "lt": 0.35}}.get(scene, {})
if "scr" in s:
    c.scrambling_distance = s["scr"]
if "lt" in s:
    c.light_sampling_threshold = max(c.light_sampling_threshold, s["lt"])
print(f"scene_settings {scene}: scrambling={c.scrambling_distance} light_threshold={c.light_sampling_threshold} "
      f"glossy={c.glossy_bounces}", flush=True)
