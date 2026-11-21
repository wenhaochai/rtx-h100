# Fixed-sample, fixed-seed render of the open scene on the first CUDA device, raw EXR (no denoise, no compositing).
# blender -b scene.blend --python render_check.py -- <out.exr> <samples> [seed]
import sys, time, bpy
args = sys.argv[sys.argv.index("--") + 1:]
out, spp = args[:2]
seed = int(args[2]) if len(args) > 2 else 0
prefs = bpy.context.preferences.addons["cycles"].preferences
import os
dev_type = os.environ.get("CHECK_DEVICE", "CUDA")   # OPTIX for the RTX route
prefs.compute_device_type = dev_type
prefs.get_devices()
cuda = [d for d in prefs.devices if d.type == dev_type]
for d in prefs.devices:
    d.use = False
cuda[0].use = True
sc = bpy.context.scene
c = sc.cycles
# classroom keys cycles.seed with an fcurve, which re-evaluates at render time and overrides the seed set below;
# without this, its seed-0 and seed-1 renders are the same image and its noise floor is ~0
ad = sc.animation_data
if ad and ad.action:
    act = ad.action
    # 4.x: Action.fcurves; 5.x: layered actions (layers -> strips -> channelbags -> fcurves)
    bags = [act] if hasattr(act, "fcurves") else [cb for layer in act.layers for strip in layer.strips
                                                  for cb in getattr(strip, "channelbags", [])]
    for bag in bags:
        fc = bag.fcurves.find("cycles.seed")
        if fc:
            bag.fcurves.remove(fc)
            print("render_check removed the cycles.seed fcurve", flush=True)
c.device = "GPU"; c.samples = int(spp); c.use_adaptive_sampling = False; c.time_limit = 0
c.seed = seed; c.use_animated_seed = False; c.use_denoising = False; c.use_auto_tile = False
sc.render.use_compositing = False; sc.render.use_sequencer = False
sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
sc.render.filepath = out
t = time.time()
bpy.ops.render.render(write_still=True)
print(f"render_check device={cuda[0].name} samples={spp} seed={seed} seconds={time.time() - t:.3f} out={out}", flush=True)
