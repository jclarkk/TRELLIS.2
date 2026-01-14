import argparse
import os
import time

# Set environment variables relative to app.py configuration
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

def parse_args():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Inference CLI")
    parser.add_argument("--images", type=str, nargs='+', required=True, help="Path(s) to input image(s)")
    parser.add_argument("--output", type=str, required=True, help="Path to output GLB file")
    
    # Model and Generation Settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--randomize_seed", action="store_true", help="Randomize seed")
    parser.add_argument("--resolution", type=str, default="1024", choices=["512", "1024", "1536", "2048"], help="Generation resolution")
    parser.add_argument("--no_texture_gen", action="store_true", help="Skip texture generation")
    parser.add_argument("--no_pbr", action="store_true", help="Does not attach the PBR textures to the final GLB")
    parser.add_argument("--max_num_tokens", type=int, default=49152, help="Max number of tokens")

    parser.add_argument("--low_vram", action="store_true", help="Enable low VRAM mode")

    # Stage 1: Sparse Structure
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.7)
    parser.add_argument("--ss_sampling_steps", type=int, default=12)
    parser.add_argument("--ss_rescale_t", type=float, default=5.0)

    # Stage 2: Shape Generation
    parser.add_argument("--shape_slat_guidance_strength", type=float, default=7.5)
    parser.add_argument("--shape_slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--shape_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--shape_slat_rescale_t", type=float, default=3.0)

    # Stage 3: Texture Generation
    parser.add_argument("--tex_slat_guidance_strength", type=float, default=1.0)
    parser.add_argument("--tex_slat_guidance_rescale", type=float, default=0.0)
    parser.add_argument("--tex_slat_sampling_steps", type=int, default=12)
    parser.add_argument("--tex_slat_rescale_t", type=float, default=3.0)

    # Export Settings
    parser.add_argument("--decimation_target", type=int, default=500000, help="Target face count for decimation")
    parser.add_argument("--texture_size", type=int, default=2048, choices=[1024, 2048, 4096], help="Texture size")
    parser.add_argument("--remesh_method", type=str, default="dual_contouring", choices=["dual_contouring", "faithful_contouring", "none"], help="Remesh method")
    parser.add_argument("--simplify_method", type=str, default="cumesh", choices=["cumesh", "meshlib"], help="Simplify method")
    parser.add_argument("--prune_invisible_faces", type=bool, default=True, help="Prune invisible faces")

    return parser.parse_args()

def main():
    args = parse_args()

    # Seed
    seed = np.random.randint(0, np.iinfo(np.int32).max) if args.randomize_seed else args.seed

    t0 = time.time()
    
    # Load Pipeline
    print(f"Loading pipeline (Texture Models: {not args.no_texture_gen})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B', load_texture_models=not args.no_texture_gen, low_vram=args.low_vram)
    pipeline.cuda()

    t1 = time.time()
    print(f"Pipeline loaded in {t1 - t0:.2f} seconds.")

    # Load and Preprocess Images
    images = []
    for img_path in args.images:
        print(f"Processing image: {img_path}")
        img = Image.open(img_path)
        img = pipeline.preprocess_image(img)
        images.append(img)

    t2 = time.time()
    print(f"Images preprocessed in {t2 - t1:.2f} seconds")

    # Run Pipeline
    print("Running generation...")
    outputs, latents = pipeline.run(
        images,
        seed=seed,
        preprocess_image=False, # Already done
        sparse_structure_sampler_params={
            "steps": args.ss_sampling_steps,
            "guidance_strength": args.ss_guidance_strength,
            "guidance_rescale": args.ss_guidance_rescale,
            "rescale_t": args.ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": args.shape_slat_sampling_steps,
            "guidance_strength": args.shape_slat_guidance_strength,
            "guidance_rescale": args.shape_slat_guidance_rescale,
            "rescale_t": args.shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": args.tex_slat_sampling_steps,
            "guidance_strength": args.tex_slat_guidance_strength,
            "guidance_rescale": args.tex_slat_guidance_rescale,
            "rescale_t": args.tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
            "2048": "2048_cascade",
        }[args.resolution],
        return_latent=True,
        max_num_tokens=args.max_num_tokens,
        no_texture_gen=args.no_texture_gen,
    )

    t3 = time.time()
    print(f"Pipeline execution completed in {t3 - t2:.2f} seconds")

    # Extract GLB
    print("Extracting GLB...")
    shape_slat, tex_slat, res = latents
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    
    # Prune config
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        simplify_method=args.simplify_method,
        texture_extraction=not args.no_texture_gen,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        remesh_method=args.remesh_method,
        prune_invisible=args.prune_invisible_faces,
        use_tqdm=True,
        no_pbr=args.no_pbr,
    )

    t4 = time.time()
    print(f"GLB extracted in {t4 - t3:.2f} seconds")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    glb.export(args.output, extension_webp=True)
    print(f"Saved GLB to {args.output}")

if __name__ == "__main__":
    main()
