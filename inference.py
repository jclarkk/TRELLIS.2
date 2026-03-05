import argparse
import os
import time

import torch

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
    parser.add_argument("--resolution", type=str, default="1024", choices=["512", "512g_1024t", "1024", "1024_cascade", "1536", "2048"], help="Generation resolution")
    parser.add_argument("--no_texture_gen", action="store_true", help="Skip texture generation")
    parser.add_argument("--no_pbr", action="store_true", help="Does not attach the PBR textures to the final GLB")
    parser.add_argument("--webp", action="store_true", help="Use WEBP for texture compression in GLB")
    parser.add_argument("--max_num_tokens", type=int, default=49152, help="Max number of tokens")

    parser.add_argument("--low_vram", action="store_true", help="Enable low VRAM mode")

    # Stage 1: Sparse Structure
    parser.add_argument("--ss_guidance_strength", type=float, default=6.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.2)
    parser.add_argument("--ss_sampling_steps", type=int, default=25)
    parser.add_argument("--ss_rescale_t", type=float, default=4.0)

    # Stage 2: Shape Generation
    parser.add_argument("--shape_slat_guidance_strength", type=float, default=6.5)
    parser.add_argument("--shape_slat_guidance_rescale", type=float, default=0.2)
    parser.add_argument("--shape_slat_sampling_steps", type=int, default=25)
    parser.add_argument("--shape_slat_rescale_t", type=float, default=4.0)

    # Stage 3: Texture Generation
    parser.add_argument("--tex_slat_guidance_strength", type=float, default=3.0)
    parser.add_argument("--tex_slat_guidance_rescale", type=float, default=0.2)
    parser.add_argument("--tex_slat_sampling_steps", type=int, default=25)
    parser.add_argument("--tex_slat_rescale_t", type=float, default=3.0)

    # Export Settings
    parser.add_argument("--decimation_target", type=int, default=2000000, help="Target face count for decimation")
    parser.add_argument("--texture_size", type=int, default=4096, choices=[1024, 2048, 4096], help="Texture size")
    parser.add_argument("--remesh_method", type=str, default="dual_contouring", choices=["dual_contouring", "dual_contouring_vb", "faithful_contouring", "none"], help="Remesh method")
    parser.add_argument("--simplify_method", type=str, default="cumesh", choices=["cumesh", "meshlib"], help="Simplify method")
    parser.add_argument("--repair_method", type=str, default="meshlib", choices=["cumesh", "meshlib"], help="Repair method (hole filling)")
    parser.add_argument("--fill_holes_max_perimeter", type=float, default=0.03, help="Max hole perimeter for hole filling")
    parser.add_argument("--fill_holes_unlimited", action="store_true", default=True, help="Fill ALL holes with no perimeter limit (meshlib)")
    parser.add_argument("--no_fill_holes_unlimited", action="store_false", dest="fill_holes_unlimited", help="Disable unlimited hole filling")
    parser.add_argument("--remove_floaters", action="store_true", default=True, help="Remove small disconnected components")
    parser.add_argument("--no_remove_floaters", action="store_false", dest="remove_floaters", help="Disable floater removal")
    parser.add_argument("--smooth_normals", action="store_true", default=True, help="Apply smooth vertex normals")
    parser.add_argument("--no_smooth_normals", action="store_false", dest="smooth_normals", help="Disable smooth normals")
    parser.add_argument("--remesh_quad", action="store_true", help="Use quad-based dual contouring (requires CuMesh quad support)")
    parser.add_argument("--no_prune_invisible_faces", action="store_true", help="Disable pruning of invisible faces")
    parser.add_argument("--single_sided", action="store_true", help="Disable double-sided rendering")
    parser.add_argument("--merge_vertices_dist", type=float, default=0.1, help="Distance threshold for vertex merging")
    parser.add_argument("--shade_smooth", action="store_true", default=True, help="Enable smooth shading")
    parser.add_argument("--no_shade_smooth", action="store_false", dest="shade_smooth", help="Disable smooth shading")
    parser.add_argument("--shade_smooth_angle", type=float, default=0.0, help="Angle threshold for smooth shading (degrees)")

    # Refinement & Retexturing Settings
    parser.add_argument("--high_quality", action="store_true", help="Enable High Quality iterative mode (Refine Mesh + Baked Texture)")
    parser.add_argument("--refine", action="store_true", help="(Deprecated) Enable mesh refinement pass (re-encode and re-sample shape)")
    parser.add_argument("--refine_steps", type=int, default=25, help="Shape sampling steps for refinement")
    parser.add_argument("--refine_guidance", type=float, default=6.5, help="Shape guidance strength for refinement")
    parser.add_argument("--refine_downsampling", type=int, default=16, help="Downsampling factor for refinement")
    parser.add_argument("--retexture", action="store_true", help="(Deprecated) Enable separate texturing pass on cleaned mesh")
    parser.add_argument("--retexture_resolution", type=int, default=1536, choices=[512, 1024, 1536], help="Render resolution for retexturing")
    parser.add_argument("--retexture_steps", type=int, default=25, help="Texture sampling steps for retexturing")
    parser.add_argument("--retexture_guidance", type=float, default=3.0, help="Texture guidance strength for retexturing")

    return parser.parse_args()


def postprocess_mesh(
    mesh,
    res: int,
    decimation_target: int,
    remesh_method: str,
    fill_holes_max_perimeter: float,
    fill_holes_unlimited: bool,
    remove_floaters_enabled: bool,
    smooth_normals: bool,
    remesh_quad: bool,
    repair_method: str,
    simplify_method: str,
    prune_invisible_faces: bool,
    merge_vertices_dist: float,
    shade_smooth: bool,
    shade_smooth_angle: float,
):
    import trimesh
    import numpy as np

    out_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else mesh.vertices,
        faces=mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else mesh.faces,
        process=False
    )
    
    # 1. Fill holes
    if repair_method == "cumesh" or repair_method == "meshlib":
        # we will use the same flags
        if repair_method == "meshlib":
            try:
                import meshlib.mrmeshpy as mrmeshpy
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp:
                    tmp_name = tmp.name
                
                out_mesh.export(tmp_name)
                msh = mrmeshpy.loadMesh(tmp_name)
                e = msh.topology.findHoleRepresentiveEdges()
                if fill_holes_unlimited:
                    mrmeshpy.fillHoles(msh, e)
                else:
                    mrmeshpy.fillHoles(msh, e, mrmeshpy.FillHoleParams(maxPerimeter=fill_holes_max_perimeter))
                mrmeshpy.saveMesh(msh, tmp_name)
                
                out_mesh = trimesh.load(tmp_name, process=False)
                os.remove(tmp_name)
            except ImportError:
                print("Meshlib not installed, skipping.")
        elif repair_method == "cumesh":
            import cumesh
            import torch
            v_t = torch.from_numpy(out_mesh.vertices).cuda().float()
            f_t = torch.from_numpy(out_mesh.faces).cuda().int()
            c_mesh = cumesh.CuMesh()
            c_mesh.init(v_t, f_t)
            c_mesh.fill_holes(max_hole_perimeter=fill_holes_max_perimeter)
            v_t, f_t = c_mesh.read()
            out_mesh = trimesh.Trimesh(vertices=v_t.cpu().numpy(), faces=f_t.cpu().numpy(), process=False)
            
    # Floater removal
    if remove_floaters:
        components = out_mesh.split(only_watertight=False)
        if len(components) > 1:
            total_faces = sum(len(c.faces) for c in components)
            min_faces = total_faces * 0.001
            kept = [c for c in components if len(c.faces) >= min_faces]
            if not kept:
                kept = [max(components, key=lambda c: len(c.faces))]
            if len(kept) == 1:
                out_mesh = kept[0]
            else:
                import trimesh.util
                out_mesh = trimesh.util.concatenate(kept)
            
    # 2. Quad or Dual Contouring Remesh (Assuming standard simplification)
    if simplify_method == "cumesh":
        import cumesh
        import torch
        v_t = torch.from_numpy(out_mesh.vertices).cuda().float()
        f_t = torch.from_numpy(out_mesh.faces).cuda().int()
        c_mesh = cumesh.CuMesh()
        c_mesh.init(v_t, f_t)
        c_mesh.simplify(decimation_target)
        v_t, f_t = c_mesh.read()
        out_mesh = trimesh.Trimesh(vertices=v_t.cpu().numpy(), faces=f_t.cpu().numpy(), process=False)

    return out_mesh

def main():
    args = parse_args()

    # Seed
    seed = np.random.randint(0, np.iinfo(np.int32).max) if args.randomize_seed else args.seed

    t0 = time.time()
    
    # Load Pipeline
    print(f"Loading pipeline (Texture Models: {not args.no_texture_gen})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.low_vram = args.low_vram
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
            "512g_1024t": "512g_1024t",
            "1024": "1024",
            "1024_cascade": "1024_cascade",
            "1536": "1536_cascade",
            "2048": "2048_cascade",
        }[args.resolution],
        return_latent=True,
        max_num_tokens=args.max_num_tokens,
        generate_texture_slat=not (args.no_texture_gen or args.high_quality),
    )
    
    # Convert immutable tuple into a mutable list to allow element reassignment
    latents = list(latents)

    t3 = time.time()
    print(f"Pipeline execution completed in {t3 - t2:.2f} seconds")

    # Extract mesh from latent
    print("Extracting mesh...")
    shape_slat, tex_slat, res = latents
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    attr_layout = pipeline.pbr_attr_layout

    # --- Optional Mesh Refinement Pass ---
    if args.refine or args.high_quality:
        print("\n=== High Quality Iterative Pass ===")
        import trimesh as tm
        
        v_np = mesh.vertices.cpu().numpy()
        f_np = mesh.faces.cpu().numpy()
        initial_mesh = tm.Trimesh(vertices=v_np, faces=f_np, process=False)
        
        # Repair and Simplify base mesh
        cleaned_mesh = postprocess_mesh(
            mesh=initial_mesh,
            res=latents[2],
            decimation_target=args.decimation_target,
            remesh_method=args.remesh_method,
            fill_holes_max_perimeter=args.fill_holes_max_perimeter,
            fill_holes_unlimited=args.fill_holes_unlimited,
            remove_floaters_enabled=args.remove_floaters,
            smooth_normals=args.smooth_normals,
            remesh_quad=args.remesh_quad,
            repair_method=args.repair_method,
            simplify_method=args.simplify_method,
            prune_invisible_faces=not args.no_prune_invisible_faces,
            merge_vertices_dist=args.merge_vertices_dist,
            shade_smooth=args.shade_smooth,
            shade_smooth_angle=args.shade_smooth_angle
        )
        
        print("\n=== Refinement Pass ===")
        refine_shape_params = {
            "steps": args.refine_steps,
            "guidance_strength": args.refine_guidance,
            "guidance_rescale": 0.2,
            "rescale_t": 4.0,
        }
        
        # refine_mesh expects a Y-up mesh because its internal preprocess_mesh converts Y-up to Z-up.
        # However, the mesh from run() is already Z-up. We must convert it to Y-up here.
        y_up_vertices = cleaned_mesh.vertices.copy()
        tmp = y_up_vertices[:, 1].copy()
        y_up_vertices[:, 1] = y_up_vertices[:, 2]
        y_up_vertices[:, 2] = -tmp
        cleaned_mesh.vertices = y_up_vertices

        # refine_mesh returns (out_mesh_list, (shape_slat, tex_slat, res)) when return_latent=True
        refined_outputs, refined_latents = pipeline.refine_mesh(
            mesh=cleaned_mesh,
            image=images,
            seed=seed,
            shape_slat_sampler_params=refine_shape_params,
            tex_slat_sampler_params={},
            resolution=int(args.resolution.replace('_cascade', '').replace('g_1024t', '')),
            max_num_tokens=999999,
            generate_texture_slat=not args.no_texture_gen,  # Generate texture latent for GLB export
            return_latent=True,
            downsampling=args.refine_downsampling,
        )
        latents[0] = refined_latents[0]  # shape_slat (SparseTensor)
        if refined_latents[1] is not None:
            latents[1] = refined_latents[1]  # tex_slat (SparseTensor)
        latents[2] = refined_latents[2]  # res (int)
        t3b = time.time()
        print(f"Refinement completed in {t3b - t3:.2f} seconds")
        t3 = t3b
        
        print("\n=== Post-Process Refined Mesh ===")
        # refined_outputs already contains decoded MeshWithVoxel objects from refine_mesh
        refined_mesh_rep = refined_outputs[0]
        refined_trimesh = tm.Trimesh(
            vertices=refined_mesh_rep.vertices.cpu().numpy(),
            faces=refined_mesh_rep.faces.cpu().numpy(),
            process=False
        )
        
        final_clean_mesh = postprocess_mesh(
            mesh=refined_trimesh,
            res=latents[2],
            decimation_target=args.decimation_target,
            remesh_method=args.remesh_method,
            fill_holes_max_perimeter=args.fill_holes_max_perimeter,
            fill_holes_unlimited=args.fill_holes_unlimited,
            remove_floaters_enabled=args.remove_floaters,
            smooth_normals=args.smooth_normals,
            remesh_quad=args.remesh_quad,
            repair_method=args.repair_method,
            simplify_method=args.simplify_method,
            prune_invisible_faces=not args.no_prune_invisible_faces,
            merge_vertices_dist=args.merge_vertices_dist,
            shade_smooth=args.shade_smooth,
            shade_smooth_angle=args.shade_smooth_angle
        )

        # For CLI High Quality mode, use the refined mesh directly.
        mesh = refined_outputs[0]

    del pipeline
    torch.cuda.empty_cache()
    
    # Prune config
    print("Extracting GLB...")
    
    # If no texture latent was generated (e.g. HQ mode), skip texture extraction
    texture_extraction = not args.no_texture_gen
    if mesh.attrs is None:
        texture_extraction = False
    
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        fill_holes_max_perimeter=args.fill_holes_max_perimeter,
        fill_holes_unlimited=args.fill_holes_unlimited,
        remove_floaters_enabled=args.remove_floaters,
        smooth_normals=args.smooth_normals,
        repair_method=args.repair_method,
        simplify_method=args.simplify_method,
        texture_extraction=texture_extraction,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        remesh_method=args.remesh_method,
        remesh_quad=args.remesh_quad,
        prune_invisible=not args.no_prune_invisible_faces,
        force_double_sided=not args.single_sided,
        merge_vertices_dist=args.merge_vertices_dist,
        shade_smooth=args.shade_smooth,
        shade_smooth_angle=args.shade_smooth_angle,
        use_tqdm=True,
        no_pbr=args.no_pbr,
    )

    t4 = time.time()
    print(f"GLB extracted in {t4 - t3:.2f} seconds")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    glb.export(args.output, extension_webp=args.webp)
    print(f"Saved GLB to {args.output}")

if __name__ == "__main__":
    main()
