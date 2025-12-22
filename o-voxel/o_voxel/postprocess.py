from typing import *
from tqdm import tqdm
import math
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh





def reduce_face_with_meshlib(mesh: trimesh.Trimesh, max_facenum: int = 100000):
    current_face_count = len(mesh.faces)
    if current_face_count <= max_facenum:
        return mesh

    import meshlib.mrmeshpy as mrmeshpy
    import meshlib.mrmeshnumpy as mrmeshnumpy
    import multiprocessing

    # Load mesh
    mesh_mr = mrmeshnumpy.meshFromFacesVerts(mesh.faces, mesh.vertices)

    faces_to_delete = current_face_count - max_facenum
    #  Setup simplification parameters
    mesh_mr.packOptimally()
    settings = mrmeshpy.DecimateSettings()
    settings.maxDeletedFaces = faces_to_delete
    settings.subdivideParts = multiprocessing.cpu_count()
    # settings.maxError = 0.001
    settings.packMesh = True

    print(f'Decimating mesh... targeting {max_facenum} faces from {current_face_count} faces')
    print(f'Decimating mesh... Deleting {faces_to_delete} faces')
    mrmeshpy.decimateMesh(mesh_mr, settings)
    print(f'Decimation done. Resulting mesh has {mesh_mr.topology.faceSize()} faces')

    out_verts = mrmeshnumpy.getNumpyVerts(mesh_mr)
    out_faces = mrmeshnumpy.getNumpyFaces(mesh_mr.topology)

    return trimesh.Trimesh(out_verts, out_faces)



def get_visible_faces(vertices: torch.Tensor, faces: torch.Tensor, num_views: int = 256, resolution: int = 2048, face_padding: int = 4, verbose: bool = False) -> torch.Tensor:
    """
    Identifies visible faces by rendering the mesh from multiple viewpoints.
    """
    if verbose:
        print(f"Computing visibility for {faces.shape[0]} faces using {num_views} views (res {resolution})...")

    device = vertices.device
    
    # helper: look_at function
    def look_at(eye, center, up):
        z = eye - center
        z = z / torch.norm(z, dim=-1, keepdim=True)
        x = torch.cross(up, z)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.cross(z, x)
        
        # Create 4x4 matrices
        view_mat = torch.eye(4, device=device).unsqueeze(0).repeat(eye.shape[0], 1, 1)
        view_mat[:, 0, 0:3] = x
        view_mat[:, 1, 0:3] = y
        view_mat[:, 2, 0:3] = z
        view_mat[:, 0, 3] = -torch.sum(x * eye, dim=-1)
        view_mat[:, 1, 3] = -torch.sum(y * eye, dim=-1)
        view_mat[:, 2, 3] = -torch.sum(z * eye, dim=-1)
        
        return view_mat

    # helper: perspective projection
    def perspective(fovy, aspect, near, far):
        f = 1.0 / math.tan(fovy / 2.0)
        proj_mat = torch.zeros(4, 4, device=device)
        proj_mat[0, 0] = f / aspect
        proj_mat[1, 1] = f
        proj_mat[2, 2] = (far + near) / (near - far)
        proj_mat[2, 3] = (2.0 * far * near) / (near - far)
        proj_mat[3, 2] = -1.0
        return proj_mat

    # 1. Normalize mesh to unit sphere at origin for camera placement
    v_min = vertices.min(dim=0)[0]
    v_max = vertices.max(dim=0)[0]
    center = (v_min + v_max) / 2
    scale = (v_max - v_min).max()
    
    # 2. Generate cameras using Fibonacci sphere
    indices = torch.arange(0, num_views, dtype=torch.float32, device=device)
    phi = torch.acos(1 - 2 * (indices + 0.5) / num_views)
    theta = math.pi * (1 + 5**0.5) * indices
    
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    
    # Camera positions (radius 2.0 to ensure full view)
    cam_pos = torch.stack([x, y, z], dim=1) * 2.5
    
    # Up vector (standard Y-up)
    up = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_views, 1)
    
    # View matrices
    # We look at the center of the mesh (which we normalized to origin conceptually, 
    # but let's just transform vertices centered)
    centered_verts = (vertices - center) / scale
    
    view_mats = look_at(cam_pos, torch.zeros_like(cam_pos), up)
    proj_mat = perspective(math.radians(60), 1.0, 0.1, 10.0).unsqueeze(0).repeat(num_views, 1, 1)
    mvp_mats = torch.bmm(proj_mat, view_mats)
    
    # 3. Rasterize
    ctx = dr.RasterizeCudaContext()
    
    # Prepare vertices: (N, 4) homography
    verts_hom = torch.cat([centered_verts, torch.ones_like(centered_verts[:, :1])], dim=1)
    verts_hom = verts_hom.unsqueeze(0).repeat(num_views, 1, 1) # (V, N, 4)
    
    # Transform vertices for all views
    # (V, N, 4) x (V, 4, 4) -> (V, N, 4) ? specific matmul needed
    # bmm: (B, N, M) x (B, M, P) -> (B, N, P)
    # We want (V, 4, 4) x (V, 4, N)' -> result transposed?
    # Easier: (V, N, 4) @ (V, 4, 4).T
    pos_clip = torch.bmm(verts_hom, mvp_mats.transpose(1, 2))
    
    # Rasterize
    # This might fail OOM if too many views/faces, but we render face indices
    # We'll batch views if needed, but 64 views 512x512 is small enough usually.
    # We render face ID + 1 to distinguish from background 0
    
    # Prepare faces for int32
    faces_int = faces.int()
    
    visible_faces_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=device)
    
    # Process in batches of views to be safe
    batch_size = 16
    for i in range(0, num_views, batch_size):
        end = min(i + batch_size, num_views)
        batch_pos = pos_clip[i:end]
        
        rast, _ = dr.rasterize(ctx, batch_pos, faces_int, resolution=[resolution, resolution])
        
        # rast shape: (B, H, W, 4). The last channel is barycentrics + triangle_id
        # nvdiffrast: The last channel contains triangle_id + 1.
        tri_ids = rast[..., 3].int()
        
        # Unique visible IDs in this batch
        unique_ids = torch.unique(tri_ids)
        
        # Filter background (0)
        valid_ids = unique_ids[unique_ids > 0] - 1
        
        visible_faces_mask[valid_ids.long()] = True
        
    if verbose:
        print(f"  Initial Pass: Found {visible_faces_mask.sum()} visible faces")

    # 4. Face Padding (Dilation)
    if face_padding > 0:
        if verbose:
            print(f"  Dilating mask by {face_padding} iterations...")
        
        # Convert to trimesh for adjacency
        mesh_cpu = trimesh.Trimesh(
            vertices=vertices.cpu().numpy(),
            faces=faces.cpu().numpy(),
            process=False
        )
        
        adj = mesh_cpu.face_adjacency
        adj_t = torch.tensor(adj, device=device, dtype=torch.long)
        
        # Dilate mask
        for _ in range(face_padding):
            # If neighbor is visible, make me visible
            # Gather status of both faces in adjacency pair
            mask_a = visible_faces_mask[adj_t[:, 0]]
            mask_b = visible_faces_mask[adj_t[:, 1]]
            
            # Any pair with at least one visible face makes both visible?
            # Or just propagate existing visibility?
            # "Add neighbors of currently visible faces"
            # visible_faces_mask[adj[:, 0]] |= mask_b
            # visible_faces_mask[adj[:, 1]] |= mask_a
            
            # Efficient update:
            # We want to set mask=True for faces adjacent to a True face.
            # Identify pairs where one is True. Set both to True.
            to_add = mask_a | mask_b
            
            # Scatter updates
            # Note: direct indexing with duplicate indices is nondeterministic/racey in some contexts but for ORing boolean it's usually fine
            # or we can just do it simpler:
            
            visible_faces_mask[adj_t[:, 0]] = to_add
            visible_faces_mask[adj_t[:, 1]] = to_add

    if verbose:
        print(f"  Final Pass: Found {visible_faces_mask.sum()} visible faces out of {faces.shape[0]}")
        
    return visible_faces_mask


def to_glb(
    vertices: torch.Tensor,

    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    simplify_method: str = 'cumesh',
    texture_extraction: bool = True,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    remesh_method: str = 'dual_contouring',
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    prune_invisible: bool = False,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        prune_invisible: whether to prune invisible faces (inner geometry) before texturing
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()
    
    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        if simplify_method == 'cumesh':
            mesh.simplify(decimation_target * 3, verbose=verbose)
        elif simplify_method == 'meshlib':
             # GPU -> CPU -> Meshlib -> CPU -> GPU
            v, f = mesh.read()
            t_mesh = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy())
            t_mesh = reduce_face_with_meshlib(t_mesh, decimation_target * 3)
            mesh.init(torch.from_numpy(t_mesh.vertices).float().cuda(), torch.from_numpy(t_mesh.faces).int().cuda())

        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        if simplify_method == 'cumesh':
            mesh.simplify(decimation_target, verbose=verbose)
        elif simplify_method == 'meshlib':
             # GPU -> CPU -> Meshlib -> CPU -> GPU
            v, f = mesh.read()
            t_mesh = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy())
            t_mesh = reduce_face_with_meshlib(t_mesh, decimation_target)
            mesh.init(torch.from_numpy(t_mesh.vertices).float().cuda(), torch.from_numpy(t_mesh.faces).int().cuda())

        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Remeshing Pipeline ---
    else:
        if remesh_method == 'dual_contouring':
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            resolution = grid_size.max().item()

            # Perform Dual Contouring remeshing (rebuilds topology)
            mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
                vertices, faces,
                center = center,
                scale = (resolution + 3 * remesh_band) / resolution * scale,
                resolution = resolution,
                band = remesh_band,
                project_back = remesh_project,
                verbose = verbose,
                bvh = bvh,
            ))
            if verbose:
                print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        elif remesh_method == 'faithful_contouring':
            try:
                from faithcontour import FCTEncoder, FCTDecoder, normalize_mesh
                from atom3d import MeshBVH
                from atom3d.grid import OctreeIndexer
            except ImportError:
                raise ImportError("Faithful Contouring is not installed. Please install it to use faithful_contouring remeshing. See https://github.com/Luo-Yihao/FaithC")

            V = vertices.detach().contiguous().to(device="cuda", dtype=torch.float32)
            F = faces.detach().contiguous().to(device="cuda", dtype=torch.long)

            max_level = int(math.log2(grid_size.max().item()))
            min_level = min(4, max(1, max_level - 1))

            bvh = MeshBVH(V, F, device='cuda')
            octree = OctreeIndexer(max_level=max_level, bounds=bvh.get_bounds(), device='cuda')

            encoder = FCTEncoder(bvh, octree, device='cuda')
            solver_weights = {
                'lambda_n': 1.0,
                'lambda_d': 1e-3,
                'weight_power': 1
            }
            fct_result = encoder.encode(
                min_level=min_level,
                solver_weights=solver_weights,
                compute_flux=True,
                clamp_anchors=True
            )

            decoder = FCTDecoder(resolution=grid_size.max().item(), bounds=bvh.get_bounds(), device='cuda')
            mesh_result = decoder.decode_from_result(fct_result)

            mesh.init(
                mesh_result.vertices.contiguous(),
                mesh_result.faces.contiguous().to(torch.int32),
            )

        # Simplify and clean the remeshed result (similar logic to above)
        if simplify_method == 'cumesh':
            mesh.simplify(decimation_target, verbose=verbose)
        elif simplify_method == 'meshlib':
             # GPU -> CPU -> Meshlib -> CPU -> GPU
            v, f = mesh.read()
            t_mesh = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy())
            t_mesh = reduce_face_with_meshlib(t_mesh, decimation_target)
            mesh.init(torch.from_numpy(t_mesh.vertices).float().cuda(), torch.from_numpy(t_mesh.faces).int().cuda())

        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    # --- Pruning Invisible Faces (New Step) ---
    if prune_invisible:
        if use_tqdm:
            pbar.set_description("Pruning invisible faces")
        if verbose:
            print("Pruning invisible faces...", end='', flush=True)
            
        v, f = mesh.read()
        visible_mask = get_visible_faces(v, f, verbose=verbose)
        
        # Make sure we don't prune everything if visibility check fails
        if visible_mask.sum() > 0:
            visible_faces = f[visible_mask]
            
            # Re-init mesh with only visible faces
            # Note: This might leave unreferenced vertices, but we can clean them
            mesh.init(v, visible_faces)

            if verbose:
                print(f" -> {mesh.num_faces} faces remaining")

            # Restore to required face count
            mesh.simplify(decimation_target, verbose=verbose)
        else:
            print("Warning: Visibility pruning removed all faces! Skipping pruning to be safe.")

    
    # --- UV Parameterization ---
    if not texture_extraction:
        # Skip UV unwrapping and texture baking
        if use_tqdm:
            pbar.update(3) # Skip remaining steps
            pbar.close()
        
        # Coordinate System Conversion
        vertices_np = mesh.read()[0].cpu().numpy()
        faces_np = mesh.read()[1].cpu().numpy()
        mesh.compute_vertex_normals()
        normals_np = mesh.read_vertex_normals().cpu().numpy()
        
        # Swap Y and Z axes, invert Y
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        
        return trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
        )

    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
        
    # Setup differentiable rasterizer context
    ctx = dr.RasterizeCudaContext()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    
    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    
    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    
    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    
    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    
    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    mask = mask.cpu().numpy()
    
    # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha_mode = 'OPAQUE'

    # Inpainting: fill gaps (dilation) to prevent black seams at UV boundaries
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    
    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")
    
    return textured_mesh