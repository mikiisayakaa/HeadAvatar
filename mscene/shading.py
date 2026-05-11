
#from typing import NamedTuple
import torch.nn as nn
import torch
import _gsface_shader

class _bruteforce_diffuse_shader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, albedo, envmap, light2obj_rotmat):
        assert(light2obj_rotmat.requires_grad == False)

        normal = normal.contiguous()
        albedo = albedo.contiguous()
        envmap = envmap.contiguous()
        light2obj_rotmat = light2obj_rotmat.contiguous()

        shading = _gsface_shader.bruteforce_diffuse_shader_forward(
            normal, albedo, envmap, light2obj_rotmat
        )
        ctx.save_for_backward(normal, albedo, envmap, light2obj_rotmat)
        return shading

    # TODO: backward is not well tested !!
    @staticmethod
    def backward(ctx, grad_shading):

        grad_shading = grad_shading.contiguous()

        normal, albedo, envmap, light2obj_rotmat = ctx.saved_tensors

        grad_normal, grad_albedo, grad_envmap = _gsface_shader.bruteforce_diffuse_shader_backward(
            grad_shading, normal, albedo, envmap, light2obj_rotmat
        )
        return grad_normal, grad_albedo, grad_envmap, None

bruteforce_diffuse_shader = _bruteforce_diffuse_shader.apply

#-----------------------------------------------------------

class _bruteforce_specular_shader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
                enable_diffuse, enable_specular):
        assert(light2obj_rotmat.requires_grad == False)

        normal = normal.contiguous()
        view_dir = view_dir.contiguous()
        albedo = albedo.contiguous()
        specular_albedo = specular_albedo.contiguous()
        roughness = roughness.contiguous()
        envmap = envmap.contiguous()
        light2obj_rotmat = light2obj_rotmat.contiguous()

        shading = _gsface_shader.bruteforce_specular_shader_forward(
            normal, view_dir, albedo, specular_albedo, roughness,
            envmap, light2obj_rotmat, enable_diffuse, enable_specular
        )
        ctx.save_for_backward(normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat)
        ctx.dims = [enable_diffuse, enable_specular]
        return shading

    @staticmethod
    def backward(ctx, grad_shading):

        grad_shading = grad_shading.contiguous()

        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat = ctx.saved_tensors
        enable_diffuse, enable_specular = ctx.dims

        grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap = _gsface_shader.bruteforce_specular_shader_backward(
            grad_shading, normal, view_dir, albedo, specular_albedo,
            roughness, envmap, light2obj_rotmat, enable_diffuse, enable_specular
        )

        return grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, None, None, None

def bruteforce_specular_shader(
    normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
    enable_diffuse = True, enable_specular = True
):
    return _bruteforce_specular_shader.apply(
        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
        enable_diffuse, enable_specular
    )

#-----------------------------------------------------------

class _bruteforce_specular_shader_clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
                enable_diffuse, enable_specular):
        assert(light2obj_rotmat.requires_grad == False)

        normal = normal.contiguous()
        view_dir = view_dir.contiguous()
        albedo = albedo.contiguous()
        specular_albedo = specular_albedo.contiguous()
        roughness = roughness.contiguous()
        envmap = envmap.contiguous()
        light2obj_rotmat = light2obj_rotmat.contiguous()

        shading = _gsface_shader.bruteforce_specular_shader_clamp_forward(
            normal, view_dir, albedo, specular_albedo, roughness,
            envmap, light2obj_rotmat, enable_diffuse, enable_specular
        )
        ctx.save_for_backward(normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat)
        ctx.dims = [enable_diffuse, enable_specular]
        return shading

    @staticmethod
    def backward(ctx, grad_shading):

        grad_shading = grad_shading.contiguous()

        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat = ctx.saved_tensors
        enable_diffuse, enable_specular = ctx.dims

        grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap = _gsface_shader.bruteforce_specular_shader_clamp_backward(
            grad_shading, normal, view_dir, albedo, specular_albedo,
            roughness, envmap, light2obj_rotmat, enable_diffuse, enable_specular
        )

        return grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, None, None, None

def bruteforce_specular_shader_clamp(
    normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
    enable_diffuse = True, enable_specular = True
):
    return _bruteforce_specular_shader_clamp.apply(
        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
        enable_diffuse, enable_specular
    )

#-----------------------------------------------------------

class _brutefoce_specular_shader2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
                enable_specular):
        assert(light2obj_rotmat.requires_grad == False)

        normal = normal.contiguous()
        view_dir = view_dir.contiguous()
        albedo = albedo.contiguous()
        specular_albedo = specular_albedo.contiguous()
        roughness = roughness.contiguous()
        envmap = envmap.contiguous()
        light2obj_rotmat = light2obj_rotmat.contiguous()

        shading, diffuse_shading = _gsface_shader.bruteforce_specular_shader_forward2(
            normal, view_dir, albedo, specular_albedo, roughness,
            envmap, light2obj_rotmat, enable_specular
        )
        ctx.save_for_backward(normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat)
        ctx.dims = [enable_specular]
        return shading, diffuse_shading

    @staticmethod
    def backward(ctx, grad_shading, grad_diffuse_shading):

        grad_shading = grad_shading.contiguous()
        grad_diffuse_shading = grad_diffuse_shading.contiguous()

        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat = ctx.saved_tensors
        enable_specular = ctx.dims[0]

        grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap = _gsface_shader.bruteforce_specular_shader_backward2(
            grad_shading, grad_diffuse_shading, normal, view_dir, albedo, specular_albedo,
            roughness, envmap, light2obj_rotmat, enable_specular
        )

        return grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, None, None, None

def bruteforce_specular_shader2(
    normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, enable_specular = True
):
    return _brutefoce_specular_shader2.apply(
        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, enable_specular
    )
    
class _bruteforce_specular_shader2_batch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, mask,
                enable_specular):
        assert(light2obj_rotmat.requires_grad == False)

        normal = normal.contiguous()
        view_dir = view_dir.contiguous()
        albedo = albedo.contiguous()
        specular_albedo = specular_albedo.contiguous()
        roughness = roughness.contiguous()
        envmap = envmap.contiguous()
        mask = mask.contiguous()
        light2obj_rotmat = light2obj_rotmat.contiguous()

        shading, diffuse_shading = _gsface_shader.bruteforce_specular_shader_forward2_batch(
            normal, view_dir, albedo, specular_albedo, roughness,
            envmap, light2obj_rotmat, mask, enable_specular
        )
        ctx.save_for_backward(normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, mask)
        ctx.dims = [enable_specular]
        return shading, diffuse_shading

    @staticmethod
    def backward(ctx, grad_shading, grad_diffuse_shading):

        grad_shading = grad_shading.contiguous()
        grad_diffuse_shading = grad_diffuse_shading.contiguous()

        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, mask = ctx.saved_tensors
        enable_specular = ctx.dims[0]

        grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap = _gsface_shader.bruteforce_specular_shader_backward2_batch(
            grad_shading, grad_diffuse_shading, normal, view_dir, albedo, specular_albedo,
            roughness, envmap, light2obj_rotmat, mask, enable_specular
        )

        return grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, None, None, None

def bruteforce_specular_shader2_batch(
    normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, mask, enable_specular = True
):
    return _bruteforce_specular_shader2_batch.apply(
        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, mask, enable_specular
    )

#-----------------------------------------------------------

class _bruteforce_specularvisibility_shader2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
                face_vertex_list, nearest_triangle_id, barycentric_coord, visibility, enable_specular):
        assert(light2obj_rotmat.requires_grad == False)

        normal = normal.contiguous()
        view_dir = view_dir.contiguous()
        albedo = albedo.contiguous()
        specular_albedo = specular_albedo.contiguous()
        roughness = roughness.contiguous()
        envmap = envmap.contiguous()
        light2obj_rotmat = light2obj_rotmat.contiguous()

        shading, diffuse_shading = _gsface_shader.bruteforce_specularvisibility_shader_forward2(
            normal, view_dir, albedo, specular_albedo, roughness,
            envmap, light2obj_rotmat, face_vertex_list, nearest_triangle_id,
            barycentric_coord, visibility, enable_specular
        )
        bak = [
            normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
            face_vertex_list, nearest_triangle_id, barycentric_coord, visibility
        ]
        ctx.save_for_backward(*bak)
        ctx.dims = [enable_specular]
        return shading, diffuse_shading

    @staticmethod
    def backward(ctx, grad_shading, grad_diffuse_shading):

        grad_shading = grad_shading.contiguous()
        grad_diffuse_shading = grad_diffuse_shading.contiguous()

        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat, \
        face_vertex_list, nearest_triangle_id, barycentric_coord, visibility = ctx.saved_tensors
        enable_specular = ctx.dims[0]

        grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap = _gsface_shader.bruteforce_specularvisibility_shader_backward2(
            grad_shading, grad_diffuse_shading, normal, view_dir, albedo, specular_albedo,
            roughness, envmap, light2obj_rotmat, face_vertex_list, nearest_triangle_id,
            barycentric_coord, visibility, enable_specular
        )
        return grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, None, None, None, None, None, None

        # grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, grad_barycentric_coord = _gsface_shader.bruteforce_specularvisibility_shader_backward2(
        #     grad_shading, grad_diffuse_shading, normal, view_dir, albedo, specular_albedo,
        #     roughness, envmap, light2obj_rotmat, face_vertex_list, nearest_triangle_id,
        #     barycentric_coord, visibility, enable_specular
        # )
        # return grad_normal, grad_view_dir, grad_albedo, grad_specular_albedo, grad_roughness, grad_envmap, None, None, None, grad_barycentric_coord, None, None

def bruteforce_specularvisibility_shader2(
    normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
    face_vertex_list, nearest_triangle_id, barycentric_coord, visibility, enable_specular = True
):
    return _bruteforce_specularvisibility_shader2.apply(
        normal, view_dir, albedo, specular_albedo, roughness, envmap, light2obj_rotmat,
        face_vertex_list, nearest_triangle_id, barycentric_coord, visibility, enable_specular
    )

#-----------------------------------------------------------
class _get_uv_of_triangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, query_pos):
        ctx.save_for_backward(f, query_pos)
        coord = _gsface_shader.get_uv_of_triangle_forward(
            f, query_pos
        )
        return coord
    @staticmethod
    def backward(ctx, grad_coord):
        f, query_pos = ctx.saved_tensors
        grad_query_pos = _gsface_shader.get_uv_of_triangle_backward(
            f, query_pos, grad_coord
        )
        return None, grad_query_pos

def get_uv_of_triangle(tri, query_pos):
    return _get_uv_of_triangle.apply(tri, query_pos)

#-----------------------------------------------------------
class _get_nearest_mesh_points(torch.autograd.Function):
    @staticmethod
    def forward(ctx, adjacency_head, adjacency_list, face_vertex_list,
        vertex_pos, query_pos, idxs):

        triangle_idx, triangle_uv = _gsface_shader.get_nearest_mesh_points_forward(
            adjacency_head, adjacency_list, face_vertex_list,
            vertex_pos, query_pos, idxs
        )

        ctx.save_for_backward(
            face_vertex_list, vertex_pos, query_pos, triangle_idx
        )
        return triangle_idx, triangle_uv

    @staticmethod
    def backward(ctx, _, grad_uv):
        face_vertex_list, vertex_pos, query_pos, nearest_triangle_id = ctx.saved_tensors
        grad_pos = _gsface_shader.get_nearest_mesh_points_backward(
            face_vertex_list, vertex_pos, query_pos, nearest_triangle_id, grad_uv
        )
        return None, None, None, None, grad_pos, None

def get_nearest_mesh_points_forward(
    adjacency_head, adjacency_list, face_vertex_list,
    vertex_pos, query_pos, idxs
):
    # adjacency_head 2 x N_vertex
    # adjacency_list N_tbd
    # face_vertex_list 3 x N_face
    # vertex_pos 3 x N_vertex
    # query_pos 3 x P
    # idxs P x K
    return _get_nearest_mesh_points.apply(
        adjacency_head, adjacency_list, face_vertex_list,
        vertex_pos, query_pos, idxs
    )

#-----------------------------------------------------------

def generate_camera_ray_forward(proj,c2w=None,H=512,W=512,flipY=False,normalize=False,z_sign=1.):
    # proj 4x4, row major
    # [optional] c2w 4x4, row major
    # return 3xHxW
    return _gsface_shader.generate_camera_ray_forward(
        proj, c2w, H, W, flipY, normalize, z_sign
    )

def sample_envmap_forward(query_point, envmap):
    # query_point 3xP
    # envmap CxHxW
    # [return] CxP
    return _gsface_shader.sample_envmap_forward(query_point, envmap)

def mt_format_conversion_forward(jaw, eyes):
    # jaw N x 6
    # eyes N x 12
    # [outputs] N x 15
    return _gsface_shader.mt_format_conversion_forward(
        jaw, eyes
    )

def extract_bitfield_forward(bitfield, N_bit, i_start, i_end):
    # bitfield = N_slot x N
    # [return] (i_end - i_start) x N_bit
    return _gsface_shader.extract_bitfield_forward(
        bitfield, N_bit, i_start, i_end
    )
