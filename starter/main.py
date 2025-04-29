import argparse
import pickle

import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import torch
import mcubes
import numpy as np

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, unproject_depth_image, get_points_renderer, get_points_renderer_neon

import imageio

#1.1
def render_cow_gif(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    num_views = 36
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 4,
    		elev = 0,
    		azim = 360 * i / num_views
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        image = renderer(mesh, cameras=cameras, lights=lights)
        image = (image.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        images.append(image)
            
    duration = 1000 // 15
    imageio.mimsave('submissions/my_gif.gif', images, duration=duration, loop=0)

#2.1
def render_tetrahedron(image_size=256, color=[0.7, 0.7, 1], device=None):

    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    color1=[0.5, 0, 1]
    color2=[1, 0.5, 0]

    # Get the vertices, faces, and textures.
    vertices = np.array([[-1, 0, 0], [1, 0, 1], [1, 0, -1], [0, 1, 0]])
    vertices = torch.from_numpy(vertices).float()
    y_min = torch.min(vertices[:, 1])
    y_max = torch.max(vertices[:, 1])
    textures = []
    for vert in vertices:
        alpha = (vert[1] - y_min)/(y_max - y_min)
        color = alpha * torch.tensor(color2)  + (1-alpha) * torch.tensor(color1)
        textures.append(color)
    textures = torch.from_numpy(np.array(textures)).float().unsqueeze(0)  # (1, N_v, 3)
    vertices = vertices.unsqueeze(0)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = torch.from_numpy(faces).float().unsqueeze(0)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    num_views = 36
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 4,
    		elev = 30,
    		azim = 360 * i / num_views
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        image = renderer(mesh, cameras=cameras, lights=lights)
        image = (image.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        images.append(image)
            
    duration = 1000 // 15
    imageio.mimsave('submissions/tetrahedron.gif', images, duration=duration, loop=0)

#2.2
def render_cube(image_size=256, color=[0.7, 0.7, 1], device=None):

    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)
    color1=[0.5, 0, 1]
    color2=[1, 0.5, 0]

    # Get the vertices, faces, and textures.
    vertices = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1], [1, 1.43, 0], [0, 1.43, 1], [-1, 1.43, 0], [0, 1.43, -1]])
    vertices = torch.from_numpy(vertices).float()
    y_min = torch.min(vertices[:, 1])
    y_max = torch.max(vertices[:, 1])
    textures = []
    for vert in vertices:
        alpha = (vert[1] - y_min)/(y_max - y_min)
        color = alpha * torch.tensor(color2)  + (1-alpha) * torch.tensor(color1)
        textures.append(color)
    textures = torch.from_numpy(np.array(textures)).float().unsqueeze(0)  # (1, N_v, 3)
    vertices = vertices.unsqueeze(0)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7], [4, 5, 1], [5, 6, 2], [6, 7, 3], [7, 4, 0]])
    faces = torch.from_numpy(faces).float().unsqueeze(0)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    num_views = 36
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 4,
    		elev = 30,
    		azim = 360 * i / num_views
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, -3, 0]], device=device)

        image = renderer(mesh, cameras=cameras, lights=lights)
        image = (image.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        images.append(image)
            
    duration = 1000 // 15
    imageio.mimsave('submissions/cube.gif', images, duration=duration, loop=0)

#3
def render_cow_texture(
    cow_path="data/cow.obj", image_size=256, color1=[0.5, 0, 1], color2=[1, 0.5, 0], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    z_min = torch.min(vertices[:, 2])
    z_max = torch.max(vertices[:, 2])
    textures = []
    for vert in vertices:
        alpha = (vert[2] - z_min)/(z_max - z_min)
        color = alpha * torch.tensor(color2)  + (1-alpha) * torch.tensor(color1)
        textures.append(color)
    textures = torch.from_numpy(np.array(textures)).float().unsqueeze(0)  # (1, N_v, 3)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    num_views = 36
    images = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 4,
    		elev = 0,
    		azim = 360 * i / num_views
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        image = renderer(mesh, cameras=cameras, lights=lights)
        image = (image.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        images.append(image)
            
    duration = 1000 // 15
    imageio.mimsave('submissions/textures.gif', images, duration=duration, loop=0)

#4
def render_cow_transform(
    cow_path="data/cow.obj",
    image_size=256,
    color = [0.7, 0.7, 1],
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()

    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    #Uncomment to see each tranform. All images in this function are saved as "transform.jpg" unless specified otherwise!

    # #1
    # R_relative=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    # T_relative=[0, 0, 0]
    # #2
    # R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # T_relative=[0, 0, 3]
    # #3
    # R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # T_relative=[0.5, -0.5, 0]
    # #4
    # R_relative=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    # T_relative=[-3, 0, 3]

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative

    renderer = get_mesh_renderer(image_size=image_size)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend[0, ..., :3].cpu().numpy()
    plt.imsave('submissions/transform.jpg', rend)

#5.1
def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_plant(
    data_path="data/rgbd_data.pkl",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    data = load_rgbd_data(data_path)
    rgb1, mask1, depth1, rgb2, mask2, depth2, cameras1, cameras2 = data.values()
    rgb1 = torch.tensor(rgb1)
    mask1 = torch.tensor(mask1)
    depth1 = torch.tensor(depth1)
    rgb2 = torch.tensor(rgb2)
    mask2 = torch.tensor(mask2)
    depth2 = torch.tensor(depth2)
    verts1, rgba1 = unproject_depth_image(rgb1, mask1, depth1, cameras1)
    verts2, rgba2 = unproject_depth_image(rgb2, mask2, depth2, cameras2)
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    verts1 = verts1.to(device).unsqueeze(0)
    rgba1 = rgba1.to(device).unsqueeze(0)
    verts2 = verts2.to(device).unsqueeze(0)
    rgba2 = rgba2.to(device).unsqueeze(0)
    combined_verts = torch.cat((verts1, verts2), dim=1)  
    combined_rgba = torch.cat((rgba1, rgba2), dim=1)
    point_cloud1 = pytorch3d.structures.Pointclouds(points=verts1, features=rgba1)
    point_cloud2 = pytorch3d.structures.Pointclouds(points=verts2, features=rgba2)
    combined_cloud = pytorch3d.structures.Pointclouds(points=combined_verts, features=combined_rgba)
    theta = 9 * np.pi/180
    R_correct = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    num_views = 36
    rend = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 7,
    		elev = 0,
    		azim = 360 * i / num_views,
            up = ((0, -1, 0),)
        )
        R = torch.tensor(R_correct).float() @ R

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        image1 = renderer(point_cloud1, cameras=cameras)
        image1 = (image1.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        image2 = renderer(point_cloud2, cameras=cameras)
        image2 = (image2.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        image_combined = renderer(combined_cloud, cameras=cameras)
        image_combined = (image_combined.cpu().numpy()[0, ..., :3] *255).astype(np.uint8) # (B, H, W, 4) -> (H, W, 3)
        side_by_side = np.concatenate((image1, image2, image_combined), axis=1)  
        rend.append(side_by_side)

            
    duration = 1000 // 15
    imageio.mimsave('submissions/plant.gif', rend, duration=duration, loop=0)

#5.2
def render_torus(image_size=256, num_samples=1000, device=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta, indexing='ij')
    R = 2
    r = 0.5

    x = (R + r*torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r*torch.cos(Theta)) * torch.sin(Phi)
    z = r*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    num_views = 36
    rends = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 6,
    		elev = 0,
    		azim = 360 * i / num_views,
            up = ((0, 1, 0),)
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(torus_point_cloud, cameras=cameras)
        rends.append((rend.cpu().numpy()[0, ..., :3]*255).astype(np.uint8))

    duration = 1000 // 15
    imageio.mimsave('submissions/torus.gif', rends, duration=duration, loop=0)

def render_mobius_strip(image_size=256, num_samples=1000, device=None):
    """
    Renders a mobius strip using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    r = torch.linspace(-1.5, 1.5, num_samples)
    alpha = torch.linspace(0, 2 * np.pi, num_samples)
    R, Alpha = torch.meshgrid(r, alpha, indexing='ij')
    
    x = (2 + R/2*torch.cos(Alpha/2)) * torch.cos(Alpha)
    y = R/2*torch.sin(Alpha/2)
    z = (1.5 + R/2*torch.cos(Alpha/2)) * torch.sin(Alpha)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color1=[0.5, 0, 1]
    color2=[1, 0.5, 0]
    y_min = torch.min(points[:, 1])
    y_max = torch.max(points[:, 1])
    textures = []
    for point in points:
        alpha = (point[1] - y_min)/(y_max - y_min)
        color = alpha * torch.tensor(color2)  + (1-alpha) * torch.tensor(color1)
        textures.append(color)
    textures = torch.from_numpy(np.array(textures)).float()  # (1, N_v, 3)

    mStrip_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[textures],
    ).to(device)

    num_views = 36
    rends = []

    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 6,
    		elev = -45,
    		azim = 360 * i / num_views,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(mStrip_point_cloud, cameras=cameras)
        rends.append((rend.cpu().numpy()[0, ..., :3]*255).astype(np.uint8))

    duration = 1000 // 15
    imageio.mimsave('submissions/mobiusStrip.gif', rends, duration=duration, loop=0)

#5.3
def render_torus_implicit(image_size=256, voxel_size=125, device=None):
    if device is None:
        device = get_device()
    R = 1
    r = 1/3
    
    X, Y, Z = torch.meshgrid([torch.linspace(-(R+r+0.5), (R+r+0.5), voxel_size)]*3, indexing = 'ij')
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R) ** 2 + (Z ** 2) - (r ** 2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    vertices = (vertices / voxel_size) * (2*(R+r)) - (R+r)
    
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    num_views = 36
    rends = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 2,
    		elev = 0,
    		azim = 360 * i / num_views,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        renderer = get_mesh_renderer(image_size=image_size, device=device)
        rend = renderer(mesh, cameras=cameras)
        rends.append((rend.cpu().numpy()[0, ..., :3]*255).astype(np.uint8))

    duration = 1000 // 15
    imageio.mimsave('submissions/torus_implicit.gif', rends, duration=duration, loop=0)

def render_barth_implicit(image_size=256, voxel_size=125, device=None):
    if device is None:
        device = get_device()
    
    phi = (1 + np.sqrt(5))/2
    x, y, z = torch.meshgrid([torch.linspace(-1.5, 1.5, voxel_size)]*3, indexing = 'ij')
    term1 = (phi**2 * x**2 - y**2)
    term2 = (phi**2 * y**2 - z**2)
    term3 = (phi**2 * z**2 - x**2)
    term4 = (x**2 + y**2 + z**2 - 1)
    voxels = 4 * term1 * term2 * term3 - (1 + 2 * phi) * term4**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    vertices = (vertices / voxel_size) * (3) - 1.5
    
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(textures.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    num_views = 36
    rends = []
    for i in range(num_views):
        R, T = pytorch3d.renderer.look_at_view_transform(
    		dist = 5,
    		elev = 0,
    		azim = 360 * i / num_views,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        renderer = get_mesh_renderer(image_size=image_size, device=device)
        rend = renderer(mesh, cameras=cameras)
        rends.append((rend.cpu().numpy()[0, ..., :3]*255).astype(np.uint8))

    duration = 1000 // 15
    imageio.mimsave('submissions/barth_implicit.gif', rends, duration=duration, loop=0)

#6
def render_knot_animation(image_size=256, num_samples=1500, device=None):
    """
    Renders a trefoil knot knotting and unknotting using parametric sampling.
    """

    if device is None:
        device = get_device()

    a, c = 1, 0.3
    num_views = 72
    
    s = torch.linspace(0, 2*np.pi, num_samples, device=device)
    u = 3*s
    v = 2*s
    
    rends = []

    for i in range(num_views):
        t = i / (num_views - 1)
        b = 1 - 0.75 * torch.sin(torch.tensor(np.pi * t))
        
        gamma = torch.stack([
            (a + b*torch.cos(u)) * torch.cos(v),
            (a + b*torch.cos(u)) * torch.sin(v),
            b*torch.sin(u)
        ], dim=1)

        du = 3
        dv = 2
        dgamma_ds = torch.stack([
            -b*du*torch.sin(u)*torch.cos(v) - (a + b*torch.cos(u))*dv*torch.sin(v),
            -b*du*torch.sin(u)*torch.sin(v) + (a + b*torch.cos(u))*dv*torch.cos(v),
            b*du*torch.cos(u)
        ], dim=1)
        
        t = dgamma_ds / torch.norm(dgamma_ds, dim=1, keepdim=True)

        n_T = torch.stack([
            torch.cos(2*s)*torch.cos(3*s),
            torch.sin(2*s)*torch.cos(3*s),
            torch.sin(3*s)
        ], dim=1)
        n_T = n_T / torch.norm(n_T, dim=1, keepdim=True)

        b_vec = torch.cross(t, n_T, dim=1)
        b_vec = b_vec / torch.norm(b_vec, dim=1, keepdim=True)

        alpha = torch.linspace(0, 2*np.pi, 50, device=device)  
        s_grid, alpha_grid = torch.meshgrid(s, alpha, indexing='ij')
        
        gamma = gamma.unsqueeze(1)  
        n_T = n_T.unsqueeze(1)  
        b_vec = b_vec.unsqueeze(1)   
        
        points = gamma + c * (torch.cos(alpha_grid)[..., None] * n_T + 
                              torch.sin(alpha_grid)[..., None] * b_vec)
        
        points = points.reshape(-1, 3)

        
        time = torch.linspace(0, 2*np.pi, num_views)
        textures = torch.stack([
            torch.sin(points[:,0] + time[i]) * 0.75,  
            torch.cos(points[:,1] + time[i]) * 0.25,   
            torch.cos(points[:,1] + time[i]) * 0.75  
        ], dim=1)

        trefoil_point_cloud = pytorch3d.structures.Pointclouds(
            points=[points], features=[textures],
        ).to(device)

        R, T = pytorch3d.renderer.look_at_view_transform(
            dist = 6,
            elev =10,
            azim = 360*i / num_views,
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(trefoil_point_cloud, cameras=cameras)
        image = (rend.cpu().numpy()[0, ..., :3]*255).astype(np.uint8)
        rends.append(image)

    duration = 1000 // 15
    imageio.mimsave('submissions/trefoil.gif', rends, duration=duration, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    # Uncomment function call to execute!

    #1.1
    # render_cow_gif(cow_path=args.cow_path, image_size=args.image_size)

    #2.1
    # render_tetrahedron(image_size=args.image_size)

    #2.2
    # render_cube(image_size=args.image_size)

    #3
    # render_cow_texture(cow_path=args.cow_path, image_size=args.image_size)

    #4
    # render_cow_transform(cow_path=args.cow_path, image_size=args.image_size)

    #5.1
    # render_plant(image_size=args.image_size)

    #5.2
    # render_torus(image_size=args.image_size)
    # render_mobius_strip(image_size=args.image_size)

    #5.3
    # render_torus_implicit(image_size=args.image_size)
    # render_barth_implicit(image_size=args.image_size)

    #6
    # render_knot_animation(image_size=args.image_size)
