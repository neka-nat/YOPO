import torch
import numpy as np

def transform_3d_coordinates(coordinates, RT):
    """
    Transform 3D coordinates using a 3x4 or 4x4 transformation matrix.
    
    Input: 
        coordinates: [N, 3] - numpy array or torch tensor
        RT: [3, 4] or [4, 4] - numpy array or torch tensor  
    Return 
        new_coordinates: [N, 3] - same type as input

    """
    is_torch = torch.is_tensor(coordinates)

    if RT.shape[0] == 3:
        if is_torch:
            RT = torch.cat([RT, torch.tensor([[0, 0, 0, 1]], dtype=RT.dtype, device=RT.device)], dim=0)
        else:
            RT = np.vstack([RT, np.array([0, 0, 0, 1], dtype=RT.dtype)])
    
    if is_torch:
        assert coordinates.shape[1] == 3
        device = coordinates.device
        dtype = coordinates.dtype
        ones = torch.ones((coordinates.shape[0], 1), dtype=dtype, device=device)
        coordinates_homo = torch.cat([coordinates, ones], dim=1)
        new_coordinates = torch.matmul(coordinates_homo, RT.T)
        new_coordinates = new_coordinates[:, :3] / new_coordinates[:, -1:]
    else:
        assert coordinates.shape[1] == 3
        coordinates_homo = np.hstack([coordinates, np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype)])
        new_coordinates = coordinates_homo @ RT.T
        new_coordinates = new_coordinates[:, :3] / new_coordinates[:, -1:]
    
    return new_coordinates

def project_3d_to_2d(coordinates_3d, intrinsics):
    """
    Project 3D coordinates to 2D using camera intrinsics.
    
    Input: 
        coordinates_3d: [N, 3] - numpy array or torch tensor
        intrinsics: [3, 3] - numpy array or torch tensor
    Return 
        projected_coordinates: [N, 2] - same type as input (int32 for numpy, long for torch)
    """
    is_torch = torch.is_tensor(coordinates_3d)
    
    if is_torch:
        projected_coordinates = torch.matmul(coordinates_3d, intrinsics.T)
        projected_coordinates = projected_coordinates[:, :2] / projected_coordinates[:, 2:3]
        projected_coordinates = projected_coordinates.long()
    else:
        projected_coordinates = coordinates_3d @ intrinsics.T
        projected_coordinates = projected_coordinates[:, :2] / projected_coordinates[:, 2:3]
        projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def normalize_vector(v, dim=1, return_mag =False):
    v_mag = torch.sqrt(v.pow(2).sum(dim=dim, keepdim=True))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.expand_as(v)
    v = v/v_mag
    return v

def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)),1)#batch*3
    return out

def rep6d2rotmat(rep_6d):
    """Convert 6D representation to rotation matrix.
    Args:
        rep_6d (torch.Tensor): 6D representation of shape (B, 6).

    Returns:
        rotmat (torch.Tensor): Rotation matrix of shape (B, 3, 3).
    """
    assert rep_6d.dim() == 2 and rep_6d.size(1) == 6, \
        "Input must be a batch of 6D representations with shape (B, 6)"
    
    # Split the 6D representation into two parts
    x_raw = rep_6d[:, :3]
    y_raw = rep_6d[:, 3:]
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)#batch*3
    x = cross_product(y,z)#batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def rotmat2rep6d(rotmat):
    """Convert rotation matrix to 6D representation.
    Args:
        rotmat (torch.Tensor): Rotation matrix of shape (B, 3, 3).

    Returns:
        rep_6d (torch.Tensor): 6D representation of shape (B, 6).
    """
    assert rotmat.dim() == 3 and rotmat.size(1) == 3 and rotmat.size(2) == 3, \
        "Input must be a batch of rotation matrices with shape (B, 3, 3)"
    x = rotmat[:, :, 0]
    y = rotmat[:, :, 1]
    rep_6d = torch.cat((x, y), dim=1) # batch*6
    return rep_6d