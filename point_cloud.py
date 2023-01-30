
import torch
import numpy as np


# fx = 756.8
# fy = 756.0
# centerX = 492.8
# centerY = 270.4

# Scannet
#fx = 1170.187988
#fy = 1170.187988
#centerX = 647.750000
#centerY = 483.750000

# NYU Dataset Parameters
fx = 518.8
fy = 519.4
centerX = 325.5
centerY = 253.7
scalingFactor = 1

# Accepts torch tensor
def generate_point_cloud(depth):
    """
        Generate a Torch point cloud in  from Torch depth image.

        Input:
        depth_file -- filename of depth image

    Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """


    rows, cols = depth.shape
    c, r = torch.arange(cols).unsqueeze(0).cuda().float(), torch.arange(rows).unsqueeze(1).cuda().float()
    z = depth
    x = z * (c - centerX) / fx
    y = z * (r - centerY) / fy
    #x = z * (c) / fx
    #y = z * (r) / fy
    return torch.stack([x, y, z], dim=2)



def generate_point_cloud_np(depth):
    """
    Generate a colored point cloud in numpy format a depth image.

    Input:
    depth_file -- filename of depth image

    """

    depth = np.fliplr(depth)

    points = []
    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            Z = depth[u, v] / scalingFactor
            X = (u - centerX) * Z / fx
            Y = (v - centerY) * Z / fy
            points.append([X, Y, Z])

    return points



def generate_write_pointcloud(rgb, depth, ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    rgb = np.fliplr(rgb)
    depth = np.fliplr(depth)

    #K = np.matrix([[fx, 0, centerX], [0, fy, centerY], [0, 0, 1]])
    points = []
    for v in range(rgb.shape[1]):
        for u in range(rgb.shape[0]):
            color = rgb[u, v]
            Z = depth[u, v]/scalingFactor
            X = (u - centerX) * Z / fx
            Y = (v - centerY) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

def generate_write_pointcloud_masked(rgb, depth, ply_file, mask):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    rgb = np.fliplr(rgb)
    depth = np.fliplr(depth)
    mask = np.fliplr(mask)

    #K = np.matrix([[fx, 0, centerX], [0, fy, centerY], [0, 0, 1]])
    points = []
    for v in range(rgb.shape[1]):
        for u in range(rgb.shape[0]):
            if mask[u][v] == True:
                color = rgb[u, v]
                Z = depth[u, v]/scalingFactor
                X = (u - centerX) * Z / fx
                Y = (v - centerY) * Z / fy
                points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

def write_pointclouds(pcl, ply_file):

    points = []
    color = [255, 255, 255]
    for p in pcl:
        X, Y, Z = p[0], p[1], p[2]
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))

    file = open(ply_file, "w")
    file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        ''' % (len(points), "".join(points)))
    file.close()