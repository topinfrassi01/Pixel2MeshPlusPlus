import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
import tensorflow as tf
import trimesh

from modules.chamfer import nn_distance

#pylint: disable=no-name-in-module
from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE

def _convert_np_to_point3D(arr):
    points = []
    for i in range(0, arr.shape[0]):
        p = arr[i,:]
        points.append(POINT3D(p[0], p[1], p[2]))

    return points

def _normalize(points):
    centroid = np.mean(points, axis=1)
    points = points - centroid

    maximum = np.max(np.abs(points), axis=1)

    return points / maximum, maximum

def _icp_rigid_register(source, dest):
    ''' Taken from : https://github.com/aalavandhaann/go-icp_cython
    '''

    source = _convert_np_to_point3D(source)
    dest = _convert_np_to_point3D(dest)

    goicp = GoICP()
    rNode = ROTNODE()
    tNode = TRANSNODE()
    
    rNode.a = -3.1416
    rNode.b = -3.1416
    rNode.c = -3.1416
    rNode.w = 6.2832
    
    tNode.x = -0.5
    tNode.y = -0.5
    tNode.z = -0.5
    tNode.w = 1.0

    goicp.MSEThresh = 0.001
    goicp.trimFraction = 0.0
    
    if(goicp.trimFraction < 0.001):
        goicp.doTrim = False

    goicp.loadModelAndData(len(source), source, len(dest), dest)
    #LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
    goicp.setDTSizeAndFactor(300, 2.0)
    goicp.setInitNodeRot(rNode)
    goicp.setInitNodeTrans(tNode)
    goicp.BuildDT()
    goicp.Register()

    optR = np.array(goicp.optimalRotation())
    optT = goicp.optimalTranslation()
    optT.append(1.0)
    optT = np.array(optT)

    transform = np.zeros((4,4))
    transform[:3, :3] = optR
    transform[:,3] = optT

    homogeneous_dest = np.hstack(dest, np.ones((len(dest), 1)))

    transform_model_points = np.matmul(transform, homogeneous_dest.transpose()).transpose()[:,:3]

    return transform_model_points


def _register_without_scale(source, dest):
    source_normalized, _ = _normalize(source)
    dest_normalized, dest_scale = _normalize(dest)

    dest_registered = _icp_rigid_register(source_normalized, dest_normalized)
    
    dest_registered = dest_registered * dest_scale

    return dest_registered

def chamfer_distance(source, dest, session):
    xyz1 = tf.placeholder(tf.float32, shape=(None, 3))
    xyz2 = tf.placeholder(tf.float32, shape=(None, 3))
    dist1, idx1, dist2, idx2 = nn_distance(xyz1, xyz2)

    d1, _i1, d2, _i2 = session.run([dist1, idx1, dist2, idx2], feed_dict={xyz1: source, xyz2: dest})

    return np.mean(d1) + np.mean(d2)


def dice(source, dest, thresholds, session):
    len_points = source.shape[0]
    len_labels = dest.shape[0]
    f_scores = []

    xyz1 = tf.placeholder(tf.float32, shape=(None, 3))
    xyz2 = tf.placeholder(tf.float32, shape=(None, 3))
    dist1, idx1, dist2, idx2 = nn_distance(xyz1, xyz2)

    d1, _i1, d2, _i2 = session.run([dist1, idx1, dist2, idx2], feed_dict={xyz1: source, xyz2: dest})

    for i in range(len(thresholds)):
        thresh = thresholds[i]

        num = len(np.where(d1 <= thresh)[0]) + 0.0
        P = 100.0 * (num / len_points)

        num = len(np.where(d2 <= thresh)[0]) + 0.0
        R = 100.0 * (num / len_labels)

        f_scores.append((2 * P * R) / (P + R + 1e-6))

    return np.array(f_scores)

def mean_surface_distance(source, dest, n_sample=None, registered=False):
    #TODO : Check if we need/want to register before metrics
    if not registered:
        dest = _register_without_scale(source, dest)

    if n_sample:
        dest = np.random.permutation(dest)[:n_sample]

    nearest_neighbors = NearestNeighbors(n_neighbors=1, n_jobs=4)
    nearest_neighbors.fit(dest)

    distances, _ = nearest_neighbors.kneighbors(source)

    return np.mean(np.square(distances), axis=0)


def symmetric_mean_surface_distance(source, dest, n_sample=None, registered=False):
    #TODO : Check if we need/want to register before metrics
    mse_1 = mean_surface_distance(source, dest, n_sample, registered)
    mse_2 = mean_surface_distance(source=dest, dest=source, n_sample=n_sample, registered=registered)

    return (mse_1 * len(source) + mse_2 * len(dest)) / (len(source) + len(dest))


def hausdorff(source, dest, registered=False):
    #TODO : Check if we need/want to register before metrics
    if not registered:
        dest = _register_without_scale(source, dest)

    return directed_hausdorff(source, dest)

def symmetric_hausdorff(source, dest, registered=False):
    #TODO : Check if we need/want to register before metrics
    if not registered:
        dest = _register_without_scale(source, dest)

    return max(directed_hausdorff(source, dest), directed_hausdorff(dest, source))


def signed_mean_point_to_surface(points, vertices, faces, n_sample=None, registered=False):
    ''' Tiré de : https://github.com/intel-isl/Open3D/issues/2062
    les points à l'extérieur ont une distance positives, intérieur négatives
    '''
    
    #TODO : Check if we need/want to register before metrics
    if not registered:
        vertices = _register_without_scale(points, vertices)

    if n_sample:
        points = np.random.permutation(points)[:n_sample]

    tri_mesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf_tri_mesh = trimesh.proximity.ProximityQuery(tri_mesh_box)

    return np.mean(-sdf_tri_mesh.signed_distance(points))


def mean_point_to_surface(points, vertices, faces, n_sample=None, registered=False):
    #TODO : Check if we need/want to register before metrics
    if not registered:
        vertices = _register_without_scale(points, vertices)

    if n_sample:
        points = np.random.permutation(points)[:n_sample]

    tri_mesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf_tri_mesh = trimesh.proximity.ProximityQuery(tri_mesh_box)

    return np.mean(np.abs(sdf_tri_mesh.signed_distance(points)))