import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
import tensorflow as tf
import open3d as o3d
import trimesh

from modules.chamfer import nn_distance

def _rigid_register(source, dest, with_registration_results=False):
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)

    dest_pc = o3d.geometry.PointCloud()
    dest_pc.points = o3d.utility.Vector3dVector(dest)

    #TODO : Figure out best param from threshold
    threshold=2
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pc, dest_pc, threshold, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

    if with_registration_results:
        return np.asarray(dest_pc.transform(reg_p2p.transformation).points), reg_p2p    

    return np.asarray(dest_pc.transform(reg_p2p.transformation).points)


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


def absolute_mean_surface_distance(source, dest, n_sample=None, registered=False):
    if not registered:
        dest = _rigid_register(source, dest)

    if n_sample:
        dest = np.random.permutation(dest)[:n_sample]

    nearest_neighbors = NearestNeighbors(n_neighbors=1, n_jobs=4)
    nearest_neighbors.fit(dest)

    distances, _ = nearest_neighbors.kneighbors(source)

    return np.mean(np.abs(distances), axis=0)


def symmetric_absolute_mean_surface_distance(source, dest, n_sample=None, registered=False):
    mse_1 = absolute_mean_surface_distance(source, dest, n_sample, registered)
    mse_2 = absolute_mean_surface_distance(source=dest, dest=source, n_sample=n_sample, registered=registered)

    return (mse_1 * len(source) + mse_2 * len(dest)) / (len(source) + len(dest))


def hausdorff(source, dest, registered=False):
    if not registered:
        dest = _rigid_register(source, dest)

    return directed_hausdorff(source, dest)


def symmetric_hausdorff(source, dest, registered=False):
    if not registered:
        dest = _rigid_register(source, dest)

    return max(directed_hausdorff(source, dest), directed_hausdorff(dest, source))


def signed_mean_point_to_surface(points, vertices, faces, n_sample=None, registered=False):
    ''' Tiré de : https://github.com/intel-isl/Open3D/issues/2062
    les points à l'extérieur ont une distance positives, intérieur négatives
    '''
    if not registered:
        vertices = _rigid_register(points, vertices)

    if n_sample:
        points = np.random.permutation(points)[:n_sample]

    tri_mesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf_tri_mesh = trimesh.proximity.ProximityQuery(tri_mesh_box)

    return np.mean(-sdf_tri_mesh.signed_distance(points))


def abs_mean_point_to_surface(points, vertices, faces, n_sample=None, registered=False):
    if not registered:
        vertices = _rigid_register(points, vertices)

    if n_sample:
        points = np.random.permutation(points)[:n_sample]

    tri_mesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf_tri_mesh = trimesh.proximity.ProximityQuery(tri_mesh_box)

    return np.mean(np.abs(sdf_tri_mesh.signed_distance(points)))


def rms_point_to_surface(points, vertices, faces, n_sample=None, registered=False):
    if not registered:
        vertices = _rigid_register(points, vertices)

    if n_sample:
        points = np.random.permutation(points)[:n_sample]

    tri_mesh_box = trimesh.Trimesh(vertices=vertices, faces=faces)
    sdf_tri_mesh = trimesh.proximity.ProximityQuery(tri_mesh_box)

    return np.sqrt(np.sum(sdf_tri_mesh.signed_distance(points))**2)/points.shape[0]