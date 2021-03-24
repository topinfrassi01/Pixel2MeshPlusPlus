import unittest
import os
#pylint: disable=import-error
import modules.metrics as metrics
import numpy as np
import tensorflow as tf
from math import sin, cos, sqrt

np.random.seed(1)

class Metrics_Rigid_Register_Tests(unittest.TestCase):
    def test_rigid_registration_same(self):
        source = np.random.uniform(size=(10,3))
        dest = source

        _, stats = metrics._rigid_register(source, dest, True)

        self.assertAlmostEqual(stats.fitness, 1)
        self.assertAlmostEqual(stats.inlier_rmse, 0)
        
    def test_rigid_registration_translation(self):
        source = np.random.uniform(size=(10,3))
        dest = source + 0.2

        _, stats = metrics._rigid_register(source, dest, True)

        self.assertAlmostEqual(stats.fitness, 1)
        self.assertAlmostEqual(stats.inlier_rmse, 0)


    def test_rigid_registration_rotation(self):
        quarter_pi = 3.1416 / 10
        transform = np.asarray([
            [1, 0, 0],
            [0, cos(quarter_pi), -sin(quarter_pi)],
            [0, sin(quarter_pi), cos(quarter_pi)]
        ])

        source = np.random.uniform(size=(10,3))
        dest = (np.matmul(transform, source.T).T)

        _, stats = metrics._rigid_register(source, dest, True)
        
        self.assertAlmostEqual(stats.fitness, 1)
        self.assertAlmostEqual(stats.inlier_rmse, 0)


class TestChamferDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    @classmethod
    def tearDownClass(cls):
        del os.environ['CUDA_VISIBLE_DEVICES']

    def setUp(self):
        config = tf.ConfigProto()

        #pylint: disable=no-member
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        

    def tearDown(self):
        self.sess.close()

    def test_chamfer_same(self):
        source = np.asarray([
            [1,1,1],
            [2,2,2]
        ])
        dest = source

        distance = metrics.chamfer_distance(source, dest, self.sess)
        
        self.assertAlmostEqual(distance, 0)

    def test_chamfer_same_number_of_points(self):
        source = np.asarray([
            [1,1,1]
        ])
        dest = np.asarray([
            [2,2,2]
        ])

        distance = metrics.chamfer_distance(source, dest, self.sess)

        #Expected result is six because from 2,2,2 to 1,1,1 there is a sqrt(3) distance, 
        #which squared gives 3 and chamfer distance is computed on chamfer(p,q)+chamfer(q,p)
        self.assertAlmostEqual(distance, 6)


    def test_chamfer_averages(self):
        source = np.asarray([
            [1,1,1],
            [1,1,1]
        ])
        dest = np.asarray([
            [2,2,2],
            [2,2,2]
        ])

        distance = metrics.chamfer_distance(source, dest, self.sess)

        #Expected result is six because from 2,2,2 to 1,1,1 there is a sqrt(3) distance, 
        #which squared gives 3 and chamfer distance is computed on avg(chamfer(p,q))+avg(chamfer(q,p))
        self.assertAlmostEqual(distance, 6)


class TestDiceDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    @classmethod
    def tearDownClass(cls):
        del os.environ['CUDA_VISIBLE_DEVICES']

    def setUp(self):
        config = tf.ConfigProto()

        #pylint: disable=no-member
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        
    def tearDown(self):
        self.sess.close()

    def test_dice_same_points(self):
        source = np.asarray([
            [2,2,2],
            [2,2,2]
        ])
        dest = np.asarray([
            [2,2,2],
            [2,2,2]
        ])

        distance = metrics.dice(source, dest, thresholds=[0.05], session=self.sess)

        self.assertAlmostEqual(distance[0], 100, delta=0.001)

    def test_dice_completely_outside(self):
        source = np.asarray([
            [2,2,2],
            [2,2,2]
        ])
        dest = np.asarray([
            [20,20,20],
            [20,20,20]
        ])

        distance = metrics.dice(source, dest, thresholds=[0.05], session=self.sess)

        self.assertAlmostEqual(distance[0], 0, delta=0.001)


class TestMeanSurfaceErrorDistance(unittest.TestCase):
    def test_amse(self):
        source = np.asarray([
            [1,1,1]
        ])

        dest = np.asarray([
            [2,2,2]
        ])

        distance = metrics.absolute_mean_surface_distance(source, dest)

        self.assertAlmostEqual(distance, sqrt(3), delta=0.001)

    def test_sym_amse(self):
        source = np.asarray([
            [1,1,1],
            [-1,-1,-1]
        ])

        dest = np.asarray([
            [2,2,2],
            [-2,-2,-2]
        ])

        distance = metrics.symmetric_absolute_mean_surface_distance(source, dest)

        self.assertAlmostEqual(distance, 2*sqrt(3), delta=0.001)


class TestPointToSurfaceDistance(unittest.TestCase):
    def test_signed_mp2s(self):
        pass

    def test_abs_p2s(self):
        pass

    def test_rms_p2s(self):
        pass
    
if __name__ == '__main__':
    unittest.main()