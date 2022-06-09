import unittest
import siddon as sd
import numpy as np

class TestSiddonAlgorithm(unittest.TestCase):
    
    def setUp(self):
        self.parameters = {}
        self.parameters['num_planes'] = [(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)]
        self.parameters['voxel_lengths'] = [(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5)]
        self.parameters['beam_coor'] = [((5,0),(1,1),(3,3)),((6,-1),(1,1),(3,3)),((3,1),(1,1),(3,3)),((1,1),(85,-3),(3,3)),((2,2),(1,1),(45,1)),((5,-2),(5,-2),(5,-2))]
        self.parameters['ini_planes'] = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(-1,-1,-1)]
        
        self.total_distance_expected = [5,5,2,5,4,8.660254038]
        # self.voxel_distance_expected = 
    
    def test_total_distance(self):
        for n in range(len(self.total_distance_expected)):
            voxel_info = sd.Siddon(self.parameters['num_planes'][n],self.parameters['voxel_lengths'][n],self.parameters['beam_coor'][n],self.parameters['ini_planes'][n])
            total = 0
            for i in range(len(voxel_info)):
                total += voxel_info[i]['d']
            self.assertAlmostEqual(self.total_distance_expected[n],total)
    
    # def test_voxel_distance(self):
    #     for n in range(len(self.total_distance_expected)):
    #         voxel_info = sd.Siddon(self.parameters['num_planes'][n],self.parameters['voxel_lengths'][n],self.parameters['beam_coor'][n],self.parameters['ini_planes'][n])

# class TestTERMAMonoenergetically(unittest.TestCase):
    
#     def setUp(self):
    
#     def test_terma_returns(self):

class TestSuperposition(unittest.TestCase):
    
    def test_with_uniform_terma(self):
        test_kernel = np.array([[[0,0,0],[0,0.1,0],[0,0,0]],
                     [[0,0.1,0],[0.1,0.4,0.1],[0,0.1,0]],
                     [[0,0,0],[0,0.1,0],[0,0,0]]])
        test_array_1 = np.array([[[1,1],[1,1]],
                      [[1,1],[1,1]]])
        voxel_info_1 = []
        n=0
        for x in range(len(test_array_1)):
            for y in range(len(test_array_1[0])):
                for z in range(len(test_array_1[0][0])):
                    voxel_info_1.append({})
                    voxel_info_1[n]['indices'] = (x+1,y+1,z+1)
                    voxel_info_1[n]['TERMA'] = 1
                    n += 1
        
        
        self.assertEqual(sd.Superposition(test_kernel,(2,2,2),(3,3,3),(2,2,2),voxel_info_1).all(),test_array_1.all())