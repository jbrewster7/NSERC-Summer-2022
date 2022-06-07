import unittest
import siddon as sd

class TestSiddonAlgorithm(unittest.TestCase):
    
    def setUp(self):
        self.parameters = {}
        self.parameters['num_planes'] = [(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)]
        self.parameters['voxel_lengths'] = [(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5),(2.5,2.5,2.5)]
        self.parameters['beam_coor'] = [((5,0),(1,1),(3,3)),((6,-1),(1,1),(3,3)),((3,1),(1,1),(3,3)),((1,1),(85,-3),(3,3)),((2,2),(1,1),(45,1)),((5,-2),(5,-2),(5,-2))]
        self.parameters['ini_planes'] = [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(-1,-1,-1)]
        
        self.total_distance_expected = [5,5,2,5,4,8.660254038]
    
    def test_total_distance(self):
        for n in range(len(self.total_distance_expected)):
            voxel_info = sd.Siddon(self.parameters['num_planes'][n],self.parameters['voxel_lengths'][n],self.parameters['beam_coor'][n],self.parameters['ini_planes'][n])
            total = 0
            for i in range(len(voxel_info)):
                total += voxel_info[i]['d']
            self.assertAlmostEqual(self.total_distance_expected[n],total)

class TestTERMAMonoenergetically(unittest.TestCase):