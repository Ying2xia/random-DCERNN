import unittest
import numpy as np
from scipy.special import expit
from fedcernn.simulation import generate_simulation, error_expectiles

class CandidateDesignTests(unittest.TestCase):
    def test_analytic_functions_and_domains(self):
        for name,p in [('correlated_multi_index',20),('hierarchical_composition',30)]:
            data=generate_simulation(1000,name,'normal',seed=418)
            x=data.X
            self.assertEqual(x.shape,(1000,p))
            self.assertTrue(np.all((x>-1)&(x<1)))
            self.assertTrue(np.all((data.scale>.6)&(data.scale<1)))
            if p==20:
                mu=2*np.sin(np.pi*(x[:,0]+x[:,1])/2)+1.5*np.tanh(x[:,2]-x[:,3])+.75*np.cos(np.pi*x[:,4])+.75*x[:,5]+.5*np.sin(np.pi*x[:,6]*x[:,7])
                sigma=.6+.4*expit(x[:,0]+x[:,8])
                self.assertGreater(np.corrcoef(x[:,0],x[:,1])[0,1],.3)
            else:
                mu=2*np.sin(np.pi*((x[:,0]+x[:,1])/2+.5*x[:,2]*x[:,3]))+1.5*((x[:,4]-x[:,5])**2/4-1/6)+.75*np.cos(np.pi*x[:,6])+.5*x[:,7]*x[:,8]
                sigma=.6+.4*expit(x[:,0]+x[:,9])
            np.testing.assert_allclose(data.location,mu)
            np.testing.assert_allclose(data.scale,sigma)
            np.testing.assert_allclose(data.true_expectiles,mu[:,None]+sigma[:,None]*error_expectiles('normal',data.taus))
            self.assertFalse(np.array_equal(data.X,generate_simulation(1000,name,'normal',seed=419).X))

    def test_existing_designs_remain_registered(self):
        for name,p in [('smooth_additive',10),('friedman',10),('correlated_interaction',20),('sparse_interaction',30)]:
            self.assertEqual(generate_simulation(10,name,seed=12).n_features,p)

if __name__=='__main__':unittest.main()
