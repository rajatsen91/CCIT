from unittest import TestCase

from CCIT import *


class TestCCIT(TestCase):
    def test_datagen(self):
        allsamples = generate_samples_cos()
        m,n = allsamples.shape
        print 'CI Samples correctly generated'
        self.assertTrue(m == 1000)
        self.assertTrue(n == 22)

    def test_CCIT(self):
        allsamples = generate_samples_cos()
        pval = CCIT(allsamples[:,0:1],allsamples[:,1:2],allsamples[:,2:22])
        print 'pvalue: ' + str(pval)
        self.assertTrue(pval <= 1)

        
