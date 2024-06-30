import random
import unittest

import numpy as np

from solution import MultinomialLogReg, OrdinalLogReg, \
    multinomial_bad_ordinal_good, MBOG_TRAIN


class HW2Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [1, 1]])
        self.y = np.array([0, 0, 1, 1, 2])
        self.train = self.X[::2], self.y[::2]
        self.test = self.X[1::2], self.y[1::2]

    def test_multinomial(self):
        l = MultinomialLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_ordinal(self):
        l = OrdinalLogReg()
        c = l.build(self.X, self.y)
        prob = c.predict(self.test[0])
        self.assertEqual(prob.shape, (2, 3))
        self.assertTrue((prob <= 1).all())
        self.assertTrue((prob >= 0).all())
        np.testing.assert_almost_equal(prob.sum(axis=1), 1)

    def test_multinomial_bad_ordinal_good(self):
        rand = random.Random(0)
        X, y = multinomial_bad_ordinal_good(100, rand)
        self.assertEqual(len(X), 100)
        self.assertEqual(y.shape, (100,))
        rand = random.Random(1)
        X1, y1 = multinomial_bad_ordinal_good(100, rand)
        self.assertTrue((X != X1).any())
        trainX, trainY = multinomial_bad_ordinal_good(MBOG_TRAIN, random.Random(42))
        self.assertEqual(len(trainX), MBOG_TRAIN)


if __name__ == "__main__":
    unittest.main()
