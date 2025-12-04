
import unittest
from niosh_calculator import NIOSHCalculator

class TestNIOSHCalculatorBounds(unittest.TestCase):
    def setUp(self):
        self.calc = NIOSHCalculator()

    def test_hm_bounds(self):
        # H > 25 should return 0
        self.assertEqual(self.calc._hm(26.0), 0.0)
        # H <= 25 should be valid
        self.assertGreater(self.calc._hm(25.0), 0.0)
        # H < 10 should be clamped to 10 (HM = 1.0)
        self.assertEqual(self.calc._hm(5.0), 1.0)

    def test_vm_bounds(self):
        # V > 70 should return 0
        self.assertEqual(self.calc._vm(71.0), 0.0)
        # V <= 70 should be valid
        self.assertGreater(self.calc._vm(70.0), 0.0)
        # V < 0 should return 0 (as implemented, though technically V cannot be negative physically, but input validation)
        self.assertEqual(self.calc._vm(-1.0), 0.0)
        # V = 0 should be valid
        self.assertGreater(self.calc._vm(0.0), 0.0)

    def test_dm_bounds(self):
        # D > 70 should return 0
        self.assertEqual(self.calc._dm(71.0), 0.0)
        # D <= 70 should be valid
        self.assertGreater(self.calc._dm(70.0), 0.0)
        # D < 10 should be clamped to 10 (DM = 1.0)
        self.assertEqual(self.calc._dm(5.0), 1.0)

if __name__ == '__main__':
    unittest.main()
