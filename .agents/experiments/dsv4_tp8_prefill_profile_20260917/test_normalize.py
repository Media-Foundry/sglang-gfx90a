"""No GPU/model launch: normalization and terminal-environment contracts."""
import unittest
from normalize import EXEC, SOURCE, normalize, verify


class Normalize(unittest.TestCase):
    def test_actual_measured_launcher(self):
        source=SOURCE.read_text()
        candidate,actions=normalize(source)
        self.assertEqual(normalize(candidate)[0],candidate)
        report=verify(source,candidate)
        self.assertEqual(report['status'],'cpu_environment_equivalent')
        self.assertFalse(report['service_validated'])
        self.assertEqual(actions['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT'],'1')
        self.assertEqual(actions['SGLANG_DSV4_PREFILL_CK_ROUTE_PRODUCER'],'1')
        self.assertIsNone(actions['SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR'])

    def test_literal_last_assignment_and_unset(self):
        source='set -euo pipefail\ncd /tmp\nexport VALUE=first EMPTY=\nexport VALUE="last value"\nunset EMPTY\n'+EXEC+'\n'
        candidate,actions=normalize(source)
        self.assertEqual(actions,{'VALUE':'last value','EMPTY':None})
        self.assertEqual(len(verify(source,candidate)['checks']),2)

    def test_reject_nonliteral_code(self):
        for line in ('export X=$UNSET','export X=`false`','export X=x; false',
                     'source /tmp/file','echo unexpected','export BAD-NAME=x'):
            with self.subTest(line=line):
                with self.assertRaises((AssertionError,ValueError)):
                    normalize('cd /tmp\n'+line+'\n'+EXEC+'\n')
        with self.assertRaises(ValueError):normalize('cd /tmp\n'+EXEC+'\nexport AFTER=x\n')


if __name__=='__main__':unittest.main()
