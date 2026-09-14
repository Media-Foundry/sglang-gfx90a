"""Preflight experiment imports without starting a model or accessing GPUs."""
from pathlib import Path
import subprocess
import sys
import unittest


class Tools(unittest.TestCase):
    def test_no_stdlib_shadowing(self):
        root=Path(__file__).resolve().parent
        self.assertFalse({p.stem for p in root.glob('*.py')} & sys.stdlib_module_names)

    def test_imports_and_help_from_experiment_directory(self):
        root=Path(__file__).resolve().parent
        script='import cProfile, profile; from transformers import AutoTokenizer; from pathlib import Path; assert Path(profile.__file__).parent == Path(cProfile.__file__).parent'
        subprocess.run([sys.executable,'-c',script],cwd=root,check=True,timeout=60)
        for name in ('run.py','sweep.py','capture_timeline.py'):
            subprocess.run([sys.executable,name,'--help'],cwd=root,check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)


if __name__=='__main__':unittest.main()
