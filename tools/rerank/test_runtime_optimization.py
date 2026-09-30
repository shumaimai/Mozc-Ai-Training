"""Verify duplicate scoring and stable native ordering at the runtime boundary."""
import importlib.util
import os
from pathlib import Path
import unittest
import numpy as np

RUNTIME_ROOT=Path(os.environ.get("MOZCAI_RUNTIME_ROOT",
    str(Path(__file__).resolve().parents[3]/"Mozc-Ai")))
RUNTIME=RUNTIME_ROOT/"runtime/rerank_daemon.py"
spec=importlib.util.spec_from_file_location("runtime_optimization_test",RUNTIME)
runtime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeOptimizationTest(unittest.TestCase):
    def test_unique_forward_restores_every_original_position(self):
        encoded=[]
        class Tokenizer:
            def encode(self,text,out_type):
                encoded.append(text)
                return [10 if text=="a" else 20]
        class Session:
            def run(self,outputs,inputs):
                self.ids=inputs["input_ids"]
                return [self.ids[:,1].astype(np.float32)]
        scorer=runtime.OrtScorer.__new__(runtime.OrtScorer)
        scorer.tokenizer=Tokenizer();scorer.session=Session();scorer.max_len=128
        self.assertEqual(scorer.score(["a","b","a","b"]),[10.,20.,10.,20.])
        self.assertEqual(encoded,["a","b"])
        self.assertEqual(scorer.session.ids.shape,(2,3))
        self.assertEqual(scorer.score([]),[])

    def test_equal_scores_keep_native_first_candidate(self):
        class Constant:
            def score(self,texts):return [5.]*len(texts)
        response=runtime.rerank({"reading":"きしゃ","context_prev":"新聞の",
            "nbest":["汽車","記者","汽車"]},Constant(),0.,30)
        self.assertEqual(response["final_top1"],"汽車")
        self.assertFalse(response["overwritten"])

    def test_guard_does_not_invoke_model(self):
        class Fail:
            def score(self,texts):raise AssertionError("guard must skip")
        response=runtime.rerank({"reading":"い","context_prev":"彼の",
            "nbest":["意","胃"]},Fail(),1.5,30)
        self.assertTrue(response["guard_skip"])
        self.assertEqual(response["final_top1"],"意")


if __name__=="__main__":unittest.main()
