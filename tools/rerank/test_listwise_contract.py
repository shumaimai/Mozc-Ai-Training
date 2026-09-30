"""Gold coverage, duplicate labels, runtime input, and permutation checks."""
import unittest
from tools.rerank.listwise_data import prepare


class Tokenizer:
    def encode(self,text,out_type):return [ord(c) for c in text]


class ListwiseContractTest(unittest.TestCase):
    def row(self,gold="記者"):
        return {"reading":"キシャ","context_prev":" 新聞の", "gold":gold,
            "eligibility_status":"NEURAL_ELIGIBLE","candidates":[
                {"surface":"汽車","protection":"NORMAL"},
                {"surface":"記者","protection":"NORMAL"},
                {"surface":"汽車","protection":"NORMAL"}]}

    def test_duplicate_surface_has_one_target_and_runtime_text(self):
        group=prepare([self.row()],Tokenizer(),train=True)[0]
        self.assertEqual(group["candidates"],["汽車","記者"])
        self.assertEqual(group["gold_index"],1)
        self.assertEqual(group["reading"],"きしゃ")
        self.assertEqual(group["context"],"新聞の")
        self.assertEqual(group["tokens"][0][0],1)
        self.assertEqual(group["tokens"][0][-1],2)

    def test_missing_gold_is_not_injected(self):
        self.assertEqual(prepare([self.row("貴社")],Tokenizer(),train=True),[])
        group=prepare([self.row("貴社")],Tokenizer(),train=False)[0]
        self.assertEqual(group["gold_index"],-1)
        self.assertNotIn("貴社",group["candidates"])

    def test_candidate_permutation_moves_target_with_surface(self):
        row=self.row();row["candidates"].reverse()
        group=prepare([row],Tokenizer(),train=True)[0]
        self.assertEqual(group["candidates"][group["gold_index"]],"記者")

    def test_protected_rows_are_evaluation_only(self):
        row=self.row();row["eligibility_status"]="PROTECTED_EVAL_ONLY"
        self.assertEqual(prepare([row],Tokenizer(),train=True),[])
        self.assertEqual(len(prepare([row],Tokenizer(),train=False)),1)


if __name__=="__main__":unittest.main()
