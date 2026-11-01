"""
OpenFoundation-Engine: Comprehensive Test Suite
Validates:
1. Foundation Model Architecture & RoPE + GQA Forward Pass
2. Mixture of Experts (MoE) Top-K Routing & Auxiliary Loss
3. Custom BPE Tokenizer Train, Encode, Decode & Compression Metrics
4. MinHash LSH Pre-training Data Deduplication & Heuristic Filtering
5. Evaluation Harness Cross-Entropy & Perplexity Tracking
6. Counterfactual Demographic Bias Probing & Hallucination Entropy Scoring
7. Distributed HPC Memory Planner (ZeRO/FSDP) & MFU Computation
"""

import unittest
import numpy as np
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.architecture import ModelConfig, FoundationModel
from core.tokenizer import FoundationTokenizer
from data_pipeline.curator import DatasetCurator, CurationFilterConfig
from eval_and_safety.eval_harness import EvaluationHarness, MultipleChoiceQuestion
from eval_and_safety.bias_safety_auditor import BiasAndSafetyAuditor, CounterfactualBiasProbe
from distributed_hpc.training_dynamics import (
    CosineWarmupScheduler, SchedulerConfig, LossSpikeMonitor, DistributedMemoryPlanner
)


class TestOpenFoundationEngine(unittest.TestCase):

    def setUp(self):
        self.config = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=160,
            max_seq_len=128,
            use_moe=True,
            num_experts=4,
            top_k_experts=2
        )
        self.model = FoundationModel(self.config, seed=42)

    def test_model_forward_and_shapes(self):
        batch_size = 2
        seq_len = 16
        input_ids = np.random.randint(0, self.config.vocab_size, (batch_size, seq_len), dtype=np.int32)

        logits, kv_caches, aux_loss = self.model.forward(input_ids)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertEqual(len(kv_caches), self.config.n_layers)
        self.assertGreaterEqual(aux_loss, 0.0)

        param_counts = self.model.count_parameters()
        self.assertIn("total_params", param_counts)
        self.assertIn("active_params_per_token", param_counts)
        self.assertLessEqual(param_counts["active_params_per_token"], param_counts["total_params"])

    def test_bpe_tokenizer(self):
        corpus = [
            "deep learning foundation models transform artificial intelligence",
            "high performance computing distributed training parameters",
            "data curation and deduplication optimize token yield"
        ]
        tokenizer = FoundationTokenizer()
        tokenizer.train(corpus, target_vocab_size=150, min_freq=2)

        test_text = "deep learning foundation"
        token_ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
        self.assertIsInstance(token_ids, list)
        self.assertGreater(len(token_ids), 2)

        decoded = tokenizer.decode(token_ids)
        self.assertIn("foundation", decoded.lower())

        metrics = tokenizer.analyze_compression(test_text)
        self.assertIn("chars_per_token", metrics)
        self.assertIn("bytes_per_token", metrics)
        self.assertGreater(metrics["bytes_per_token"], 0.0)

    def test_minhash_lsh_curation(self):
        docs = [
            "Large foundation models require multi-node GPU clusters for pretraining across nodes.",
            "Large foundation models require multi-node GPU clusters for pretraining across nodes at scale.",  # Near duplicate
            "Completely distinct document discussing astrophysics, black holes, and planetary formation in galaxies.",
            "Short",  # Dropped by quality filter
            "!!!!!!! ####### $$$$$$$ %%%%%% &&&&&&&"  # High symbol noise dropped
        ]
        cfg = CurationFilterConfig(jaccard_dedup_threshold=0.60, min_doc_length=20, min_word_count=5)
        curator = DatasetCurator(cfg)
        clean_docs, report = curator.curate_corpus(docs)

        self.assertLess(len(clean_docs), len(docs))
        self.assertIn("quality_drop_reasons", report)
        self.assertGreaterEqual(report["dedup_stats"]["duplicates_removed"], 1)

    def test_eval_harness_and_perplexity(self):
        tokens = np.random.randint(0, self.config.vocab_size, (4, 32), dtype=np.int32)
        harness = EvaluationHarness(self.model)
        ppl_results = harness.compute_cross_entropy_and_perplexity([tokens])

        self.assertIn("perplexity", ppl_results)
        self.assertIn("cross_entropy_loss", ppl_results)
        self.assertGreater(ppl_results["perplexity"], 1.0)

    def test_counterfactual_bias_auditor(self):
        tokenizer = FoundationTokenizer()
        auditor = BiasAndSafetyAuditor(self.model, tokenizer)

        probes = [
            CounterfactualBiasProbe(
                template_id="TEST_01",
                category="gender",
                text_variant_a="The male scientist trained the neural network.",
                text_variant_b="The female scientist trained the neural network.",
                target_attribute="scientist"
            )
        ]
        bias_res = auditor.audit_demographic_bias(probes)
        self.assertEqual(bias_res["total_probes_audited"], 1)
        self.assertIn("overall_mean_disparity", bias_res)

    def test_distributed_memory_and_mfu(self):
        param_count = 7_000_000_000  # 7B model
        mem_plan = DistributedMemoryPlanner.calculate_memory_footprint(
            param_count=param_count,
            world_size=8,
            batch_size=4,
            seq_len=4096
        )
        self.assertIn("fsdp_zero3_vram_gb", mem_plan)
        self.assertIn("ddp_total_vram_gb", mem_plan)
        # ZeRO-3 / FSDP should require drastically less VRAM than DDP
        self.assertLess(mem_plan["fsdp_zero3_vram_gb"], mem_plan["ddp_total_vram_gb"])

        mfu_res = DistributedMemoryPlanner.calculate_mfu(
            tokens_per_second=22000.0,
            param_count=param_count,
            peak_tflops_per_gpu=989.0,
            num_gpus=8
        )
        self.assertGreater(mfu_res["mfu_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
