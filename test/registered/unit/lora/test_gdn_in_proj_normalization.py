"""CPU-only regression tests for GDN LoRA in_proj normalization.

Covers the PEFT adapter layouts that Qwen3.5-style GatedDeltaNet layers can
ship with:

* split ``in_proj_b`` + ``in_proj_a``
* already-fused ``in_proj_ba``
* split ``in_proj_q`` / ``k`` / ``v`` / ``z``
* 2-way ``in_proj_qkv`` + ``in_proj_z`` (Megatron-Bridge export)

No CUDA or server is required; the test calls the normalization helpers
directly with small tensors.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import get_normalized_target_modules, get_stacked_multiply
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PREFIX = "base_model.model.model.layers.0.linear_attn"


def _key(name: str) -> str:
    return f"{PREFIX}.{name}"


class TestGDNInProjNormalization(unittest.TestCase):
    def setUp(self):
        # The normalization helpers only touch the weights dict, so bypass
        # LoRAAdapter.__init__ and avoid loading a model.
        self.adapter = LoRAAdapter.__new__(LoRAAdapter)

    def test_split_b_a_is_stacked_into_ba(self):
        b_a = torch.randn(4, 8)
        b_b = torch.randn(16, 4)
        a_a = torch.randn(4, 8)
        a_b = torch.randn(16, 4)
        weights = {
            _key("in_proj_b.lora_A.weight"): b_a,
            _key("in_proj_b.lora_B.weight"): b_b,
            _key("in_proj_a.lora_A.weight"): a_a,
            _key("in_proj_a.lora_B.weight"): a_b,
        }

        self.adapter._normalize_in_proj_ba(weights)

        self.assertEqual(
            set(weights),
            {
                _key("in_proj_ba.lora_A.weight"),
                _key("in_proj_ba.lora_B.weight"),
            },
        )
        self.assertTrue(
            torch.equal(
                weights[_key("in_proj_ba.lora_A.weight")],
                torch.cat((b_a, a_a), dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                weights[_key("in_proj_ba.lora_B.weight")],
                torch.cat((b_b, a_b), dim=0),
            )
        )

    def test_merged_ba_repeats_shared_a(self):
        a = torch.randn(4, 8)
        b = torch.randn(32, 4)
        weights = {
            _key("in_proj_ba.lora_A.weight"): a,
            _key("in_proj_ba.lora_B.weight"): b,
        }

        self.adapter._normalize_in_proj_ba(weights)

        self.assertEqual(tuple(weights[_key("in_proj_ba.lora_A.weight")].shape), (8, 8))
        self.assertTrue(
            torch.equal(
                weights[_key("in_proj_ba.lora_A.weight")],
                torch.cat((a, a), dim=0),
            )
        )
        self.assertIs(weights[_key("in_proj_ba.lora_B.weight")], b)

    def test_four_way_qkvz_split_is_stacked(self):
        parts = {
            name: (torch.randn(4, 8), torch.randn(16, 4))
            for name in ("q", "k", "v", "z")
        }
        weights = {}
        for name, (a, b) in parts.items():
            weights[_key(f"in_proj_{name}.lora_A.weight")] = a
            weights[_key(f"in_proj_{name}.lora_B.weight")] = b

        self.adapter._normalize_in_proj_qkvz(weights)

        self.assertEqual(
            tuple(weights[_key("in_proj_qkvz.lora_A.weight")].shape), (16, 8)
        )
        self.assertEqual(
            tuple(weights[_key("in_proj_qkvz.lora_B.weight")].shape), (64, 4)
        )
        self.assertTrue(
            torch.equal(
                weights[_key("in_proj_qkvz.lora_A.weight")],
                torch.cat([parts[n][0] for n in ("q", "k", "v", "z")], dim=0),
            )
        )

    def test_two_way_qkv_z_split_repeats_qkv_a_three_times(self):
        qkv_a = torch.randn(4, 8)
        qkv_b = torch.randn(24, 4)
        z_a = torch.randn(4, 8)
        z_b = torch.randn(8, 4)
        weights = {
            _key("in_proj_qkv.lora_A.weight"): qkv_a,
            _key("in_proj_qkv.lora_B.weight"): qkv_b,
            _key("in_proj_z.lora_A.weight"): z_a,
            _key("in_proj_z.lora_B.weight"): z_b,
        }

        self.adapter._normalize_in_proj_qkvz(weights)

        self.assertTrue(
            torch.equal(
                weights[_key("in_proj_qkvz.lora_A.weight")],
                torch.cat((qkv_a, qkv_a, qkv_a, z_a), dim=0),
            )
        )
        self.assertTrue(
            torch.equal(
                weights[_key("in_proj_qkvz.lora_B.weight")],
                torch.cat((qkv_b, z_b), dim=0),
            )
        )

    def test_target_module_mapping_and_stacked_rank(self):
        self.assertEqual(
            get_normalized_target_modules(
                ["linear_attn.in_proj_b", "linear_attn.in_proj_a"]
            ),
            {"in_proj_ba"},
        )
        self.assertEqual(get_stacked_multiply("in_proj_ba"), 2)

    def test_qwen3_5_ba_hidden_dim(self):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        model.config = SimpleNamespace(
            hidden_size=2048,
            head_dim=128,
            num_attention_heads=16,
            linear_num_value_heads=32,
        )
        self.assertEqual(model.get_hidden_dim("in_proj_ba", layer_idx=0), (2048, 64))


if __name__ == "__main__":
    unittest.main()
