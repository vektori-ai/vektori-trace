"""Focused checks for trainer invariants that previously failed only on GPU."""

from vektori_trace.tau2.reopd_trainer import _canonical_lora_key


def test_peft_live_and_saved_lora_names_canonicalize_identically():
    live = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    saved = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert _canonical_lora_key(live) == _canonical_lora_key(saved)


def test_canonicalization_preserves_layer_identity():
    a = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    b = "base_model.model.model.layers.1.self_attn.q_proj.lora_B.default.weight"
    assert _canonical_lora_key(a) != _canonical_lora_key(b)


def test_non_default_adapter_names_are_not_silently_erased():
    named = "base_model.model.layer.lora_A.experiment.weight"
    assert _canonical_lora_key(named) == named
