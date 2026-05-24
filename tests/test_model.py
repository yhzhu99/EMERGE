from __future__ import annotations

import torch

from models.emerge import EMERGEPredictor


def test_emerge_forward_shape() -> None:
    model = EMERGEPredictor(
        input_ehr_dim=61,
        input_note_dim=768,
        input_summary_dim=768,
        hidden_dim=32,
        ehr_net="gru",
        text_fusion="concat",
        modality_fusion="ours",
        use_modality="ehr_note_summary",
        num_heads=4,
        dropout=0.0,
    )
    logits = model(
        torch.randn(3, 4, 61),
        torch.randn(3, 768),
        torch.randn(3, 768),
    )
    assert logits.shape == (3, 2)
