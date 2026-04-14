import torch
import pytest

from genmol.mm.prefix import extract_molecule_logits, pack_prefix_conditioning, pad_prefix_embeddings


def test_pack_prefix_conditioning_aligns_token_positions_after_prefix():
    token_embeddings = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
        ]
    )
    token_attention_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    prefix_embeddings = [
        torch.tensor([[10.0, 10.0]]),
        torch.tensor([[20.0, 20.0], [21.0, 21.0]]),
    ]

    packed = pack_prefix_conditioning(
        token_embeddings=token_embeddings,
        token_attention_mask=token_attention_mask,
        prefix_embeddings=prefix_embeddings,
        max_total_positions=8,
    )

    assert packed.inputs_embeds.shape == (2, 5, 2)
    assert packed.token_positions.tolist() == [
        [1, 2, -1],
        [2, 3, 4],
    ]
    assert packed.attention_mask.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert packed.position_ids.tolist() == [
        [0, 1, 2, 0, 0],
        [0, 1, 2, 3, 4],
    ]


def test_pack_prefix_conditioning_fails_fast_on_overlength():
    token_embeddings = torch.zeros((1, 4, 2))
    token_attention_mask = torch.tensor([[True, True, True, True]])
    prefix_embeddings = [torch.zeros((3, 2))]

    with pytest.raises(ValueError, match='Sample exceeds max_total_positions'):
        pack_prefix_conditioning(
            token_embeddings=token_embeddings,
            token_attention_mask=token_attention_mask,
            prefix_embeddings=prefix_embeddings,
            max_total_positions=6,
        )


def test_extract_molecule_logits_uses_token_positions():
    logits = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    token_positions = torch.tensor(
        [
            [1, 2, -1],
            [2, 3, 4],
        ]
    )
    extracted = extract_molecule_logits(logits, token_positions)
    assert extracted.shape == (2, 3, 3)
    assert torch.equal(extracted[0, 0], logits[0, 1])
    assert torch.equal(extracted[0, 1], logits[0, 2])
    assert torch.equal(extracted[1, 0], logits[1, 2])
    assert torch.equal(extracted[1, 2], logits[1, 4])
    assert torch.equal(extracted[0, 2], torch.zeros(3))


def test_pad_prefix_embeddings_returns_mask():
    padded, mask = pad_prefix_embeddings(
        [
            torch.ones((1, 2)),
            torch.full((3, 2), 2.0),
        ]
    )
    assert padded.shape == (2, 3, 2)
    assert mask.tolist() == [
        [True, False, False],
        [True, True, True],
    ]
