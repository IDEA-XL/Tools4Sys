#!/usr/bin/env python3
"""Merge de novo sweep summary JSON files without recomputing existing rows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _read_json(path: Path) -> list[dict]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f'Expected list payload in {path}')
    return payload


def _write_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def _key(row: dict) -> tuple[str, str, float]:
    sweep_value = float(row['sweep_value'])
    if not math.isfinite(sweep_value):
        raise ValueError(f'Non-finite sweep_value in row: {row}')
    return (str(row['experiment']), str(row['sweep_axis']), round(sweep_value, 8))


def _build_markdown(rows: list[dict], output_json_path: Path, output_rows_path: Path | None) -> str:
    lines = [
        '# GenMol De Novo Sweep Results',
        '',
        f'- `summary_json`: `{output_json_path}`',
    ]
    if output_rows_path is not None:
        lines.append(f'- `raw_rows_jsonl`: `{output_rows_path}`')
    lines.extend(
        [
            '- `diversity`: internal diversity computed over generated molecules for the sweep point.',
            '- `qed_mean`, `sa_score_mean`, and `soft_reward_mean`: means over the valid generated molecules for the sweep point.',
            '',
            '| Experiment | Sweep | Value | Diversity | QED | SA Score | Soft Reward | Valid Fraction | Invalid Fraction | Checkpoint |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
        ]
    )
    for row in rows:
        checkpoint_path = str(row['checkpoint_path'])
        lines.append(
            '| ' + ' | '.join(
                [
                    str(row.get('display_name') or row['experiment']),
                    str(row['sweep_axis']),
                    f"{float(row['sweep_value']):.6f}",
                    f"{float(row['diversity']):.6f}",
                    f"{float(row['qed_mean']):.6f}",
                    f"{float(row['sa_score_mean']):.6f}",
                    f"{float(row['soft_reward_mean']):.6f}",
                    f"{float(row['valid_fraction']):.6f}",
                    f"{float(row['invalid_fraction']):.6f}",
                    checkpoint_path,
                ]
            )
            + ' |'
        )
    lines.append('')
    return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', action='append', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-rows-jsonl', default=None)
    parser.add_argument('--output-markdown', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: dict[tuple[str, str, float], dict] = {}
    for input_json in args.input_json:
        path = Path(input_json)
        rows = _read_json(path)
        for row in rows:
            key = _key(row)
            if key in merged:
                raise ValueError(f'Duplicate row for key={key} while merging {path}')
            merged[key] = row
    if not merged:
        raise ValueError('No rows remain after merging')

    rows = sorted(
        merged.values(),
        key=lambda row: (str(row['sweep_axis']), float(row['sweep_value']), str(row['experiment'])),
    )

    output_json_path = Path(args.output_json)
    _write_json(output_json_path, rows)

    output_rows_path = Path(args.output_rows_jsonl) if args.output_rows_jsonl else None
    if output_rows_path is not None:
        _write_jsonl(output_rows_path, rows)

    if args.output_markdown:
        output_markdown_path = Path(args.output_markdown)
        output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        output_markdown_path.write_text(_build_markdown(rows, output_json_path, output_rows_path))

    print(json.dumps(
        {
            'output_json': str(output_json_path),
            'output_rows_jsonl': str(output_rows_path) if output_rows_path is not None else None,
            'output_markdown': args.output_markdown,
            'num_rows': len(rows),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == '__main__':
    main()
