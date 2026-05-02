import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from scripts.aggregate_mmgenmol_sweep_results import (
    _build_markdown,
    _build_plot_jobs,
    _plot_metric,
)


def _read_json(path):
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f'Expected list payload in {path}')
    return payload


def _write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def _parse_allowed_values(specs):
    allowed = {}
    for spec in specs:
        if ':' not in spec:
            raise ValueError(f'Invalid --allowed-values entry: {spec}')
        sweep_type, values_text = spec.split(':', 1)
        sweep_type = sweep_type.strip()
        if not sweep_type:
            raise ValueError(f'Empty sweep type in --allowed-values entry: {spec}')
        values = []
        for part in values_text.split(','):
            part = part.strip()
            if not part:
                continue
            values.append(float(part))
        if not values:
            raise ValueError(f'No values provided in --allowed-values entry: {spec}')
        allowed[sweep_type] = set(values)
    return allowed


def _float_key(value):
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f'Non-finite sweep value: {value}')
    return round(numeric, 8)


def _merge_rows(input_paths, allowed_values):
    merged = {}
    for input_path in input_paths:
        rows = _read_json(input_path)
        for row in rows:
            sweep_type = row['sweep_type']
            sweep_value = float(row['sweep_value'])
            if allowed_values and sweep_type in allowed_values and sweep_value not in allowed_values[sweep_type]:
                continue
            key = (row['model_name'], sweep_type, _float_key(sweep_value))
            if key in merged:
                raise ValueError(f'Duplicate row for key={key} while merging {input_path}')
            merged[key] = row
    if not merged:
        raise ValueError('No rows remain after merging/filtering')
    return sorted(
        merged.values(),
        key=lambda row: (row['sweep_type'], float(row['sweep_value']), row['model_name']),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--plot-name-prefix', required=True)
    parser.add_argument('--plot-title-prefix', required=True)
    parser.add_argument('--allowed-values', action='append', default=[])
    args = parser.parse_args()

    allowed_values = _parse_allowed_values(args.allowed_values)
    rows = _merge_rows(args.input_json, allowed_values)

    os.makedirs(args.output_dir, exist_ok=True)
    output_json_path = os.path.join(args.output_dir, f'{args.output_prefix}.json')
    output_rows_path = os.path.join(args.output_dir, f'{args.output_prefix}.rows.jsonl')
    output_markdown_path = os.path.join(args.output_dir, f'{args.output_prefix}.md')
    _write_json(output_json_path, rows)
    _write_jsonl(output_rows_path, rows)

    plot_paths = []
    sweep_types = sorted({row['sweep_type'] for row in rows})
    plot_suffix = args.output_prefix.rsplit("_", 1)[-1]
    titled_plot_paths = []
    for (
        title,
        plot_path,
        sweep_type,
        metric_key,
        metric_label,
        title_prefix,
        model_names,
        with_unidock_baseline,
    ) in _build_plot_jobs(
        rows,
        output_dir=args.output_dir,
        plot_name_prefix=args.plot_name_prefix,
        plot_title_prefix=args.plot_title_prefix,
        plot_suffix=plot_suffix,
        sweep_types=sweep_types,
    ):
        _plot_metric(
            rows,
            sweep_type,
            metric_key,
            metric_label,
            plot_path,
            plot_title_prefix=title_prefix,
            model_names=model_names,
            with_unidock_baseline=with_unidock_baseline,
        )
        plot_paths.append(plot_path)
        titled_plot_paths.append((title, plot_path))

    markdown = _build_markdown(rows, titled_plot_paths, output_json_path, output_rows_path)
    with open(output_markdown_path, 'w') as handle:
        handle.write(markdown)

    print(json.dumps({
        'output_json_path': output_json_path,
        'output_rows_path': output_rows_path,
        'output_markdown_path': output_markdown_path,
        'plot_paths': plot_paths,
        'num_rows': len(rows),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
