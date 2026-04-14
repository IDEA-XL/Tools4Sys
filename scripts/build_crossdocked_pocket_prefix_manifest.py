import argparse
import json
import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.crossdocked import build_crossdocked_manifest, save_crossdocked_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crossdocked_lmdb_path', required=True)
    parser.add_argument('--crossdocked_split_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--max_total_positions', type=int, default=256)
    args = parser.parse_args()

    entries, stats = build_crossdocked_manifest(
        lmdb_path=args.crossdocked_lmdb_path,
        split_path=args.crossdocked_split_path,
        max_total_positions=args.max_total_positions,
    )
    stats_path = save_crossdocked_manifest(entries=entries, stats=stats, output_path=args.output_path)
    print(json.dumps({'manifest_path': args.output_path, 'stats_path': stats_path, **stats.__dict__}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
