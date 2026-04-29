import math
import random
from dataclasses import asdict

from genmol.mm.docking import summarize_docking_records
from genmol.mm.reward import MolecularReward


def order_preserving_unique(items):
    return list(dict.fromkeys(items))


def nanmean(values):
    filtered = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not filtered:
        return float('nan')
    return float(sum(filtered) / len(filtered))


class OfficialMoleculeMetricSuite:
    def __init__(self, qed_oracle=None, sa_oracle=None, diversity_evaluator=None):
        if qed_oracle is None or sa_oracle is None or diversity_evaluator is None:
            from tdc import Evaluator, Oracle

            self.qed_oracle = Oracle('qed') if qed_oracle is None else qed_oracle
            self.sa_oracle = Oracle('sa') if sa_oracle is None else sa_oracle
            self.diversity_evaluator = Evaluator('diversity') if diversity_evaluator is None else diversity_evaluator
        else:
            self.qed_oracle = qed_oracle
            self.sa_oracle = sa_oracle
            self.diversity_evaluator = diversity_evaluator

    def summarize(self, smiles_list):
        total_count = len(smiles_list)
        if total_count <= 0:
            raise ValueError('smiles_list must be non-empty')

        valid_smiles = [smiles for smiles in smiles_list if smiles is not None]
        unique_valid_smiles = order_preserving_unique(valid_smiles)

        validity = float(len(valid_smiles) / total_count)
        uniqueness = 0.0 if not valid_smiles else float(len(unique_valid_smiles) / len(valid_smiles))

        if unique_valid_smiles:
            qeds = [float(value) for value in self.qed_oracle(unique_valid_smiles)]
            sas = [float(value) for value in self.sa_oracle(unique_valid_smiles)]
            quality = float(
                sum(1 for qed_value, sa_value in zip(qeds, sas) if qed_value >= 0.6 and sa_value <= 4.0) / total_count
            )
            diversity = 0.0 if len(unique_valid_smiles) == 1 else float(self.diversity_evaluator(unique_valid_smiles))
            qed_mean = nanmean(qeds)
            sa_mean = nanmean(sas)
        else:
            quality = 0.0
            diversity = 0.0
            qed_mean = float('nan')
            sa_mean = float('nan')

        return {
            'official_validity': validity,
            'official_uniqueness': uniqueness,
            'official_quality': quality,
            'official_diversity': diversity,
            'official_qed_mean': qed_mean,
            'official_sa_mean': sa_mean,
            'num_valid': len(valid_smiles),
            'num_unique_valid': len(unique_valid_smiles),
        }


def summarize_reward_records(records):
    if not records:
        raise ValueError('records must be non-empty')
    rewards = [float(record.reward) for record in records]
    valid_flags = [1.0 if record.is_valid else 0.0 for record in records]
    alert_flags = [1.0 if record.alert_hit else 0.0 for record in records]
    return {
        'reward_mean': float(sum(rewards) / len(rewards)),
        'reward_valid_fraction': float(sum(valid_flags) / len(valid_flags)),
        'alert_hit_fraction': float(sum(alert_flags) / len(alert_flags)),
        'qed_mean': nanmean([record.qed for record in records]),
        'sa_mean': nanmean([record.sa for record in records]),
        'sa_score_mean': nanmean([record.sa_score for record in records]),
        'drugclip_score_mean': nanmean([record.drugclip_score for record in records]),
        'unidock_score_mean': nanmean([record.unidock_score for record in records]),
        'soft_reward_mean': nanmean([record.soft_reward for record in records]),
    }


def select_manifest_entries(entries, num_pockets, seed):
    if not entries:
        raise ValueError('entries must be non-empty')
    if num_pockets is None:
        return list(entries)
    if num_pockets <= 0:
        raise ValueError(f'num_pockets must be positive, got {num_pockets}')
    if num_pockets > len(entries):
        raise ValueError(f'num_pockets exceeds available entries: {num_pockets} vs {len(entries)}')
    rng = random.Random(seed)
    return rng.sample(list(entries), num_pockets)


def build_rows(entries, specs, rollout, reward_records, docking_records_by_mode=None):
    if not (len(entries) == len(specs) == len(rollout.safe_strings) == len(rollout.smiles) == len(reward_records)):
        raise ValueError(
            'Row payload lengths must match: '
            f'entries={len(entries)} specs={len(specs)} safe={len(rollout.safe_strings)} '
            f'smiles={len(rollout.smiles)} records={len(reward_records)}'
        )
    if docking_records_by_mode is not None:
        for mode, docking_records in docking_records_by_mode.items():
            if len(docking_records) != len(entries):
                raise ValueError(
                    f'docking_records length mismatch for mode {mode!r}: {len(docking_records)} vs {len(entries)}'
                )

    rows = []
    for row_idx, (entry, spec, safe_string, smiles, record) in enumerate(
        zip(entries, specs, rollout.safe_strings, rollout.smiles, reward_records)
    ):
        row = {
            'source_index': int(entry['source_index']),
            'split': entry['split'],
            'protein_filename': entry.get('protein_filename'),
            'ligand_filename': entry.get('ligand_filename'),
            'residue_count': int(entry['residue_count']),
            'ligand_smiles': entry.get('ligand_smiles'),
            'spec': asdict(spec),
            'safe': safe_string,
            'smiles': smiles,
            'reward': float(record.reward),
            'is_valid': bool(record.is_valid),
            'alert_hit': bool(record.alert_hit),
            'qed': record.qed,
            'sa': record.sa,
            'sa_score': record.sa_score,
            'drugclip_score': record.drugclip_score,
            'unidock_score': record.unidock_score,
            'soft_reward': record.soft_reward,
        }
        if docking_records_by_mode is not None:
            for mode, docking_records in docking_records_by_mode.items():
                docking_record = docking_records[row_idx]
                prefix = f'{mode}_'
                row.update(
                    {
                        f'{prefix}is_success': bool(docking_record.is_success),
                        f'{prefix}error': docking_record.error,
                        f'{prefix}receptor_pdb_path': docking_record.receptor_pdb_path,
                        f'{prefix}receptor_pdbqt_path': docking_record.receptor_pdbqt_path,
                        f'{prefix}ligand_sdf_path': docking_record.ligand_sdf_path,
                        f'{prefix}ligand_pdbqt_path': docking_record.ligand_pdbqt_path,
                        f'{prefix}center_x': float(docking_record.center_x),
                        f'{prefix}center_y': float(docking_record.center_y),
                        f'{prefix}center_z': float(docking_record.center_z),
                        f'{prefix}size_x': float(docking_record.size_x),
                        f'{prefix}size_y': float(docking_record.size_y),
                        f'{prefix}size_z': float(docking_record.size_z),
                        f'{prefix}score_only_affinity': docking_record.score_only_affinity,
                        f'{prefix}minimize_affinity': docking_record.minimize_affinity,
                        f'{prefix}dock_affinity': docking_record.dock_affinity,
                    }
                )
        rows.append(row)
    return rows


class PocketPrefixEvaluationKernel:
    def __init__(self, metric_suite=None, reward_model=None, docking_model=None):
        self.metric_suite = OfficialMoleculeMetricSuite() if metric_suite is None else metric_suite
        self.reward_model = MolecularReward() if reward_model is None else reward_model
        self.docking_model = docking_model

    def close(self):
        self.reward_model.close()
        if self.docking_model is not None:
            self.docking_model.close()

    def summarize(self, smiles_list, entries=None):
        official_metrics = self.metric_suite.summarize(smiles_list)
        reward_records = self.reward_model.score(smiles_list, pocket_entries=entries)
        reward_metrics = summarize_reward_records(reward_records)
        docking_metrics = None
        docking_records = None
        if self.docking_model is not None:
            if entries is None:
                raise ValueError('entries must be provided when docking_model is enabled')
            docking_records = self.docking_model.score(entries=entries, smiles_list=smiles_list)
            docking_metrics = summarize_docking_records(docking_records)
        return official_metrics, reward_metrics, reward_records, docking_metrics, docking_records
