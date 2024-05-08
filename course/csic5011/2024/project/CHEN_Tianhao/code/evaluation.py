"""evaluation of contact map guidance"""

import torch
from chroma import Chroma, Protein
from contact_map_conditioner import ContactMapConditioner
import subprocess, shlex
from tqdm.auto import tqdm
import numpy as np
import json
import os


TM_ALIGN = './TMalign/TMalign'

DEFAULT_SAMPLE_KWARGS = {
    'langevin_factor': 8,
    'inverse_temperature': 8,
    'sde_func': "langevin",
    'steps': 500
}


def run_cmd(cmd: str):
    return subprocess.check_output(shlex.split(cmd))


def distance(X, eps=1e-6):
    dX = X.unsqueeze(2) - X.unsqueeze(1)
    D = torch.sqrt((dX**2).sum(-1) + eps)
    return D


def extract_metrics(tm_output):
    """
    extract RMSD, TM-score, Seq-Id from TMalign output
    """
    # locate metrics
    rmsd_start = tm_output.find(b"RMSD")
    rmsd_end = tm_output.find(b",", rmsd_start)
    tm1_start = tm_output.find(b"TM-score")
    tm1_end = tm_output.find(b"(", tm1_start)
    seq_id_start = tm_output.find(b'Seq_ID=n_identical/n_aligned=', rmsd_end) + len('Seq_ID=n_identical/n_aligned=')
    seq_id_end = tm1_start
    tm2_start = tm_output.find(b"TM-score", tm1_end)
    tm2_end = tm_output.find(b"(", tm2_start)
    # extract metrics
    rmsd = float(tm_output[rmsd_start+5:rmsd_end])
    seq_id = float(tm_output[seq_id_start:seq_id_end])
    tm_score = float(tm_output[tm1_start+9:tm1_end]), float(tm_output[tm2_start+9:tm2_end])
    return {'rmsd': rmsd, 'tm_score': tm_score, 'seq_id': seq_id}


def get_metrics(protein_1: Protein, protein_2: Protein):
    X_1, _, _ = protein_1.to_XCS()
    X_2, _, _ = protein_2.to_XCS()
    assert X_1.shape == X_2.shape
    # contact map diff
    D_1 = distance(X_1)
    D_2 = distance(X_2)
    contact_map_err = (D_1 - D_2).abs().mean().item()
    # RMSD, TM-score, Seq_id
    protein_1.to_PDB('pdb1.pdb')
    protein_2.to_PDB('pdb2.pdb')
    cmd = f"{TM_ALIGN} pdb1.pdb pdb2.pdb"
    metrics = extract_metrics(run_cmd(cmd))
    metrics['contact_map_err'] = contact_map_err
    return metrics


def eval_func(protein: Protein, num_eval=10, **kwargs):
    """
    protein: target protein, whose contact map will be used for recovring structure
    """
    # process args
    global chroma
    weight = kwargs.pop('weight', 0.01)
    eps = kwargs.pop('eps', 1e-6)
    ca_only = kwargs.pop('ca_only', False)
    if len(kwargs) == 0:
        kwargs = DEFAULT_SAMPLE_KWARGS

    X, C, S = protein.to_XCS()
    num_residue = X.shape[1]

    # conditioner
    D_inter = distance(X)
    if ca_only:
        D_inter = D_inter[..., 1:2]
    noise_schedule = chroma.backbone_network.noise_schedule
    contact_conditioner = ContactMapConditioner(
        D_inter,
        noise_schedule,
        weight=weight,
        eps=eps,
        ca_only=ca_only
    )

    # do generation
    metrics_cond_list = []
    metrics_rand_list = []
    for _ in tqdm(range(num_eval)):
        # contact map conditioned protein
        protein_cond = chroma.sample(
            chain_lengths=[num_residue],
            conditioner=contact_conditioner,
            **kwargs
        )
        # random generation
        protein_rand = chroma.sample(
            chain_lengths=[num_residue],
            **kwargs
        )
        metrics_cond = get_metrics(protein, protein_cond)
        metrics_rand = get_metrics(protein, protein_rand)
        metrics_cond_list.append(metrics_cond)
        metrics_rand_list.append(metrics_rand)
    return metrics_cond_list, metrics_rand_list


def get_stat(metrics_list):
    metrics_avg = {
        'rmsd': np.mean([metrics['rmsd'] for metrics in metrics_list]),
        'tm_score': (
            np.mean([metrics['tm_score'][0] for metrics in metrics_list]),
            np.mean([metrics['tm_score'][1] for metrics in metrics_list])
        ),
        'seq_id': np.mean([metrics['seq_id'] for metrics in metrics_list]),
        'contact_map_err': np.mean([metrics['contact_map_err'] for metrics in metrics_list])
    }
    return metrics_avg


def read_jsonl(fname):
    list_obj = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            list_obj.append(json.loads(line))
    return list_obj


def save_jsonl(list_obj, fname):
    with open(fname, 'w') as f:
        for entry in list_obj:
            f.writelines(json.dumps(entry)+'\n')


def plot_results(eval_results, output_dir='./fig'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    os.makedirs(output_dir, exist_ok=True)

    # collect data
    num_residue_list = []
    metrics_by_num_residue = []
    all_metrics_list_cond = {'rmsd': [], 'tm_score': [], 'seq_id': [], 'contact_map_err': []}
    all_metrics_list_rand = {'rmsd': [], 'tm_score': [], 'seq_id': [], 'contact_map_err': []}
    for entry in eval_results:
        metrics_cond = {
            'rmsd': [met['rmsd'] for met in entry['metrics_cond_list']],
            'tm_score': [met['tm_score'][0] for met in entry['metrics_cond_list']],
            'seq_id': [met['seq_id'] for met in entry['metrics_cond_list']],
            'contact_map_err': [met['contact_map_err'] for met in entry['metrics_cond_list']]
        }
        metrics_rand = {
            'rmsd': [met['rmsd'] for met in entry['metrics_rand_list']],
            'tm_score': [met['tm_score'][0] for met in entry['metrics_rand_list']],
            'seq_id': [met['seq_id'] for met in entry['metrics_rand_list']],
            'contact_map_err': [met['contact_map_err'] for met in entry['metrics_rand_list']]
        }
        metrics_by_num_residue.append({'cond': metrics_cond, 'rand': metrics_rand})
        for key in metrics_cond.keys():
            all_metrics_list_cond[key].extend(metrics_cond[key])
            all_metrics_list_rand[key].extend(metrics_rand[key])
        num_residue_list.extend([entry['num_residue']]*len(metrics_cond['rmsd']))

    all_metrics_list = {key: all_metrics_list_cond[key] + all_metrics_list_rand[key] for key in all_metrics_list_cond.keys()}
    all_metrics_list['type'] = ['cond'] * len(all_metrics_list_cond['rmsd']) + ['rand'] * len(all_metrics_list_rand['rmsd'])
    all_metrics_list['num_residue'] = num_residue_list*2
    df = pd.DataFrame(all_metrics_list)

    # box plot of metric ~ num_residue, for metric in [RMSD, TM-score, Seq-Id, Contact map MAE]
    for key in all_metrics_list_cond.keys():
        plt.figure()
        sns.boxplot(y=key, x='num_residue', data=df, hue='type')
        plt.title(key)
        plt.savefig(os.path.join(output_dir, f"{key}.png"))

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_target_protein",
        type=int,
        default=5
    )
    parser.add_argument(
        "-m",
        "--num_sample_per_target",
        type=int,
        default=1
    )
    parser.add_argument(
        "--ca_only",
        action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chroma = Chroma(device=device)

    # target protein length from 10 to 200
    target_num_residue_list = (np.arange(20) * 10 + 10).tolist()
    target_proteins_list = [chroma.sample(
        samples=args.num_target_protein,
        chain_lengths=[num_residue],
        **DEFAULT_SAMPLE_KWARGS
    ) for num_residue in target_num_residue_list]

    for idx in range(len(target_proteins_list)):
        if not isinstance(target_proteins_list[idx], list):
            target_proteins_list[idx] = [target_proteins_list[idx]]

    eval_results = []

    for target_proteins, target_num_residue in zip(target_proteins_list, target_num_residue_list):
        print(f"Evaluating protein of length {target_num_residue}...")
        metrics_cond_list = []
        metrics_rand_list = []
        for target_protein in target_proteins:
            metrics_cond_list_per_target, metrics_rand_list_per_target = eval_func(
                target_protein,
                args.num_sample_per_target,
                ca_only=args.ca_only
            )
            metrics_cond_list.extend(metrics_cond_list_per_target)
            metrics_rand_list.extend(metrics_rand_list_per_target)
        eval_results_entry = {
            'num_residue': target_num_residue,
            'metrics_cond_list': metrics_cond_list,
            'metrics_rand_list': metrics_rand_list,
            'metrics_cond_avg': get_stat(metrics_cond_list),
            'metrics_rand_avg': get_stat(metrics_rand_list)
        }
        eval_results.append(eval_results_entry)

    save_jsonl(eval_results, 'eval_results.json')
    plot_results(eval_results, './fig')

    