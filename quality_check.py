import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_data():
    print("Loading lerobot/pusht dataset...")
    ds = load_dataset("lerobot/pusht", split="train")
    df = ds.to_pandas()
    print(f"Loaded {len(df)} frames")
    return df


def check_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})


def check_episode_continuity(df):
    issues = []
    for ep_idx in df['episode_index'].unique():
        ep = df[df['episode_index'] == ep_idx].sort_values('frame_index')
        frame_diffs = ep['frame_index'].diff().dropna()
        gaps = frame_diffs[frame_diffs != 1]
        if len(gaps) > 0:
            issues.append({
                'episode_index': int(ep_idx),
                'gap_count': len(gaps)
            })
    return issues


def check_timestamp_consistency(df):
    issues = []
    for ep_idx in df['episode_index'].unique():
        ep = df[df['episode_index'] == ep_idx].sort_values('frame_index')
        ts_diffs = ep['timestamp'].diff().dropna()
        non_monotonic = ts_diffs[ts_diffs < 0]
        if len(non_monotonic) > 0:
            issues.append({
                'episode_index': int(ep_idx),
                'non_monotonic_count': int(len(non_monotonic)),
                'min_diff': round(float(non_monotonic.min()), 6)
            })
    return issues


def check_action_anomalies(df):
    actions = np.array(df['action'].tolist())
    results = []
    for dim in range(actions.shape[1]):
        col = actions[:, dim]
        mean, std = col.mean(), col.std()
        outlier_count = int(np.sum(np.abs(col - mean) > 3 * std))
        results.append({
            'action_dim': dim,
            'mean': round(float(mean), 4),
            'std': round(float(std), 4),
            'outlier_count': outlier_count,
            'outlier_pct': round(outlier_count / len(col) * 100, 2)
        })
    return results


def check_reward_distribution(df):
    rewards = df['next.reward']
    return {
        'min': round(float(rewards.min()), 4),
        'max': round(float(rewards.max()), 4),
        'mean': round(float(rewards.mean()), 4),
        'std': round(float(rewards.std()), 4),
        'zero_reward_pct': round(float((rewards == 0).sum() / len(rewards) * 100), 2),
        'positive_reward_pct': round(float((rewards > 0).sum() / len(rewards) * 100), 2)
    }


def check_episode_length(df):
    ep_lengths = df.groupby('episode_index')['frame_index'].count()
    mean_len = ep_lengths.mean()
    std_len = ep_lengths.std()
    return {
        'total_episodes': len(ep_lengths),
        'mean_length': round(float(mean_len), 1),
        'std_length': round(float(std_len), 1),
        'min_length': int(ep_lengths.min()),
        'max_length': int(ep_lengths.max()),
        'abnormally_short': int(len(ep_lengths[ep_lengths < mean_len - 2*std_len])),
        'abnormally_long': int(len(ep_lengths[ep_lengths > mean_len + 2*std_len]))
    }


def visualize(df, output_dir="output"):
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LeRobot PushT Dataset Quality Report', fontsize=14)

    # Episode长度分布
    ep_lengths = df.groupby('episode_index')['frame_index'].count()
    axes[0, 0].hist(ep_lengths, bins=30, color='steelblue', edgecolor='white')
    axes[0, 0].set_title('Episode Length Distribution')
    axes[0, 0].set_xlabel('Frames per Episode')
    axes[0, 0].set_ylabel('Count')

    # Reward分布
    axes[0, 1].hist(df['next.reward'], bins=50, color='coral', edgecolor='white')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].set_xlabel('Reward Value')
    axes[0, 1].set_ylabel('Count')

    # Action异常值比例
    actions = np.array(df['action'].tolist())
    outlier_pcts = []
    for dim in range(actions.shape[1]):
        col = actions[:, dim]
        mean, std = col.mean(), col.std()
        outlier_pcts.append(
            int(np.sum(np.abs(col - mean) > 3 * std)) / len(col) * 100
        )
    axes[1, 0].bar(range(len(outlier_pcts)), outlier_pcts, color='mediumpurple')
    axes[1, 0].set_title('Action Outlier Rate by Dimension (>3σ)')
    axes[1, 0].set_xlabel('Action Dimension')
    axes[1, 0].set_ylabel('Outlier %')
    axes[1, 0].set_ylim(bottom=0)

    # 时间戳间隔分布（episode内部，不跨episode）
    ts_diffs_list = []
    for ep_idx in df[df['episode_index'] < 20]['episode_index'].unique():
        ep = df[df['episode_index'] == ep_idx].sort_values('frame_index')
        diffs = ep['timestamp'].diff().dropna()
        ts_diffs_list.append(diffs)
    ts_diffs = pd.concat(ts_diffs_list)
    axes[1, 1].hist(ts_diffs, bins=50, color='mediumseagreen', edgecolor='white')
    axes[1, 1].set_title('Timestamp Interval Distribution (first 20 eps)')
    axes[1, 1].set_xlabel('Δt (seconds)')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    path = f'{output_dir}/quality_dashboard.png'
    plt.savefig(path, dpi=150)
    print(f"Dashboard saved to {path}")
    plt.close()


def run_all_checks(df):
    print("\n===== Running Quality Checks =====\n")
    results = {}

    print("[1/5] Checking missing values...")
    results['missing_values'] = check_missing_values(df).to_dict()

    print("[2/5] Checking episode continuity...")
    continuity_issues = check_episode_continuity(df)
    results['continuity_issues'] = {
        'issue_count': len(continuity_issues),
        'details': continuity_issues[:5]
    }

    print("[3/5] Checking timestamp consistency...")
    ts_issues = check_timestamp_consistency(df)
    results['timestamp_issues'] = {
        'issue_count': len(ts_issues),
        'details': ts_issues[:5]
    }

    print("[4/5] Checking action anomalies...")
    results['action_anomalies'] = check_action_anomalies(df)

    print("[5/5] Checking reward & episode stats...")
    results['reward_distribution'] = check_reward_distribution(df)
    results['episode_stats'] = check_episode_length(df)

    return results


if __name__ == "__main__":
    df = load_data()
    results = run_all_checks(df)

    Path("output").mkdir(exist_ok=True)
    with open("output/quality_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nReport saved to output/quality_report.json")

    print("\nGenerating visualizations...")
    visualize(df)

    print("\n===== Summary =====")
    print(f"Total frames: {len(df)}")
    print(f"Total episodes: {results['episode_stats']['total_episodes']}")
    print(f"Continuity issues: {results['continuity_issues']['issue_count']}")
    print(f"Timestamp issues: {results['timestamp_issues']['issue_count']}")
    print(f"Reward stats: {results['reward_distribution']}")