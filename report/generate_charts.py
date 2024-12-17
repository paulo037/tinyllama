import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
current_directory = os.path.dirname(os.path.abspath(__file__))


def load_csv(file_name):
    return pd.read_csv(file_name)


def plot_metric(metric, name, data_1, data_123, scale='linear', window_size=50):
    tot123 = 81e6
    data_123['Total Tokens'] = np.linspace(0, tot123, len(data_123))
    data_1['Total Tokens'] = data_123['Total Tokens'][:len(data_1)]


    data_1['smoothed_value'] = data_1['value'].rolling(window=window_size, min_periods=1).mean()
    data_123['smoothed_value'] = data_123['value'].rolling(window=window_size, min_periods=1).mean()

    sns.set_palette("tab10")
    plt.figure(figsize=(12, 6))

    sns.lineplot(x=data_123['Total Tokens'], y=data_123['value'], color='C1', alpha=0.3, linewidth=1, label=None)
    sns.lineplot(x=data_1['Total Tokens'], y=data_1['value'], color='C0', alpha=0.3, linewidth=1, label=None)

    sns.lineplot(x=data_123['Total Tokens'], y=data_123['smoothed_value'], color='C1', label='TinyLlama Math&Code', linewidth=2)
    sns.lineplot(x=data_1['Total Tokens'], y=data_1['smoothed_value'], color='C0', label='TinyLlama', linewidth=2)

    plt.yscale(scale)
    plt.xlabel('Total Tokens', fontsize=12)
    plt.ylabel(name, fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(f'{current_directory}/charts/{metric}.svg')





metrics = ['train_loss', 'train_perplexity', 'validation_cross_entropy', 'validation_perplexity']
names = ['Training Loss',  'Training Perplexity', 'Validation Loss', 'Validation Perplexity']
scales = ['linear', 'log', 'linear', 'log']
window_sizes = [50, 50, 1, 1]
dfs = []

for i in range(1, 6):
    df = load_csv(f'{current_directory}/metrics/metric{i}.csv')
    dfs.append(df)
        
for metric, name, scale, window_size in zip(metrics, names, scales, window_sizes):
    dfs_copy = [df.query("key == @metric").copy() for df in dfs]
    
    df1 = pd.concat([dfs_copy[0], dfs_copy[-2], dfs_copy[-1]], ignore_index=True)
    df123 = pd.concat(dfs_copy[:-2], ignore_index=True)
    plot_metric(metric, name, df1, df123, scale, window_size)