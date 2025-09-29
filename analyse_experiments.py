import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set up for saving plots instead of displaying
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_and_explore_data():
    """Load the experiment results and provide basic exploration"""
    df = pd.read_csv('/home/na658/multi-task2/experiment_results/experiment_results.csv')
    
    print("=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\n=== Data Types ===")
    print(df.dtypes)
    print("\n=== First few rows ===")
    print(df.head())
    print("\n=== Basic Statistics ===")
    print(df.describe())
    print("\n=== Unique values in categorical columns ===")
    categorical_cols = ['seed', 'active_dim_1', 'active_dim_2', 'overlap', 'linear_readout', 
                       'one_task', 'load_model', 'init_method', 'lmda', 'c']
    for col in categorical_cols:
        if col in df.columns:
            print(f"{col}: {df[col].unique()}")
    
    return df

def plot_training_data_scaling(df):
    """Plot relationship between training data sizes"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot of n_train1 vs n_train2
    axes[0].scatter(df['n_train1'], df['n_train2'], alpha=0.6)
    axes[0].set_xlabel('n_train1')
    axes[0].set_ylabel('n_train2')
    axes[0].set_title('Training Data Size Relationship')
    axes[0].grid(True, alpha=0.3)
    
    # Distribution of training sizes
    axes[1].hist(df['n_train1'], bins=20, alpha=0.7, label='n_train1')
    axes[1].hist(df['n_train2'], bins=20, alpha=0.7, label='n_train2')
    axes[1].set_xlabel('Training Size')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Training Data Sizes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_data_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_analysis(df):
    """Analyze training and validation losses"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Final training loss vs validation loss
    axes[0,0].scatter(df['final_train_loss'], df['final_val_loss'], alpha=0.6)
    axes[0,0].set_xlabel('Final Training Loss')
    axes[0,0].set_ylabel('Final Validation Loss')
    axes[0,0].set_title('Training vs Validation Loss')
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    
    # Loss vs training data size
    axes[0,1].scatter(df['n_train2'], df['final_val_loss'], alpha=0.6)
    axes[0,1].set_xlabel('n_train2')
    axes[0,1].set_ylabel('Final Validation Loss')
    axes[0,1].set_title('Validation Loss vs Training Data Size')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # Loss vs active dimensions
    for active_dim in df['active_dim_2'].unique():
        subset = df[df['active_dim_2'] == active_dim]
        axes[1,0].scatter(subset['n_train2'], subset['final_val_loss'], 
                         label=f'active_dim_2={active_dim}', alpha=0.6)
    axes[1,0].set_xlabel('n_train2')
    axes[1,0].set_ylabel('Final Validation Loss')
    axes[1,0].set_title('Validation Loss by Active Dimensions')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Loss vs lambda parameter
    for lmda in df['lmda'].unique():
        subset = df[df['lmda'] == lmda]
        axes[1,1].scatter(subset['n_train2'], subset['final_val_loss'], 
                         label=f'lmda={lmda}', alpha=0.6)
    axes[1,1].set_xlabel('n_train2')
    axes[1,1].set_ylabel('Final Validation Loss')
    axes[1,1].set_title('Validation Loss by Lambda Parameter')
    axes[1,1].set_yscale('log')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_analysis(df):
    """Analyze convergence behavior"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Final epoch vs training data size
    axes[0,0].scatter(df['n_train2'], df['final_epoch'], alpha=0.6)
    axes[0,0].set_xlabel('n_train2')
    axes[0,0].set_ylabel('Final Epoch')
    axes[0,0].set_title('Convergence Epoch vs Training Data Size')
    axes[0,0].grid(True, alpha=0.3)
    
    # Final epoch vs validation loss
    axes[0,1].scatter(df['final_epoch'], df['final_val_loss'], alpha=0.6)
    axes[0,1].set_xlabel('Final Epoch')
    axes[0,1].set_ylabel('Final Validation Loss')
    axes[0,1].set_title('Convergence Epoch vs Final Validation Loss')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # Convergence by active dimensions
    for active_dim in df['active_dim_2'].unique():
        subset = df[df['active_dim_2'] == active_dim]
        axes[1,0].scatter(subset['n_train2'], subset['final_epoch'], 
                         label=f'active_dim_2={active_dim}', alpha=0.6)
    axes[1,0].set_xlabel('n_train2')
    axes[1,0].set_ylabel('Final Epoch')
    axes[1,0].set_title('Convergence by Active Dimensions')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Convergence by lambda
    for lmda in df['lmda'].unique():
        subset = df[df['lmda'] == lmda]
        axes[1,1].scatter(subset['n_train2'], subset['final_epoch'], 
                         label=f'lmda={lmda}', alpha=0.6)
    axes[1,1].set_xlabel('n_train2')
    axes[1,1].set_ylabel('Final Epoch')
    axes[1,1].set_title('Convergence by Lambda Parameter')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_effects(df):
    """Analyze effects of different parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot of validation loss by active_dim_2
    df.boxplot(column='final_val_loss', by='active_dim_2', ax=axes[0,0])
    axes[0,0].set_title('Validation Loss by Active Dimensions')
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    
    # Box plot of validation loss by lambda
    df.boxplot(column='final_val_loss', by='lmda', ax=axes[0,1])
    axes[0,1].set_title('Validation Loss by Lambda Parameter')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # Heatmap of validation loss by active_dim_2 and n_train2
    pivot_data = df.pivot_table(values='final_val_loss', 
                               index='active_dim_2', 
                               columns='n_train2', 
                               aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.2e', ax=axes[1,0], cbar_kws={'label': 'Validation Loss'})
    axes[1,0].set_title('Validation Loss Heatmap: Active Dim vs Training Size')
    
    # Heatmap of final epoch by active_dim_2 and n_train2
    pivot_epoch = df.pivot_table(values='final_epoch', 
                                index='active_dim_2', 
                                columns='n_train2', 
                                aggfunc='mean')
    sns.heatmap(pivot_epoch, annot=True, fmt='.0f', ax=axes[1,1], cbar_kws={'label': 'Final Epoch'})
    axes[1,1].set_title('Convergence Epoch Heatmap: Active Dim vs Training Size')
    
    plt.tight_layout()
    plt.savefig('parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seed_analysis(df):
    """Analyze variability across different seeds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Validation loss by seed
    df.boxplot(column='final_val_loss', by='seed', ax=axes[0,0])
    axes[0,0].set_title('Validation Loss Variability by Seed')
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    
    # Final epoch by seed
    df.boxplot(column='final_epoch', by='seed', ax=axes[0,1])
    axes[0,1].set_title('Convergence Epoch Variability by Seed')
    axes[0,1].grid(True, alpha=0.3)
    
    # Scatter plot showing seed effects
    for seed in df['seed'].unique():
        subset = df[df['seed'] == seed]
        axes[1,0].scatter(subset['n_train2'], subset['final_val_loss'], 
                         label=f'Seed {seed}', alpha=0.6)
    axes[1,0].set_xlabel('n_train2')
    axes[1,0].set_ylabel('Final Validation Loss')
    axes[1,0].set_title('Validation Loss by Seed and Training Size')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Standard deviation of validation loss by training size
    std_by_training = df.groupby('n_train2')['final_val_loss'].std()
    axes[1,1].plot(std_by_training.index, std_by_training.values, 'o-')
    axes[1,1].set_xlabel('n_train2')
    axes[1,1].set_ylabel('Std Dev of Validation Loss')
    axes[1,1].set_title('Variability in Validation Loss by Training Size')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_val_vs_samples_by_lambda(df):
    """Plot validation loss vs number of samples (n_train2), averaged over seeds,
    with different lines for different lmda values. Generates two figures for
    active_dim_2 == 5 and active_dim_2 == 40.
    """
    for active_dim in [5, 40]:
        subset = df[df['active_dim_2'] == active_dim]
        if subset.empty:
            continue

        grouped = (subset
                   .groupby(['lmda', 'n_train2'], as_index=False)
                   .agg(mean_val_loss=('final_val_loss', 'mean'),
                        std_val_loss=('final_val_loss', 'std'),
                        count=('final_val_loss', 'size')))

        fig, ax = plt.subplots(figsize=(8, 6))
        for lmda_value, data_lmda in grouped.groupby('lmda'):
            data_lmda = data_lmda.sort_values('n_train2')
            ax.plot(data_lmda['n_train2'], data_lmda['mean_val_loss'], marker='o', label=f"lmda={lmda_value}")

            # Optional error bars (std over seeds). Commented out to keep plot clean.
            # ax.fill_between(
            #     data_lmda['n_train2'],
            #     (data_lmda['mean_val_loss'] - data_lmda['std_val_loss']).clip(lower=0),
            #     data_lmda['mean_val_loss'] + data_lmda['std_val_loss'],
            #     alpha=0.15
            # )

        ax.set_title(f"Validation Loss vs Samples by lmda (active_dim_2={active_dim})")
        ax.set_xlabel('n_train2 (number of samples)')
        ax.set_ylabel('Mean Final Validation Loss (over seeds)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(title='lmda')
        plt.tight_layout()
        plt.savefig(f'val_loss_vs_samples_active_dim_{active_dim}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function"""
    print("Loading and exploring experiment data...")
    df = load_and_explore_data()
    
    print("\nGenerating plots...")
    
    print("1. Training Data Scaling Analysis")
    plot_training_data_scaling(df)
    
    print("2. Loss Analysis")
    plot_loss_analysis(df)
    
    print("3. Convergence Analysis")
    plot_convergence_analysis(df)
    
    print("4. Parameter Effects Analysis")
    plot_parameter_effects(df)
    
    print("5. Seed Variability Analysis")
    plot_seed_analysis(df)
    
    print("6. Val loss vs samples by lambda (active_dim_2=5 and 40)")
    plot_val_vs_samples_by_lambda(df)
    
    print("\nAnalysis complete! All plots have been saved as PNG files:")
    print("- training_data_scaling.png")
    print("- loss_analysis.png") 
    print("- convergence_analysis.png")
    print("- parameter_effects.png")
    print("- seed_analysis.png")
    print("- val_loss_vs_samples_active_dim_5.png")
    print("- val_loss_vs_samples_active_dim_40.png")

if __name__ == "__main__":
    main()