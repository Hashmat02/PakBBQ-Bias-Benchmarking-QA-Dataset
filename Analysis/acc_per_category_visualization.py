import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv('category_accuracy.csv')

df['category'] = df['category'].replace('ENG-linguistic(language_formality_biases)', 'Language formality')

# Round accuracy columns to 2 decimals
for col in ['overall_accuracy', 'disambig_accuracy', 'ambig_accuracy']:
    df[col] = df[col].round(2)

languages_to_plot = ['ENG', 'UR']
sns.set(style='whitegrid', font_scale=1)  # slightly bigger font

language_titles = {'ENG': 'English', 'UR': 'Urdu'}

for language in languages_to_plot:
    lang_df = df[df['language'] == language]

    categories = lang_df['category'].unique()
    models = sorted(lang_df['model'].unique())

    n_cols, n_rows = 4, 2  # 4 columns and 2 rows fixed grid

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    for i, category in enumerate(categories):
        ax = axes[i]
        cat_df = lang_df[lang_df['category'] == category].sort_values('model')

        ax.plot(cat_df['model'], cat_df['overall_accuracy'], marker='o', label='Overall')
        ax.plot(cat_df['model'], cat_df['disambig_accuracy'], marker='o', label='Disambig')
        ax.plot(cat_df['model'], cat_df['ambig_accuracy'], marker='o', label='Ambig')

        ax.set_title(f'{category}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 100)

        # Show x-axis labels only for bottom row
        if i // n_cols == n_rows - 1:
            ax.set_xlabel('Model', fontsize=10)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        else:
            ax.set_xticks([])
            ax.set_xlabel('')

        ax.grid(False)

        if i == 0:
            ax.legend(title='Accuracy Type', fontsize=10, title_fontsize=11)

    # Remove unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a big language label on top centered
    fig.suptitle(language_titles.get(language, language), fontsize=20, fontweight='bold', y=1.05)

    plt.tight_layout(pad=3.0)  # increase padding
    plt.subplots_adjust(top=0.9)  # make space for suptitle

    plt.savefig(f'accuracy_lineplot_{language}_4x2_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
