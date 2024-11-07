import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_result_file(filepath):
    return pd.read_csv(filepath)

def annotate_p_values(axes, x1, x2, y, p_val, offset=0, line_color='black', text_color='black', fontsize=18):
    """Helper function to annotate p-values on the plot with adjustable offset and line color."""
    h = 0.1 + offset
    line_style = {'lw': 2, 'color': line_color, 'linestyle': '--'}
    axes.plot([x1, x1, x2, x2], [y, y+h, y+h, y], **line_style)
    axes.text((x1 + x2) * 0.5, y + h, f'p = {p_val:.2e}', ha='center', va='bottom', color=text_color, fontsize=fontsize)

def plot_box_plots(df, category):
    category_df = df[df['intent_category'] == category]
    aspects = ['completeness', 'factuality', 'usefulness']
    answers = ['answer_1', 'answer_2', 'answer_3']

    fig, axes = plt.subplots(1, 3, figsize=(36, 12), sharey=True)  # Increased figure size
    box_colors = ['skyblue', 'orange', 'green']
    system_labels = ['Extractive QA', 'Generative QA', 'Response Selector']

    for i, aspect in enumerate(aspects):
        data_to_plot = []
        labels = []
        colors = []

        for j, answer in enumerate(answers):
            answer_idx = answer.split("_")[1]
            aspect_column = f'{aspect}_{answer_idx}'
            if aspect_column in category_df.columns:
                valid_ratings = category_df[aspect_column].dropna().values
                if len(valid_ratings) > 0:
                    data_to_plot.append(valid_ratings)
                    colors.append(box_colors[j])
                    labels.append(system_labels[j])

        box = axes[i].boxplot(
            data_to_plot, 
            labels=labels, 
            patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 3},  # Thicker median line
            boxprops={'linewidth': 2},  # Thicker box edges
            whiskerprops={'linewidth': 2},  # Thicker whiskers
            capprops={'linewidth': 2}  # Thicker caps
        )

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        title = "Overall Usefulness" if aspect == "usefulness" else aspect.capitalize()
        axes[i].set_title(f'{title}', fontsize=34)  # Increased title font size
        axes[i].set_ylabel('Ratings', fontsize=30, labelpad=25)
        axes[i].tick_params(axis='y', labelsize=29)
        axes[i].tick_params(axis='x', labelsize=29)

        axes[i].set_ylim(0.5, 6.2)
        axes[i].set_yticks([1, 2, 3, 4, 5])

        # Annotate p-values for comparisons
        if len(data_to_plot) >= 2:
            t_stat1, p_val1 = stats.ttest_ind(data_to_plot[0], data_to_plot[1])
            annotate_p_values(axes[i], 1, 2, 5.4, p_val1, offset=0.1, line_color='blue', text_color='blue', fontsize=20)

            if len(data_to_plot) == 3:
                t_stat2, p_val2 = stats.ttest_ind(data_to_plot[0], data_to_plot[2])
                t_stat3, p_val3 = stats.ttest_ind(data_to_plot[1], data_to_plot[2])
                annotate_p_values(axes[i], 1, 3, 5.6, p_val2, offset=0.25, line_color='green', text_color='green', fontsize=20)
                annotate_p_values(axes[i], 2, 3, 5.3, p_val3, offset=0.3, line_color='red', text_color='red', fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.88, bottom=0.15)
    plt.savefig(f"{category}_plot_with_pvalues.png", dpi=400)  # High dpi for quality in LaTeX

# Load and plot data
df = load_result_file("./results_reports.csv")
plot_box_plots(df, 'out_of_scope')
plot_box_plots(df, 'faq')
