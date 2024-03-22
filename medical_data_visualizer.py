import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
BMI_series = df['weight'] / (df['height'] / 100)**2 # BMI = kg/(m^2)
condition = BMI_series > 25
df['overweight'] = condition.astype(int)

# Normalize data by making 0 always good and 1 always bad
# (if the value of 'cholesterol' or 'gluc' is 1, make the value 0
# and if the value is more than 1, make the value 1)
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] > 1, 'gluc'] = 1

# Draw categorical plot
def draw_cat_plot():
    # Create dataframe for categorical plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight' columns
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], var_name='variable', value_name='value')

    # Group and reformat the data to split it by 'cardio' and show the counts of each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value',]).value_counts().reset_index(name='total')

    # Draw categorical plot with 'sns.catplot()'
    plot = sns.catplot(x='variable', y='total', data=df_cat, kind='bar', hue='value', col='cardio')

    # Access the figure object for the plot
    fig = plot.fig

    fig.savefig('catplot.png')
    return fig

# Draw heat map
def draw_heat_map():
    # Clean data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
             (df['height'] >= df['height'].quantile(0.025)) &
             (df['height'] <= df['height'].quantile(0.975)) &
             (df['weight'] >= df['weight'].quantile(0.025)) &
             (df['weight'] <= df['weight'].quantile(0.975))] 

    # Calculate correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", ax=ax)

    fig.savefig('heatmap.png')
    return fig