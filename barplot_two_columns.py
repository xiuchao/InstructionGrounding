import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# 1) Prepare the data
# ------------------
data = {
    "Model Name": [
        "VLM-LLM", "Gemini-2.5-Pro", "Gemini-2.0-Flash", "GPT-5", "GPT-5-mini", "GPT-4.5", "GPT-4o", "o4-mini", "GPT-4o-mini", "GPT-4V", "Qwen2-VL-72B",
        "Llama-3.2V-90B", "Llama-3.2V-90B-Q4", "Llama-3.2V-11B", "Llama-3.2V-11B-Q4"
    ],

                 # vlm-llm, ge2.5, ge2,  gpt5,  gpt5mini, gpt4.5, gpt4o, o4, 4omini, gpt4v, qwen2,90b,   90q4,  11b,   11q4
    "Easy_im":     [0.050, 0.778, 0.833, 0.95,  0.850,  0.944,  0.850, 0.95,   0.750, 0.650, 0.800, 0.750, 0.800, 0.650, 0.650],
    "Easy_attr":   [0.516, 0.889, 0.889, 1.0,   1.0,    0.889,  1.000, 1.0,    0.717, 0.750, 0.917, 0.850, 0.667, 0.667, 0.567],
    "Easy_rel":    [0.131, 0.830, 0.774, 0.867, 0.894,  0.894,  0.778, 0.907,  0.550, 0.598, 0.830, 0.704, 0.598, 0.631, 0.502],
    "Medium_im":   [0.010, 0.847, 0.819, 0.903, 0.819,  0.917,  0.819, 0.847,  0.764, 0.750, 0.792, 0.708, 0.625, 0.764, 0.694],
    "Medium_attr": [0.336, 0.814, 0.721, 0.960, 0.956,  0.838,  0.948, 0.988,  0.771, 0.737, 0.756, 0.853, 0.719, 0.710, 0.757],
    "Medium_rel":  [0.186, 0.815, 0.642, 0.899, 0.823,  0.722,  0.680, 0.837,  0.596, 0.662, 0.738, 0.711, 0.554, 0.556, 0.555],
    "Hard_im":     [0.000, 0.985, 1.00,  0.917, 0.958,  0.958,  0.901, 0.917,  0.750, 0.625, 0.875, 0.875, 0.542, 0.833, 0.542],
    "Hard_attr":   [0.318, 0.784, 0.668, 0.916, 0.759,  0.719,  0.697, 0.809,  0.382, 0.417, 0.700, 0.491, 0.464, 0.536, 0.498],
    "Hard_rel":    [0.174, 0.858, 0.469, 0.816, 0.813,  0.698,  0.469, 0.804,  0.248, 0.455, 0.529, 0.521, 0.300, 0.342, 0.450]
}

df = pd.DataFrame(data)

# Calculate average columns (implicit/im, attribute/attr, relationship/rel)
df["Avg_im"]   = df[["Easy_im",   "Medium_im",   "Hard_im"]].mean(axis=1)
df["Avg_attr"] = df[["Easy_attr", "Medium_attr", "Hard_attr"]].mean(axis=1)
df["Avg_rel"]  = df[["Easy_rel",  "Medium_rel",  "Hard_rel"]].mean(axis=1)


# ---------------------
# 2) Define plot settings
# ---------------------
# Colors
im_color   = "#FFC107"  # Amber
attr_color = "#008080"  # Teal
rel_color  = "#673AB7"  # Plum

# Typical single-column width for IEEE conferences
column_width = 4.25  # inches
aspect_ratio = 0.75  # for a ~4:3 aspect ratio
dpi_value    = 300


# -----------------------------
# 3) Helper function for bar plot
# -----------------------------
def create_barplot(df, models_list, x_offsets, output_filename):
    """
    Creates a bar plot of Avg_im, Avg_attr, Avg_rel for the specified models_list
    and saves it as output_filename (PDF).
    """

    # Filter dataframe to the chosen models
    df_filtered = df[df["Model Name"].isin(models_list)]

    # Figure setup
    fig_height = column_width * aspect_ratio
    plt.figure(figsize=(column_width, fig_height), dpi=dpi_value)

    # Sort out x positions
    x = range(len(df_filtered["Model Name"]))

    # Find the highest values in each category to annotate them
    highest_im   = df_filtered["Avg_im"].max()
    highest_attr = df_filtered["Avg_attr"].max()
    highest_rel  = df_filtered["Avg_rel"].max()

    # Plot bars
    for i, model in enumerate(df_filtered["Model Name"]):
        im_value   = df_filtered.loc[df_filtered["Model Name"] == model, "Avg_im"  ].values[0]
        attr_value = df_filtered.loc[df_filtered["Model Name"] == model, "Avg_attr"].values[0]
        rel_value  = df_filtered.loc[df_filtered["Model Name"] == model, "Avg_rel" ].values[0]

        # Draw bars side by side
        plt.bar(i,           im_value,   width=0.25, color=im_color,   label="implicit"    if i == 0 else "")
        plt.bar(i + 0.25,    attr_value, width=0.25, color=attr_color, label="attribute"   if i == 0 else "")
        plt.bar(i + 0.50,    rel_value,  width=0.25, color=rel_color,  label="relationship" if i == 0 else "")

        # Annotate only if it's the highest in its category
        if im_value   == highest_im:
            plt.text(i - x_offsets[0],   im_value ,   f"{im_value:.2f}",   ha='center', fontsize=10)
        if attr_value == highest_attr:
            plt.text(i + x_offsets[1],   attr_value, f"{attr_value:.2f}", ha='center', fontsize=10)
        if rel_value  == highest_rel:
            plt.text(i + x_offsets[2],   rel_value + 0.02,  f"{rel_value:.2f}",  ha='center', fontsize=10)

    # Legend styling --------------
    if False:
        legend = plt.legend(
            fontsize=9, loc='center',
            bbox_to_anchor=(0.8, 0.8),    # Adjust as desired
            framealpha=0.4, facecolor="lightgray", borderpad=0.8,
        )
        legend.get_frame().set_linewidth(0.5)

    # Axes, ticks, grid
    plt.ylim([0, 0.95])
    plt.yticks(np.arange(0.2, 1.0, 0.2), fontsize=9)  # e.g., every 0.1 from 0.1 to 0.9

    plt.ylabel("Average Acc", fontsize=10)
    plt.xticks([p + 0.25 for p in x], df_filtered["Model Name"], rotation=40, ha='right', fontsize=9)
    plt.grid(axis='both', linestyle='--', alpha=0.6)

    # Remove top/right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    plt.close()

# -------------------------------------
# 4) Actually create the two single‐column figures
# -------------------------------------
# Figure A: Close‐source Models
filtered_models = [
   "Gemini-2.5-Pro", "Gemini-2.0-Flash",  "GPT-5", "GPT-4o", "o4-mini", "GPT-5-mini", #"GPT-4V",
]
create_barplot(df, filtered_models, x_offsets=(0.1, 0.4, 0.6), output_filename="Figure4r.pdf")

# Figure B: Open‐source Models
quantified_models = [  "VLM-LLM",  "Qwen2-VL-72B", 
    "Llama-3.2V-90B", "Llama-3.2V-90B-Q4",
    "Llama-3.2V-11B", "Llama-3.2V-11B-Q4"
]

#create_barplot(df, quantified_models, x_offsets=(0., 0.3, 0.55), output_filename="Figure5r.pdf")
