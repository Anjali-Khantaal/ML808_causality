"""
Step 4: Causal DAG Construction (Causal Discovery)
---------------------------------------------------
This script performs the following:
1. Loads preprocessed data (SP500_Returns, GS10_Change, FEDFUNDS_Level, Inflation_YoY).
2. Constructs a domain-knowledge DAG.
3. Runs the PC algorithm (using causal-learn) and converts its output to a NetworkX graph.
4. Runs DirectLiNGAM (via lingam) to discover a candidate DAG.
5. Performs pairwise Granger causality tests.
6. Uses DoWhy for a simple intervention analysis on the effect of Inflation_YoY on SP500_Returns.
7. Merges the manual DAG and PC algorithm output to create a union (“merged DAG”).
8. Visualizes all DAGs and saves the merged DAG.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------
# 1. Load Preprocessed Data
# ---------------------------
# Ensure that "final_processed_data.csv" is available (from Step 3).
data = pd.read_csv("final_processed_data.csv", index_col=0, parse_dates=True)
# Select key variables.
data_cd = data[['SP500_Returns', 'GS10_Change', 'FEDFUNDS_Level', 'Inflation_YoY']].dropna()
print("Data loaded for causal discovery. Shape:", data_cd.shape)

# ---------------------------
# 2. Domain Knowledge: Manual DAG
# ---------------------------
# Domain assumptions:
#   FEDFUNDS_Level -> Inflation_YoY.
#   FEDFUNDS_Level -> GS10_Change.
#   Inflation_YoY -> SP500_Returns.
#   GS10_Change -> SP500_Returns.
manual_dag = nx.DiGraph()
manual_dag.add_edges_from([
    ("FEDFUNDS_Level", "Inflation_YoY"),
    ("FEDFUNDS_Level", "GS10_Change"),
    ("Inflation_YoY", "SP500_Returns"),
    ("GS10_Change", "SP500_Returns")
])

plt.figure(figsize=(4,4))
nx.draw(manual_dag, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
plt.title("Domain Knowledge DAG")
plt.show()

# ---------------------------
# 3. PC Algorithm (causal-learn)
# ---------------------------
from causallearn.search.ConstraintBased.PC import pc

# Convert DataFrame to numpy array.
data_array = data_cd.values

# Run PC algorithm with Fisher’s Z test.
pc_result = pc(data_array, alpha=0.05, indep_test='fisherz')
print("PC algorithm completed.")

# Extract adjacency matrix.
adj_matrix = pc_result.G.graph
print("PC algorithm adjacency matrix:\n", adj_matrix)

# Convert to NetworkX DiGraph.
pc_nx = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
mapping = {i: label for i, label in enumerate(list(data_cd.columns))}
pc_nx = nx.relabel_nodes(pc_nx, mapping)

plt.figure(figsize=(4,4))
nx.draw(pc_nx, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10)
plt.title("PC Algorithm Causal DAG")
plt.show()

# ---------------------------
# 4. DirectLiNGAM
# ---------------------------
from lingam import DirectLiNGAM

model_lingam = DirectLiNGAM()
model_lingam.fit(data_cd)
lingam_adj_matrix = model_lingam.adjacency_matrix_
print("LiNGAM adjacency matrix:\n", lingam_adj_matrix)

# Create LiNGAM DAG.
lingam_dag = nx.DiGraph()
cols = list(data_cd.columns)
threshold = 1e-6  # Ignore near-zero weights.
for i in range(len(cols)):
    for j in range(len(cols)):
        if np.abs(lingam_adj_matrix[i, j]) > threshold:
            lingam_dag.add_edge(cols[i], cols[j], weight=lingam_adj_matrix[i, j])

plt.figure(figsize=(4,4))
nx.draw(lingam_dag, with_labels=True, node_color='lightyellow', node_size=2000, font_size=10)
plt.title("LiNGAM Discovered Causal DAG")
plt.show()

# ---------------------------
# 5. Pairwise Granger Causality Tests
# ---------------------------
import statsmodels.tsa.stattools as tsastat

maxlag = 12
variables = list(data_cd.columns)
granger_results = {}

print("Pairwise Granger Causality Test Results (min p-value across lags):")
for cause in variables:
    for effect in variables:
        if cause != effect:
            test_result = tsastat.grangercausalitytests(data_cd[[effect, cause]], maxlag=maxlag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
            min_p_value = np.min(p_values)
            granger_results[(cause, effect)] = min_p_value
            print(f"{cause} -> {effect}: min p-value = {min_p_value:.4f}")

# ---------------------------
# 6. DoWhy Intervention Analysis
# ---------------------------
import dowhy
from dowhy import CausalModel

# Define assumed causal graph (DOT format).
graph_dot = """
digraph {
    FEDFUNDS_Level -> Inflation_YoY;
    FEDFUNDS_Level -> GS10_Change;
    Inflation_YoY -> SP500_Returns;
    GS10_Change -> SP500_Returns;
}
"""

model = CausalModel(
    data=data_cd,
    treatment="Inflation_YoY",
    outcome="SP500_Returns",
    graph=graph_dot
)
model.view_model()  # This creates a visual representation (requires Graphviz).

identified_estimand = model.identify_effect()
print("\nIdentified estimand:\n", identified_estimand)

estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print("\nCausal Estimate (Effect of Inflation YoY on SP500 Returns):", estimate.value)

refute_placebo = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
refute_subset = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter")
print("\nDoWhy Refutation Results:")
print("Placebo treatment refuter:\n", refute_placebo)
print("Data subset refuter:\n", refute_subset)

# ---------------------------
# 7. Merged DAG (Domain ∪ PC)
# ---------------------------
merged_dag = nx.DiGraph()
merged_dag.add_edges_from(manual_dag.edges())  # from domain knowledge
merged_dag.add_edges_from(pc_nx.edges())         # from PC algorithm

# Print merged DAG edges for review.
print("Merged DAG edges:", list(merged_dag.edges()))
plt.figure(figsize=(4,4))
nx.draw(merged_dag, with_labels=True, node_color='pink', node_size=2000, font_size=10)
plt.title("Merged DAG (Domain ∪ PC)")
plt.show()

# Save the merged DAG.
plt.savefig("./merged_causal_dag.png")

# ---------------------------
# 8. Final Visualization: Compare All DAGs
# ---------------------------
plt.figure(figsize=(12,4))

plt.subplot(1, 3, 1)
nx.draw(manual_dag, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
plt.title("Domain Knowledge DAG")

plt.subplot(1, 3, 2)
nx.draw(pc_nx, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10)
plt.title("PC Algorithm DAG")

plt.subplot(1, 3, 3)
nx.draw(lingam_dag, with_labels=True, node_color='lightyellow', node_size=2000, font_size=10)
plt.title("LiNGAM DAG")

plt.tight_layout()
plt.show()

plt.savefig("final_causal_dag_comparison.png")

print("Step 4: Causal DAG Construction Complete.")
