import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_medcat_top_entities(medcat_results: pd.DataFrame, tuis, n: int = 10, status: str = "Affirmed") -> None:
    """Plot top entities.

    Args:
        medcat_results: The annotated results obtained from medcat analysis
        tuis: A list of tuis to filter before plotting; this is useful when, for example, only dieseases or symptoms should be plotted
        n: maximal number of entities to plot
        status: One of "Affirmed", "Other" or "Both". Affirmed reflect positive diagnoses, symptoms, ..., "Other" reflects absence of such entities and "Both"
        just uses all

    """
    # filter by status if wanted and count unique values of entities in the results
    # TODO: find efficient way to check if tuis is part of the type_ids since this could be a list with multiple entries
    if status != "Both":
        value_counts = medcat_results[(medcat_results["meta_anns"] == "Affirmed") & (medcat_results["type_ids"] == tuis)]["pretty_name"].value_counts()
    else:
        value_counts = medcat_results[medcat_results["type_ids"] == tuis]["pretty_name"].value_counts()
    # plot
    sns.set_theme(style="whitegrid")
    sns.barplot(x=value_counts.head(n), y=value_counts.head(n).index, orient="h", color="b")
    plt.show()
