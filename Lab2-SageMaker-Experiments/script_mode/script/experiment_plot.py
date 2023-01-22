import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sagemaker.analytics import ExperimentAnalytics


def analyze_experiment(
    experiment_name: str,
    parameter_names: str,
    metric_names: str,
    stat_name: str = "Last",
):
    re_expr = f"(?:{'|'.join([f'{k}.*- {stat_name}' for k in metric_names] + parameter_names + ['DisplayName'])})"

    trial_component_analytics = ExperimentAnalytics(
        experiment_name=experiment_name,
        parameter_names=parameter_names,
    )
    df = trial_component_analytics.dataframe()
    df = df[df["SourceArn"].isna()]
    df = df.filter(regex=re_expr)

    # join the categorical parameters
    df_temp = df[parameter_names].select_dtypes("object")

    cat_col_name = "_".join(df_temp.columns.values)

    if len(df_temp.columns) > 1:
        df.loc[:, cat_col_name] = df_temp.astype(str).apply("_".join, axis=1)
        df = df.drop(columns=df_temp.columns.values)

    ordinal_params = df[parameter_names].select_dtypes("number").columns.tolist()
    df_plot = df.melt(id_vars=["DisplayName"] + ordinal_params + [cat_col_name])
    df_plot[["Dataset", "Metrics"]] = (
        df_plot.variable.str.split(" - ").str[0].str.split(":", expand=True)
    )
    f = plt.Figure(
        figsize=(8, 6 * len(ordinal_params)),
        facecolor="w",
        layout="constrained",
        frameon=True,
    )
    f.suptitle("Experiment Analysis")
    sf = f.subfigures(1, len(ordinal_params))

    if isinstance(sf, mpl.figure.SubFigure):
        sf = [sf]

    for k, p in zip(sf, ordinal_params):

        (
            so.Plot(
                df_plot,
                y="value",
                x=p,
                color=cat_col_name,
            )
            .facet(col="Dataset", row="Metrics")
            .add(so.Dot())
            .share(y=False)
            .limit(y=(0, None))
            .on(k)
            .plot()
        )
    return f
