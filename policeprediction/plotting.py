from matplotlib import pyplot as plt


def plot_attendance_rate_by_feature(df, feature, aggregation='mean', x_ticks=True):
    plt.figure(figsize=(14, 6))
    grouped_df = df.groupby(feature).agg(aggregation)['label']
    plt.plot(grouped_df, alpha=0.7)

    plt.xlabel(feature)
    plt.ylabel(aggregation)
    if (grouped_df.index.dtype == 'O') & x_ticks:
        plt.xticks(range(grouped_df.count()), grouped_df.index)
