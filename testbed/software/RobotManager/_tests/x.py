from core.utils.plotting import new_figure_agg, open_figure_preview

if __name__ == '__main__':
    fig, (ax1, ax2) = new_figure_agg(subplots=(2, 1), figsize=(5, 4), dpi=100)

    data1 = [1, 2, 3]
    data2 = [1, 4, 9]

    ax1.plot(data1)
    ax2.plot(data2)

    open_figure_preview(fig)
