from core.utils.plotting import new_figure_agg, AggPDFPreviewer, save_figure, open_figure_preview

if __name__ == '__main__':

    fig, ax = new_figure_agg()
    fig.set_size_inches(10, 5)
    ax.title.set_text('Simple plot')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 4)
    ax.plot([1, 2, 3], [1, 4, 9])

    open_figure_preview(fig)
    print("END")