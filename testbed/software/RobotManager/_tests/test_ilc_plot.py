import pickle

from core.utils.ilc.ILC_DAMN_bib import plot_bilbo_ilc_progression

if __name__ == '__main__':
    with open("ilc_data.pkl", "rb") as f:
        data = pickle.load(f)

    plot_bilbo_ilc_progression(data['y'],
                               data['e_norm_tracking'],
                               data['reference'],
                               )

    # yv, e_norm_tracking,
    # e_norm_prediction)
