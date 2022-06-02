import matplotlib.pyplot as plt
from pandas import read_csv


def visualize_results(path, save_file: bool):
    df = read_csv(path + '/progress.csv', sep=';')
    t = range(len(df['time']))
    with plt.style.context('Solarize_Light2'):
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        ax1 = fig.add_subplot(111)
        plt.grid(False)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        plt.grid(False)

        my_colors = plt.rcParams['axes.prop_cycle']()

        ax1.set_xlabel('iterations')
        ax1.set_ylabel('loss')
        ax1.tick_params(axis='y')

        ax2.set_ylabel('f1-score/accuracy')  # we already handled the x-label with ax1

        training_loss = ax1.plot(t, df['training loss'], label='training_loss', **next(my_colors))
        validating_loss = ax1.plot(t, df['loss'], label='validating_loss', **next(my_colors))
        f1_score_0 = ax2.plot(t, df['f1_0'], label='f1_score_0', **next(my_colors))
        f1_score_period = ax2.plot(t, df['f1_PERIOD'], label='f1_score_period', **next(my_colors))
        f1_score_comma = ax2.plot(t, df['f1_COMMA'], label='f1_score_comma', **next(my_colors))
        accuracy = ax2.plot(t, df['accuracy'], label='accuracy', **next(my_colors))
        ax2.tick_params(axis='y')

        lines = training_loss + validating_loss + accuracy + f1_score_comma + f1_score_period + f1_score_0
        ax2.legend(lines, [line.get_label() for line in lines], bbox_to_anchor=(1.1, 1), loc='upper left',
                   borderaxespad=0.)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if save_file:
            plt.savefig(path + '/progress_visualization.png')
        else:
            plt.show()


if __name__ == '__main__':
    visualize_results('models/20220601_201550', False)
