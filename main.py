import os
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.ticker import FuncFormatter

def load_all_csv_from_folder(base_path):
    csv_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df


def boxplot_generator(dataset, variable, error, variable_name_for_plot):

    df_trained = dataset[(dataset['Tuned'] == 1)]
    variables = df_trained[variable].unique()

    data = [df_trained[df_trained[variable] == var][error].values for var in variables]

    plt.figure(figsize=(8, 6))
    plt.clf()

    variables2labels = variables
    for i in range(len(variables2labels)):
        if variables2labels[i] == 'RepeatedHoldOut':
            variables2labels[i] = 'MonteCarlo'
        if variables2labels[i] == 'OPTUNA':
            variables2labels[i] = 'TPE'

    box = plt.boxplot(data, tick_labels=variables2labels)
    means = [np.mean(d) for d in data]

    for i in range(len(variables)):
        plt.scatter(i + 1.25, means[i], color='orange', zorder=5, label='Średnia' if i == 0 else "")
        plt.text(i + 1.3, means[i], f'{means[i]:.2f}', verticalalignment='center', horizontalalignment='left',
                color='orange')

        # Pobierz wartości statystyk dla i-tego boxplotu
        median = np.median(data[i])
        q1 = np.percentile(data[i], 25)
        q3 = np.percentile(data[i], 75)
        whisker_low = np.min(data[i][data[i] >= box['whiskers'][i*2].get_ydata()[1]])
        whisker_high = np.max(data[i][data[i] <= box['whiskers'][i*2 + 1].get_ydata()[1]])

        # Dodanie tekstu na wykresie
        plt.text(i + 1, median, f'{median:.2f}', horizontalalignment='center', color='black')
        plt.text(i + 1, q1, f'{q1:.2f}', horizontalalignment='center', color='blue')
        plt.text(i + 1, q3, f'{q3:.2f}', horizontalalignment='center', color='blue')
        plt.text(i + 1, whisker_low, f'{whisker_low:.2f}', horizontalalignment='center', verticalalignment='top', color='red')
        plt.text(i + 1, whisker_high, f'{whisker_high:.2f}', horizontalalignment='center', verticalalignment='bottom', color='red')

    plt.xlabel(f'Nazwa {variable_name_for_plot}')
    if error == 'MedianAE':
        plt.ylabel(f'Mediana')
    else:
        plt.ylabel(f'{error}')
    plt.grid(True)
    plt.show()


def plot_better_results(data, error):
    better = 0
    same = 0
    worse = 0

    actual_not_trained = data.iloc[0][error]
    for i in range(1, len(data)):
        if data.iloc[i]['Tuned'] == 0:
            actual_not_trained = data.iloc[i][error]
        else:
            if data.iloc[i][error] < actual_not_trained:
                better += 1
            elif data.iloc[i][error] == actual_not_trained:
                same += 1
            else:
                worse += 1

    return better, same, worse


def plot_better(data, variable, error, variable_label):
    plt.style.use('_mpl-gallery')

    variables = data[variable].unique()
    variables = [x for x in variables if pd.notna(x)]

    bar_width = 0.25
    index = range(len(variables))

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, var in enumerate(variables):
        # data_final = data[((data[variable] == var) | (data['Tuned'] == 0)) & (data['Hyper_param_method'] != 'SA')]
        data_final = data[(((data[variable] == var) | (data['Tuned'] == 0)))]
        better, same, worse = plot_better_results(data_final, error)

        # Ustawienie pozycji słupków
        ax.bar(i - bar_width, better, bar_width, color='green', label='Lepsze od \nniezoptymalizowanego modelu' if i == 0 else "")
        ax.bar(i, same, bar_width, color='blue', label='Takie same jak \nniezoptymalizowany model' if i == 0 else "")
        ax.bar(i + bar_width, worse, bar_width, color='red', label='Gorsze od \nniezoptymalizowanego modelu' if i == 0 else "")

        # Dodanie wartości na słupkach
        ax.text(i - bar_width, better + 0.1, str(better), ha='center', va='bottom')
        ax.text(i, same + 0.1, str(same), ha='center', va='bottom')
        ax.text(i + bar_width, worse + 0.1, str(worse), ha='center', va='bottom')

    # Ustawienia osi i tytułów
    ax.set_xlabel(f'Nazwa {variable_label}')
    ax.set_ylabel('Ilość eksperymentów')
    ax.set_xticks(index)
    variables2labels = variables
    for i in range(len(variables2labels)):
        if variables2labels[i] == 'OPTUNA':
            variables2labels[i] = 'TPE'
        if variables2labels[i] == 'RepeatedHoldOut':
            variables2labels[i] = 'MonteCarlo'
    ax.set_xticklabels(variables2labels, rotation=45)

    ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    ax.grid(True)
    # plt.tight_layout(pad=2.0)  # Dostosowanie paddingu

    plt.tight_layout()
    plt.show()


def parse_time(t):
    # Parse a time string e.g., '0:00:00.051098' into a timedelta object
    h, m, s = t.split(':')
    sec, micro = s.split('.')
    td = timedelta(hours=int(h), minutes=int(m), seconds=int(sec), microseconds=int(micro))
    return td.total_seconds() / 60  # convert timedelta to total minutes


def time_plot(dataset, dataset1, dataset2, variable, variable_name_for_plot):
    plt.clf()

    def calculate_averages(df):
        df_trained = df[df['Tuned'] == 1]
        variables = sorted(df_trained[variable].unique().tolist())
        averages = []
        for var in variables:
            times = df_trained[df_trained[variable] == var]['time_tuning'].values
            times_in_minutes = [parse_time(t) for t in times]
            mean_time = np.mean(times_in_minutes)
            averages.append(mean_time)
        return variables, averages

    def comma_format(x, pos):
        return f'{x:.2f}'.replace('.', ',')

    variables, averages = calculate_averages(dataset)
    variables1, averages1 = calculate_averages(dataset1)
    variables2, averages2 = calculate_averages(dataset2)
    for i in range(len(variables)):
        if variables[i] == 'OPTUNA':
            variables[i] = 'TPE'
        if variables[i] == 'RepeatedHoldOut':
            variables[i] = 'MonteCarlo'
    plt.figure(figsize=(9, 5))
    plt.plot(variables, averages, marker='o', linestyle='-', linewidth=2, label=f'n_splits = 3')
    plt.plot(variables, averages1, marker='s', linestyle='--', linewidth=2,
             label=f'n_splits = 5')
    plt.plot(variables, averages2, marker='s', linestyle='-', linewidth=2,
             label=f'n_splits = 10')

    # Ustawienia osi
    plt.ylabel('Średni czas (w minutach)', fontsize=14)
    plt.xlabel('Metoda', fontsize=14)

    # Zwiększenie czcionki dla osi X i Y
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Zastosowanie formatera zamieniającego "." na ","
    plt.gca().yaxis.set_major_formatter(FuncFormatter(comma_format))

    # Dodanie siatki, legendy i wyświetlenie wykresu
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

def bar_plot(df, variable):
    plt.clf()
    plt.figure(figsize=(12, 8))

    df = df[df['Tuned'] == 1]

    # Grupy według metody walidacji i algorytmu ML, następnie obliczenie średniej i odchylenia standardowego
    grouped = df.groupby([variable, 'Regressor'])['MAE'].agg(['mean', 'std']).reset_index()

    # Iteracja przez każdy algorytm ML
    for algorithm in df['Regressor'].unique():
        subset = grouped[grouped['Regressor'] == algorithm]

        # Wykres line plot z error bars
        plt.errorbar(subset[variable], subset['mean'], yerr=subset['std'], label=algorithm, capsize=5,
                     marker='o', linestyle='-', linewidth=2)

        # Linie trendu
        z = np.polyfit(range(len(subset[variable])), subset['mean'], 1)
        p = np.poly1d(z)
        plt.plot(subset[variable], p(range(len(subset[variable]))), linestyle='--', color='gray')

    plt.xlabel('Nazwa metody walidacji')
    plt.ylabel('Średnia wartość błędu MAE')
    plt.title('Wpływ różnych metod optymalizacji na wyniki metod walidacji')
    plt.legend(title='Metoda walidacji')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    df = load_all_csv_from_folder('ml-tuning-08-07-2024-splits-3')
    df1 = load_all_csv_from_folder('ml-tuning-08-09-2024-splits-5')
    df2 = load_all_csv_from_folder('ml-tuning-08-16-2024-splits-10')

    time_plot(df, df1, df2, 'Hyper_param_method', '')
    time_plot(df, df1, df2, 'Validation_method', '')

    # plot_better(df, 'Hyper_param_method', 'MAE', 'metody optymalizacji')
    # plot_better(df, 'Validation_method', 'MAE', 'metody walidacji')
    # plot_better(df1, 'Hyper_param_method', 'MAE', 'metody optymalizacji')
    # plot_better(df1, 'Validation_method', 'MAE', 'metody walidacji')
    # plot_better(df2, 'Hyper_param_method', 'MAE', 'metody optymalizacji')
    # plot_better(df2, 'Validation_method', 'MAE', 'metody walidacji')
    #
    # boxplot_generator(df, 'Hyper_param_method', 'MAE', 'metod optymalizacji')
    # boxplot_generator(df1, 'Hyper_param_method', 'MAE', 'metod optymalizacji')
    # boxplot_generator(df2, 'Hyper_param_method', 'MAE', 'metod optymalizacji')
    # boxplot_generator(df, 'Validation_method', 'MAE', 'metod walidacji')
    # boxplot_generator(df1, 'Validation_method', 'MAE', 'metod walidacji')
    # boxplot_generator(df2, 'Validation_method', 'MAE', 'metod walidacji')
    #
    # bar_plot(df, 'Validation_method')
    # bar_plot(df1, 'Validation_method')
    # bar_plot(df2, 'Validation_method')
    # bar_plot(df, 'Hyper_param_method')
    # bar_plot(df1, 'Hyper_param_method')
    # bar_plot(df2, 'Hyper_param_method')


