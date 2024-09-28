import math
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_df_n_splits_3():
    csv_file_path0 = 'ml-tuning-08-07-2024-splits-3/model_performance_0-03-52-56.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path1 = 'ml-tuning-08-07-2024-splits-3/model_performance_1-07-59-42.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path2 = 'ml-tuning-08-07-2024-splits-3/model_performance_2-11-57-13.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path3 = 'ml-tuning-08-07-2024-splits-3/model_performance_3-16-03-05.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path4 = 'ml-tuning-08-07-2024-splits-3/model_performance_4-20-17-15.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path5 = 'ml-tuning-08-07-2024-splits-3/model_performance_5-00-28-29.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path6 = 'ml-tuning-08-07-2024-splits-3/model_performance_6-04-26-09.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path7 = 'ml-tuning-08-07-2024-splits-3/model_performance_7-08-29-08.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path8 = 'ml-tuning-08-07-2024-splits-3/model_performance_8-12-37-47.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path9 = 'ml-tuning-08-07-2024-splits-3/model_performance_9-16-51-54.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    df = pd.read_csv(csv_file_path0)
    df = pd.concat([df, pd.read_csv(csv_file_path1)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path2)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path3)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path4)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path5)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path6)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path7)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path8)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path9)], axis=0)
    return df


def load_df_n_splits_5():
    csv_file_path0 = 'ml-tuning-08-09-2024-splits-5/model_performance_0-23-28-51.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path1 = 'ml-tuning-08-09-2024-splits-5/model_performance_1-04-20-01.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path2 = 'ml-tuning-08-09-2024-splits-5/model_performance_2-08-21-57.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path3 = 'ml-tuning-08-09-2024-splits-5/model_performance_3-12-49-18.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path4 = 'ml-tuning-08-09-2024-splits-5/model_performance_4-17-20-51.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path5 = 'ml-tuning-08-09-2024-splits-5/model_performance_5-21-58-37.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path6 = 'ml-tuning-08-09-2024-splits-5/model_performance_6-02-23-44.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path7 = 'ml-tuning-08-09-2024-splits-5/model_performance_7-06-59-32.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path8 = 'ml-tuning-08-09-2024-splits-5/model_performance_8-11-35-31.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path9 = 'ml-tuning-08-09-2024-splits-5/model_performance_9-16-23-43.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV

    df = pd.read_csv(csv_file_path0)
    df = pd.concat([df, pd.read_csv(csv_file_path1)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path2)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path3)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path4)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path5)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path6)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path7)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path8)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path9)], axis=0)
    return df

def load_df_n_splits_10():

    csv_file_path0 = 'ml-tuning-08-16-2024-splits-10/model_performance_0-19-11-52.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path1 = 'ml-tuning-08-16-2024-splits-10/model_performance_1-02-16-20.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path2 = 'ml-tuning-08-16-2024-splits-10/model_performance_2-07-48-44.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path3 = 'ml-tuning-08-16-2024-splits-10/model_performance_3-13-58-20.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path4 = 'ml-tuning-08-16-2024-splits-10/model_performance_4-21-31-01.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path5 = 'ml-tuning-08-16-2024-splits-10/model_performance_5-04-11-29.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path6 = 'ml-tuning-08-16-2024-splits-10/model_performance_6-09-41-56.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path7 = 'ml-tuning-08-16-2024-splits-10/model_performance_7-16-20-49.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path8 = 'ml-tuning-08-16-2024-splits-10/model_performance_8-22-02-33.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV
    csv_file_path9 = 'ml-tuning-08-16-2024-splits-10/model_performance_9-03-20-18.csv'  # Zastąp tą ścieżką rzeczywistą ścieżką do pliku CSV

    df = pd.read_csv(csv_file_path0)
    df = pd.concat([df, pd.read_csv(csv_file_path1)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path2)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path3)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path4)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path5)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path6)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path7)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path8)], axis=0)
    df = pd.concat([df, pd.read_csv(csv_file_path9)], axis=0)
    return df

def boxplot_generator(dataset, variable, error, variable_name_for_plot):
    df_trained = dataset[(dataset['Tuned'] == 1)]
    # df_trained = dataset
    variables = df_trained[variable].unique()

    # variables = variables.tolist()
    # if variable == 'Validation_method':
    #     del variables[0]

    # df_filtered = dataset[(dataset['Tuned'] == 1)]

    # data = [df_trained[df_trained[variable] == var][f'{error}_scaled'].values for var in variables]
    data = [df_trained[df_trained[variable] == var][error].values for var in variables]

    # variables.append('None')
    # variables = np.array(variables)
    # df_filtered = dataset[(dataset['Tuned'] == 0)]
    #
    # print(len(df_filtered[error].values))
    # print(len(df_trained[df_trained['Validation_method'].isna()][error].values))
    # data.append(df_trained[df_trained['Validation_method'].isna()][error].values)

    plt.figure(figsize=(10, 6))
    variables2labels = variables
    for i in range(len(variables2labels)):
        if variables2labels[i] == 'RepeatedHoldOut':
            variables2labels[i] = 'MonteCarlo'
        if variables2labels[i] == 'OPTUNA':
            variables2labels[i] = 'TPE'

    box = plt.boxplot(data, tick_labels=variables2labels)
    # box = plt.boxplot(data, patch_artist=True, tick_labels=variables)
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
    # plt.title(f'Boxplot wartości błędów dla różnych {variable_name_for_plot}')
    plt.grid(True)
    plt.show()


def d2_multiple_boxplot_generator(dataset, variable1, variable2, error, variable_name_for_plot):
    df_trained = dataset[(dataset['Tuned'] == 1) & (dataset['Hyper_param_method'] != 'SA')]
    variables1 = df_trained[variable1].unique()

    num_vars = len(variables1)
    rows = 2  # Można zmienić na 3, jeśli chcemy trzy rzędy
    cols = (num_vars + 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(18, 10), sharey=True)
    axes = axes.flatten()  # Spłaszczenie tablicy osi dla łatwiejszego iterowania

    for ax, var1 in zip(axes, variables1):
        data_selected_for_var1 = df_trained[df_trained[variable1] == var1]
        variables2 = data_selected_for_var1[variable2].unique()
        data = [data_selected_for_var1[data_selected_for_var1[variable2] == var2][error].values for var2 in variables2]

        variables2labels = variables2
        for i in range(len(variables2labels)):
            if variables2labels[i] == 'OPTUNA':
                variables2labels[i] = 'TPE'
            if variables2labels[i] == 'RepeatedHoldOut':
                variables2labels[i] = 'MonteCarlo'

        box = ax.boxplot(data, tick_labels=variables2labels)

        # Dodawanie średnich wartości
        means = [np.mean(d) for d in data]
        for i in range(len(variables2)):
            ax.scatter(i + 1.25, means[i], color='orange', zorder=5, label='Średnia' if i == 0 else "")
            ax.text(i + 1.3, means[i], f'{means[i]:.2f}', verticalalignment='center', horizontalalignment='left', color='orange')

        # Dodawanie wartości statystyk na wykres
        for i in range(len(variables2)):
            median = np.median(data[i])
            q1 = np.percentile(data[i], 25)
            q3 = np.percentile(data[i], 75)
            whisker_low = box['whiskers'][i * 2].get_ydata()[1]
            whisker_high = box['whiskers'][i * 2 + 1].get_ydata()[1]

            ax.text(i + 1, median, f'{median:.2f}', horizontalalignment='center', verticalalignment='bottom', color='black')
            ax.text(i + 1, q1, f'{q1:.2f}', horizontalalignment='center', verticalalignment='bottom', color='blue')
            ax.text(i + 1, q3, f'{q3:.2f}', horizontalalignment='center', verticalalignment='top', color='blue')
            ax.text(i + 1, whisker_low, f'{whisker_low:.2f}', horizontalalignment='center', verticalalignment='top', color='red')
            ax.text(i + 1, whisker_high, f'{whisker_high:.2f}', horizontalalignment='center', verticalalignment='bottom', color='red')

        ax.set_xlabel(f'Nazwa {variable_name_for_plot}')
        if var1 == 'OPTUNA':
            ax.set_title(f'TPE')
        else:
            if var1 == 'RepeatedHoldOut':
                ax.set_title(f'MonteCarlo')
            else:
                ax.set_title(f'{var1}')

        ax.grid(True)

    # Usuwanie niepotrzebnych subplotów
    for j in range(len(variables1), len(axes)):
        fig.delaxes(axes[j])

    # fig.suptitle(f'Boxplot wartości błędów dla różnych {variable_name_for_plot}', fontsize=16)
    if error == 'MedianAE':
        axes[0].set_ylabel(f'Mediana')
    else:
        axes[0].set_ylabel(f'{error}')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    plt.show()


def multiple_boxplot_generator(dataset, variable1, variable2, error, variable_name_for_plot):
    df_trained = dataset[dataset['Tuned'] == 1]
    variables1 = df_trained[variable1].unique()
    regressors = df_trained['Dataset'].unique()
    print(regressors)

    for reg in regressors:
        data_selected = df_trained[df_trained['Dataset'] == reg]
        for var1 in variables1:
            data_selected_for_var1 = data_selected[data_selected[variable1] == var1]
            variables2 = data_selected_for_var1[variable2].unique()
            data = [data_selected_for_var1[data_selected_for_var1[variable2] == var2][error].values for var2 in variables2]

            plt.figure(figsize=(10, 6))
            box = plt.boxplot(data, tick_labels=variables2)

            # Dodawanie wartości statystyk na wykres
            for i in range(len(variables2)):
                # Pobierz wartości statystyk dla i-tego boxplotu
                median = np.median(data[i])
                q1 = np.percentile(data[i], 25)
                q3 = np.percentile(data[i], 75)
                whisker_low = box['whiskers'][i * 2].get_ydata()[1]
                whisker_high = box['whiskers'][i * 2 + 1].get_ydata()[1]

                # Dodanie tekstu na wykresie
                plt.text(i + 1, median, f'{median:.2f}', horizontalalignment='center', verticalalignment='bottom',
                         color='black')
                plt.text(i + 1, q1, f'{q1:.2f}', horizontalalignment='center', verticalalignment='bottom', color='blue')
                plt.text(i + 1, q3, f'{q3:.2f}', horizontalalignment='center', verticalalignment='top', color='blue')

                # Tekst dla wartości na końcach wąsów
                plt.text(i + 1, whisker_low, f'{whisker_low:.2f}', horizontalalignment='center',
                         verticalalignment='top', color='red')
                plt.text(i + 1, whisker_high, f'{whisker_high:.2f}', horizontalalignment='center',
                         verticalalignment='bottom', color='red')

            plt.xlabel(f'Nazwa {variable_name_for_plot}')
            if error == 'MedianAE':
                plt.ylabel(f'Mediana')
            else:
                plt.ylabel(f'{error}')

            # plt.title(f'Boxplot wartości błędów dla różnych {variable_name_for_plot} dla {var1} dla {reg}')
            plt.grid(True)
            plt.show()


def scale_group(group):
    scaler_MAE = StandardScaler()
    scaler_MedianAE = StandardScaler()
    scaler_MSE = StandardScaler()

    # Skalowanie kolumn MAE i MedianAE
    group['MAE_scaled'] = scaler_MAE.fit_transform(group[['MAE']])
    group['MedianAE_scaled'] = scaler_MedianAE.fit_transform(group[['MedianAE']])
    group['MSE_scaled'] = scaler_MSE.fit_transform(group[['MSE']])

    return group


def plot_better_results(data, error):
    better = 0
    same = 0
    worse = 0

    actual_not_trained = data.iloc[0][error]
    print(data)

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

    print(better, same, worse)

    return better, same, worse

def plot_better(data, variable1, variable2, variable_label, error):
    plt.style.use('_mpl-gallery')
    variables1 = data[variable1].unique()

    fig, axes = plt.subplots(len(variables1), 1, figsize=(12, 6 * len(variables1)), sharey=True)

    if len(variables1) == 1:
        axes = [axes]  # Upewnienie się, że 'axes' jest listą, gdy jest tylko jedna metoda optymalizacji

    for ax, var1 in zip(axes, variables1):
        data_selected_for_var1 = data.loc[(data[variable1] == var1) | (data['Tuned'] == 0)]
        print(data_selected_for_var1)
        variables2 = data_selected_for_var1[variable2].unique()
        variables2 = [x for x in variables2 if pd.notna(x)]

        index = range(len(variables2))
        bar_width = 0.25

        print(data_selected_for_var1.head(100))

        for i, var2 in enumerate(variables2):
            data_final = data_selected_for_var1.loc[(data_selected_for_var1[variable2] == var2) | (data_selected_for_var1['Tuned'] == 0)]

            print(data_final)
            print('......................')
            better, same, worse = plot_better_results(data_final, error)

            # Ustawienie pozycji słupków
            ax.bar(i - bar_width, better, bar_width, color='green', label='Lepsze' if i == 0 else "")
            ax.bar(i, same, bar_width, color='blue', label='Takie same' if i == 0 else "")
            ax.bar(i + bar_width, worse, bar_width, color='red', label='Gorsze' if i == 0 else "")

            # Dodanie wartości na słupkach
            ax.text(i - bar_width, better + 0.1, str(better), ha='center', va='bottom')
            ax.text(i, same + 0.1, str(same), ha='center', va='bottom')
            ax.text(i + bar_width, worse + 0.1, str(worse), ha='center', va='bottom')

        # Ustawienia osi i tytułów
        ax.set_xlabel(f'Nazwa {variable_label}')
        ax.set_ylabel('Ilość eksperymentów')
        ax.set_title(f'Porównanie wyników {error} dla {var1}')
        ax.set_xticks(index)
        ax.set_xticklabels(variables2, rotation=0)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_better_var1(data, variable, error, variable_label):
    plt.style.use('_mpl-gallery')

    variables = data[variable].unique()
    variables = [x for x in variables if pd.notna(x)]

    bar_width = 0.25
    index = range(len(variables))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, var in enumerate(variables):
        # data_final = data[((data[variable] == var) | (data['Tuned'] == 0)) & (data['Hyper_param_method'] != 'SA')]
        data_final = data[(((data[variable] == var) | (data['Tuned'] == 0)))]
        better, same, worse = plot_better_results(data_final, error)

        # Ustawienie pozycji słupków
        ax.bar(i - bar_width, better, bar_width, color='green', label='Lepsze od niezoptymalizowanego modelu' if i == 0 else "")
        ax.bar(i, same, bar_width, color='blue', label='Takie same jak niezoptymalizowany model' if i == 0 else "")
        ax.bar(i + bar_width, worse, bar_width, color='red', label='Gorsze od niezoptymalizowanego modelu' if i == 0 else "")

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
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def parse_time(t):
    # Parse a time string e.g., '0:00:00.051098' into a timedelta object
    h, m, s = t.split(':')
    sec, micro = s.split('.')
    td = timedelta(hours=int(h), minutes=int(m), seconds=int(sec), microseconds=int(micro))
    return td.total_seconds() / 60  # convert timedelta to total minutes


def line_plot1(dataset, dataset1, dataset2, variable, variable_name_for_plot):
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

    variables, averages = calculate_averages(dataset)
    variables1, averages1 = calculate_averages(dataset1)
    variables2, averages2 = calculate_averages(dataset2)
    plt.figure(figsize=(10, 6))
    plt.plot(variables, averages, marker='o', linestyle='-', linewidth=2, label=f'n_splits = 3')
    plt.plot(variables1, averages1, marker='s', linestyle='--', linewidth=2,
             label=f'n_splits = 5')
    plt.plot(variables2, averages2, marker='s', linestyle='-', linewidth=2,
             label=f'n_splits = 10')

    plt.title(f'Average Optimization Time Comparison')
    plt.xlabel(variable_name_for_plot)
    plt.ylabel('Average Time (minutes)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_averages_for_dataset(df, variable, variables2, metric, label):
    """ Helper function to process each dataset and plot their averages. """
    df_trained = df[df['Tuned'] == 1]
    # df_trained = df_trained[df_trained['Validation_method'].isin(variables2)]

    if variable == 'Validation_method':
        variables = variables2
    else:
        variables = sorted(df_trained[variable].unique())

    averages = []
    for var in variables:
        values = np.abs(df_trained[df_trained[variable] == var][metric])
        mean_value = np.mean(values)
        averages.append(mean_value)

    # Update labels for clarity
    variables2labels = [var.replace('RepeatedHoldOut', 'Monte Carlo') if var == 'RepeatedHoldOut' else var for var in
                        variables]

    plt.plot(variables2labels, averages, marker='o', linestyle='-', linewidth=2, label=label)


def line_plot_cvs(dataset, dataset1, dataset2, variable, metric, variable_name_for_plot):
    plt.figure(figsize=(10, 6))

    variables2 = ['KFold', 'RepeatedKFold', 'RepeatedHoldOut', 'HoldOut', 'Bootstrap']

    # Process and plot for each dataset
    plot_averages_for_dataset(dataset, variable, variables2, metric, 'n_splits = 3')
    plot_averages_for_dataset(dataset1, variable, variables2, metric, 'n_splits = 5')
    plot_averages_for_dataset(dataset2, variable, variables2, metric, 'n_splits = 10')

    # plt.title(f'Average Cross-Validation Scores by {variable_name_for_plot}')
    # plt.xlabel(variable_name_for_plot)
    plt.ylabel(f'Average {variable_name_for_plot}')
    plt.grid(True)
    plt.legend(title=variable_name_for_plot)
    plt.show()


# Example usage:
# Assuming 'dataset', 'dataset1', and 'dataset2' are DataFrames loaded with your data and properly formatted
# line_plot_cvs(dataset, dataset1, dataset2, 'Validation_method', 'Validation Method')


if __name__ == '__main__':

    df = load_df_n_splits_3()
    df1 = load_df_n_splits_5()
    df2 = load_df_n_splits_10()

    # line_plot_cvs(df, df1, df2, 'Validation_method', 'cross_val_score_vslid_mean','cross-validation score')
    # line_plot_cvs(df, df1, df2, 'Hyper_param_method', 'cross_val_score_vslid_mean', 'cross-validation score')

    # line_plot_cvs(df, df1, df2, 'Validation_method', 'MAE','MAE')
    # line_plot_cvs(df, df1, df2, 'Hyper_param_method', 'MAE', 'MAE')

    # line_plot1(df, df1, df2, 'Validation_method', '')
    # line_plot1(df, df1, df2, 'Hyper_param_method', '')

    # plot_better(df, 'Hyper_param_method', 'Validation_method', 'metody walidacji', 'MAE')
    plot_better_var1(df2, 'Hyper_param_method', 'MAE', 'metody optymalizacji')
    plot_better_var1(df2, 'Validation_method', 'MAE', 'metody walidacji')
    # plot_better_var1(df, 'Regressor', 'MAE', 'ML')

    # boxplot_generator(df, 'Regressor', 'MSE', 'algorytmów uczenia maszynowego')
    # boxplot_generator(df2, 'Regressor', 'MAE', 'algorytmów uczenia maszynowego')
    # boxplot_generator(df2, 'Regressor', 'MedianAE', 'algorytmów uczenia maszynowego')
    # boxplot_generator(df, 'Hyper_param_method', 'MSE', 'metod optymalizacji')
    boxplot_generator(df, 'Hyper_param_method', 'MAE', 'metod optymalizacji')
    boxplot_generator(df1, 'Hyper_param_method', 'MAE', 'metod optymalizacji')
    boxplot_generator(df2, 'Hyper_param_method', 'MAE', 'metod optymalizacji')
    # boxplot_generator(df2, 'Hyper_param_method', 'MedianAE', 'metod optymalizacji')
    # boxplot_generator(df, 'Validation_method', 'MSE', 'metod walidacji')
    boxplot_generator(df, 'Validation_method', 'MAE', 'metod walidacji')
    boxplot_generator(df1, 'Validation_method', 'MAE', 'metod walidacji')
    boxplot_generator(df2, 'Validation_method', 'MAE', 'metod walidacji')
    # boxplot_generator(df2, 'Validation_method', 'MedianAE', 'metod walidacji')
    # d2_multiple_boxplot_generator(df, 'Validation_method', 'Hyper_param_method', 'MAE', 'metody optymalizacji')
    # d2_multiple_boxplot_generator(df, 'Validation_method', 'Hyper_param_method', 'MedianAE', 'metody optymalizacji')
    # d2_multiple_boxplot_generator(df, 'Hyper_param_method', 'Validation_method', 'MAE', 'metody walidacji')
    # d2_multiple_boxplot_generator(df, 'Hyper_param_method', 'Validation_method', 'MedianAE', 'metody walidacji')
    #
    # boxplot_generator(df, 'Dataset', 'MAE', 'zbiorów danych')
    # boxplot_generator(df, 'Dataset', 'MedianAE', 'zbiorów danych')

    ######################### BAR PLOT ##############################

    # plt.figure(figsize=(12, 8))
    #
    # df = df[df['Tuned'] == 1]
    #
    # # Grupy według metody walidacji i algorytmu ML, następnie obliczenie średniej i odchylenia standardowego
    # grouped = df.groupby(['Validation_method', 'Regressor'])['MAE'].agg(['mean', 'std']).reset_index()
    #
    # # Iteracja przez każdy algorytm ML
    # for algorithm in df['Regressor'].unique():
    #     subset = grouped[grouped['Regressor'] == algorithm]
    #
    #     # Wykres line plot z error bars
    #     plt.errorbar(subset['Validation_method'], subset['mean'], yerr=subset['std'], label=algorithm, capsize=5,
    #                  marker='o', linestyle='-', linewidth=2)
    #
    #     # Linie trendu
    #     z = np.polyfit(range(len(subset['Validation_method'])), subset['mean'], 1)
    #     p = np.poly1d(z)
    #     plt.plot(subset['Validation_method'], p(range(len(subset['Validation_method']))), linestyle='--', color='gray')
    #
    # plt.xlabel('Nazwa metody walidacji')
    # plt.ylabel('Średnia wartość błędu MAE')
    # plt.title('Wpływ różnych metod optymalizacji na wyniki metod walidacji')
    # plt.legend(title='Metoda walidacji')
    # plt.grid(True)
    # plt.show()

    ######################### BAR PLOT ##############################

