# Projekt: Analiza Strojenia Hiperparametrów i Walidacji Modeli

## Opis projektu

Ten projekt ma na celu porównanie różnych metod strojenia hiperparametrów oraz technik walidacji modeli uczenia maszynowego. Skrypt analizuje zbiory danych z wynikami eksperymentów, generuje wykresy i porównuje wyniki optymalizacji. Przeprowadza również analizę czasu trwania strojenia dla różnych metod walidacyjnych i optymalizacyjnych.

## Struktura projektu

- **`load_all_csv_from_folder(base_path)`**: Funkcja, która ładuje wszystkie pliki CSV z określonego folderu i łączy je w jeden DataFrame.
- **`boxplot_generator(dataset, variable, error, variable_name_for_plot)`**: Generuje wykresy typu boxplot dla wybranej zmiennej (np. metody optymalizacji lub walidacji) oraz wybranej metryki błędu, zbiorczo dla wszystkich modeli i zbiorów danych.
- **`plot_better(data, variable, error, variable_label)`**: Generuje wykres słupkowy pokazujący, jak wyniki modeli zoptymalizowanych wypadają w porównaniu do modeli niezoptymalizowanych (lepsze, takie same, gorsze), zbiorczo dla wszystkich modeli i zbiorów danych.
- **`time_plot(dataset, dataset1, dataset2, variable, variable_name_for_plot)`**: Generuje wykresy pokazujące średni czas trwania procesu strojenia dla różnych metod walidacji i optymalizacji, zbiorczo dla wszystkich modeli i zbiorów danych.
- **`bar_plot(df, variable)`**: Generuje wykresy słupkowe przedstawiające średnią wartość błędu MAE dla różnych metod walidacji i algorytmów regresji, zbiorczo dla wszystkich zbiorów danych.

## Dane wejściowe

Projekt korzysta z plików CSV zapisujących wyniki eksperymentów ze strojeniem hiperparametrów oraz walidacją modeli. Pliki te znajdują się w folderach z prefiksem `ml-tuning-*`.

## Kroki analizy

1. **Ładowanie danych**: Wszystkie pliki CSV z wynikami są ładowane za pomocą funkcji `load_all_csv_from_folder` i łączone w jeden zbiór danych.
2. **Generowanie wykresów**:
   - **Wykresy boxplot** dla metod walidacji i optymalizacji (np. porównanie MAE dla różnych metod).
   - **Wykresy słupkowe** pokazujące, jak różne algorytmy optymalizacji wpływają na wyniki walidacji.
   - **Wykresy czasu strojenia** dla różnych metod walidacyjnych i optymalizacyjnych.
3. **Porównanie wyników**: Skrypt analizuje, które metody optymalizacji dają lepsze wyniki w porównaniu do modeli niezoptymalizowanych.

## Przykład uruchomienia

1. Uruchomienie skryptu głównego:
   ```bash
   python main.py

Wygenerowane zostaną wykresy porównawcze, które zilustrują wyniki dla różnych konfiguracji walidacyjnych i optymalizacyjnych.

## Wymagania
- Python 3.8+
- Biblioteki:
  - pandas
  - numpy
  - matplotlib