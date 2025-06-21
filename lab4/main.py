import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import timeit
    from ucimlrepo import fetch_ucirepo
    import warnings
    from pandas.errors import DtypeWarning
    return DtypeWarning, fetch_ucirepo, mo, np, pd, timeit, warnings


@app.cell
def _(DtypeWarning, fetch_ucirepo, np, pd, warnings):
    print("Fetching data from UCI repository...")

    # The ucimlrepo loader is lazy and triggers a DtypeWarning
    # Suppressin' it because our manual type conversion below handles the root cause
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DtypeWarning)
        repo = fetch_ucirepo(id=235)

    features = repo.data.features
    targets = repo.data.targets

    df_pandas = pd.concat([features, targets], axis=1)

    numeric_fields = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2",
        "Sub_metering_3"
    ]

    # Convert all '?' strings to numeric NaNs
    for col in numeric_fields:
        df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce')

    original_column_order = [
        "Date", "Time", "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2",
        "Sub_metering_3"
    ]
    df_pandas = df_pandas[original_column_order]
    print("DataFrame reconstructed successfully.")

    print("\nProcessing data with pandas...")
    print(f"Original pandas shape: {df_pandas.shape}")
    print(f"Missing values per column:\n{df_pandas.isnull().sum()}")
    df_pandas_clean = df_pandas.dropna()
    print(f"Cleaned pandas shape: {df_pandas_clean.shape}")

    print("\nProcessing data with numpy...")
    df_pandas['Date'] = df_pandas['Date'].astype('U10')
    df_pandas['Time'] = df_pandas['Time'].astype('U8')
    df_numpy = df_pandas.to_records(index=False)

    print(f"Original numpy shape: {df_numpy.shape}")
    mask = np.ones(len(df_numpy), dtype=bool)
    for field in numeric_fields:
        mask &= ~np.isnan(df_numpy[field])
    df_numpy_clean = df_numpy[mask]
    print(f"Cleaned numpy shape: {df_numpy_clean.shape}")

    print(f"\nData cleaning summary:")
    print(f"Pandas: {len(df_pandas)} -> {len(df_pandas_clean)} rows")
    print(f"Numpy: {len(df_numpy)} -> {len(df_numpy_clean)} rows")
    return df_numpy_clean, df_pandas_clean


@app.cell
def _(df_numpy_clean, df_pandas_clean, timeit):
    print("=" * 50)
    print("TASK 1: Select records where global active power > 5 kW")
    print("=" * 50)

    def task1_pandas():
        return df_pandas_clean[df_pandas_clean['Global_active_power'] > 5]

    def task1_numpy():
        return df_numpy_clean[df_numpy_clean['Global_active_power'] > 5]

    time_pandas = timeit.timeit(task1_pandas, number=1)
    result_pandas = task1_pandas()

    time_numpy = timeit.timeit(task1_numpy, number=1)
    result_numpy = task1_numpy()

    print(f"Pandas results: {len(result_pandas)} records found")
    print(f"Pandas execution time: {time_pandas:.6f} seconds")

    print(f"\nNumpy results: {len(result_numpy)} records found")
    print(f"Numpy execution time: {time_numpy:.6f} seconds")

    print(f"\nPerformance comparison:")
    if time_pandas < time_numpy:
        ratio = time_numpy / time_pandas
        print(f"Pandas is {ratio:.2f}x faster than Numpy")
    else:
        ratio = time_pandas / time_numpy
        print(f"Numpy is {ratio:.2f}x faster than Pandas")

    print(f"\nFirst 3 pandas results:")
    print(result_pandas[['Date', 'Time', 'Global_active_power']].head(3))

    print(f"\nFirst 3 numpy results:")
    for i in range(3):
        print(f"Date: {result_numpy['Date'][i]}, Time: {result_numpy['Time'][i]}, Power: {result_numpy['Global_active_power'][i]}")
    return


@app.cell
def _(df_numpy_clean, df_pandas_clean, timeit):
    print("=" * 50)
    print("TASK 2: Select records where voltage > 235 V")
    print("=" * 50)

    def task2_pandas():
        return df_pandas_clean[df_pandas_clean['Voltage'] > 235]

    def task2_numpy():
        return df_numpy_clean[df_numpy_clean['Voltage'] > 235]

    time_pandas_t2 = timeit.timeit(task2_pandas, number=1)
    result_pandas_t2 = task2_pandas()

    time_numpy_t2 = timeit.timeit(task2_numpy, number=1)
    result_numpy_t2 = task2_numpy()

    print(f"Pandas results: {len(result_pandas_t2)} records found")
    print(f"Pandas execution time: {time_pandas_t2:.6f} seconds")

    print(f"\nNumpy results: {len(result_numpy_t2)} records found")
    print(f"Numpy execution time: {time_numpy_t2:.6f} seconds")

    print(f"\nPerformance comparison:")
    if time_pandas_t2 < time_numpy_t2:
        ratio_t2 = time_numpy_t2 / time_pandas_t2
        print(f"Pandas is {ratio_t2:.2f}x faster than Numpy")
    else:
        ratio_t2 = time_pandas_t2 / time_numpy_t2
        print(f"Numpy is {ratio_t2:.2f}x faster than Pandas")

    print(f"\nFirst 3 pandas results:")
    print(result_pandas_t2[['Date', 'Time', 'Voltage']].head(3))

    print(f"\nFirst 3 numpy results:")
    for j in range(3):
        print(f"Date: {result_numpy_t2['Date'][j]}, Time: {result_numpy_t2['Time'][j]}, Voltage: {result_numpy_t2['Voltage'][j]}")
    return


@app.cell
def _(df_numpy_clean, df_pandas_clean, timeit):
    print("=" * 50)
    print("TASK 3: Select records where current intensity is 19-20 A,")
    print("then find where sub_metering_2 > sub_metering_3")
    print("=" * 50)

    def task3_pandas():
        # Chain two filters: first by intensity, then by sub-metering comparison
        step1 = df_pandas_clean[(df_pandas_clean['Global_intensity'] >= 19) & 
                               (df_pandas_clean['Global_intensity'] <= 20)]
        step2 = step1[step1['Sub_metering_2'] > step1['Sub_metering_3']]
        return step1, step2

    def task3_numpy():
        # Same two-step filter logic
        mask1 = (df_numpy_clean['Global_intensity'] >= 19) & (df_numpy_clean['Global_intensity'] <= 20)
        step1 = df_numpy_clean[mask1]
        mask2 = step1['Sub_metering_2'] > step1['Sub_metering_3']
        step2 = step1[mask2]
        return step1, step2

    time_pandas_t3 = timeit.timeit(lambda: task3_pandas()[1], number=1)
    result_pandas_step1_t3, result_pandas_step2_t3 = task3_pandas()

    time_numpy_t3 = timeit.timeit(lambda: task3_numpy()[1], number=1)
    result_numpy_step1_t3, result_numpy_step2_t3 = task3_numpy()

    print(f"Pandas results:")
    print(f"  Step 1 (19-20 A): {len(result_pandas_step1_t3)} records")
    print(f"  Step 2 (sub_metering_2 > sub_metering_3): {len(result_pandas_step2_t3)} records")
    print(f"  Pandas execution time: {time_pandas_t3:.6f} seconds")

    print(f"\nNumpy results:")
    print(f"  Step 1 (19-20 A): {len(result_numpy_step1_t3)} records")
    print(f"  Step 2 (sub_metering_2 > sub_metering_3): {len(result_numpy_step2_t3)} records")
    print(f"  Numpy execution time: {time_numpy_t3:.6f} seconds")

    print(f"\nPerformance comparison:")
    if time_pandas_t3 < time_numpy_t3:
        ratio_t3 = time_numpy_t3 / time_pandas_t3
        print(f"Pandas is {ratio_t3:.2f}x faster than Numpy")
    else:
        ratio_t3 = time_pandas_t3 / time_numpy_t3
        print(f"Numpy is {ratio_t3:.2f}x faster than Pandas")

    print(f"\nFirst 3 pandas final results:")
    print(result_pandas_step2_t3[['Date', 'Time', 'Global_intensity', 'Sub_metering_2', 'Sub_metering_3']].head(3))

    print(f"\nFirst 3 numpy final results:")
    for k in range(min(3, len(result_numpy_step2_t3))):
        print(f"Date: {result_numpy_step2_t3['Date'][k]}, Time: {result_numpy_step2_t3['Time'][k]}, "
              f"Intensity: {result_numpy_step2_t3['Global_intensity'][k]}, "
              f"Sub2: {result_numpy_step2_t3['Sub_metering_2'][k]}, Sub3: {result_numpy_step2_t3['Sub_metering_3'][k]}")
    return


@app.cell
def _(df_numpy_clean, df_pandas_clean, np, timeit):
    print("=" * 50)
    print("TASK 4: Randomly select 500,000 records and calculate mean values")
    print("for all 3 sub-metering groups")
    print("=" * 50)

    # Use a fixed seed for reproducible random sampling
    np.random.seed(42)

    def task4_pandas():
        sample = df_pandas_clean.sample(n=500000, replace=False, random_state=42)
        means = sample[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
        return sample, means

    def task4_numpy():
        indices = np.random.choice(len(df_numpy_clean), size=500000, replace=False)
        sample = df_numpy_clean[indices]
        mean1 = np.mean(sample['Sub_metering_1'])
        mean2 = np.mean(sample['Sub_metering_2'])
        mean3 = np.mean(sample['Sub_metering_3'])
        return sample, [mean1, mean2, mean3]

    time_pandas_t4 = timeit.timeit(lambda: task4_pandas()[1], number=1)
    sample_pandas_t4, means_pandas_t4 = task4_pandas()

    np.random.seed(42)
    time_numpy_t4 = timeit.timeit(lambda: task4_numpy()[1], number=1)
    sample_numpy_t4, means_numpy_t4 = task4_numpy()

    print(f"Pandas results:")
    print(f"  Sample size: {len(sample_pandas_t4)} records")
    print(f"  Sub_metering_1 mean: {means_pandas_t4['Sub_metering_1']:.4f}")
    print(f"  Sub_metering_2 mean: {means_pandas_t4['Sub_metering_2']:.4f}")
    print(f"  Sub_metering_3 mean: {means_pandas_t4['Sub_metering_3']:.4f}")
    print(f"  Pandas execution time: {time_pandas_t4:.6f} seconds")

    print(f"\nNumpy results:")
    print(f"  Sample size: {len(sample_numpy_t4)} records")
    print(f"  Sub_metering_1 mean: {means_numpy_t4[0]:.4f}")
    print(f"  Sub_metering_2 mean: {means_numpy_t4[1]:.4f}")
    print(f"  Sub_metering_3 mean: {means_numpy_t4[2]:.4f}")
    print(f"  Numpy execution time: {time_numpy_t4:.6f} seconds")

    print(f"\nPerformance comparison:")
    if time_pandas_t4 < time_numpy_t4:
        ratio_t4 = time_numpy_t4 / time_pandas_t4
        print(f"Pandas is {ratio_t4:.2f}x faster than Numpy")
    else:
        ratio_t4 = time_pandas_t4 / time_numpy_t4
        print(f"Numpy is {ratio_t4:.2f}x faster than Pandas")

    print(f"\nFirst 3 pandas sampled records:")
    print(sample_pandas_t4[['Date', 'Time', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].head(3))

    print(f"\nFirst 3 numpy sampled records:")
    for m in range(3):
        print(f"Date: {sample_numpy_t4['Date'][m]}, Time: {sample_numpy_t4['Time'][m]}, "
              f"Sub1: {sample_numpy_t4['Sub_metering_1'][m]}, Sub2: {sample_numpy_t4['Sub_metering_2'][m]}, "
              f"Sub3: {sample_numpy_t4['Sub_metering_3'][m]}")
    return


@app.cell
def _(df_numpy_clean, df_pandas_clean, np, pd, timeit):
    print("=" * 70)
    print("TASK 5: Complex filtering - after 18:00, >6kW, sub_metering_2 largest,")
    print("then every 3rd from first half + every 4th from second half")
    print("=" * 70)

    def task5_pandas():
        after_6pm = df_pandas_clean[df_pandas_clean['Time'] >= '18:00:00']
        high_consumption = after_6pm[after_6pm['Global_active_power'] > 6]
        sub2_largest = high_consumption[
            (high_consumption['Sub_metering_2'] >= high_consumption['Sub_metering_1']) &
            (high_consumption['Sub_metering_2'] >= high_consumption['Sub_metering_3'])
        ]

        total_records = len(sub2_largest)
        # Handle case with no matching records
        if total_records == 0:
            return sub2_largest, sub2_largest

        half_point = total_records // 2
        first_half = sub2_largest.iloc[:half_point]
        second_half = sub2_largest.iloc[half_point:]

        # Sample every 3rd record from the first half
        first_half_sampled = first_half.iloc[2::3]
        # Sample every 4th record from the second half
        second_half_sampled = second_half.iloc[3::4]

        final_result = pd.concat([first_half_sampled, second_half_sampled], ignore_index=True)
        return sub2_largest, final_result

    def task5_numpy():
        time_mask = df_numpy_clean['Time'] >= '18:00:00'
        after_6pm = df_numpy_clean[time_mask]

        power_mask = after_6pm['Global_active_power'] > 6
        high_consumption = after_6pm[power_mask]

        sub2_largest_mask = (
            (high_consumption['Sub_metering_2'] >= high_consumption['Sub_metering_1']) &
            (high_consumption['Sub_metering_2'] >= high_consumption['Sub_metering_3'])
        )
        sub2_largest = high_consumption[sub2_largest_mask]

        total_records = len(sub2_largest)
        # Handle case with no matching records
        if total_records == 0:
            return sub2_largest, sub2_largest

        half_point = total_records // 2
        first_half = sub2_largest[:half_point]
        second_half = sub2_largest[half_point:]

        # Sample every 3rd record from the first half
        first_half_sampled = first_half[2::3]
        # Sample every 4th record from the second half
        second_half_sampled = second_half[3::4]

        final_result = np.concatenate([first_half_sampled, second_half_sampled])
        return sub2_largest, final_result

    time_pandas_t5 = timeit.timeit(lambda: task5_pandas()[1], number=1)
    intermediate_pandas_t5, final_pandas_t5 = task5_pandas()

    time_numpy_t5 = timeit.timeit(lambda: task5_numpy()[1], number=1)
    intermediate_numpy_t5, final_numpy_t5 = task5_numpy()

    print(f"Pandas results:")
    print(f"  Intermediate results (after all filters): {len(intermediate_pandas_t5)} records")
    print(f"  Final sampled results: {len(final_pandas_t5)} records")
    print(f"  Pandas execution time: {time_pandas_t5:.6f} seconds")

    print(f"\nNumpy results:")
    print(f"  Intermediate results (after all filters): {len(intermediate_numpy_t5)} records")
    print(f"  Final sampled results: {len(final_numpy_t5)} records")
    print(f"  Numpy execution time: {time_numpy_t5:.6f} seconds")

    print(f"\nPerformance comparison:")
    if time_pandas_t5 < time_numpy_t5:
        ratio_t5 = time_numpy_t5 / time_pandas_t5
        print(f"Pandas is {ratio_t5:.2f}x faster than Numpy")
    else:
        ratio_t5 = time_pandas_t5 / time_numpy_t5
        print(f"Numpy is {ratio_t5:.2f}x faster than Pandas")

    if len(final_pandas_t5) > 0:
        print(f"\nFirst 3 pandas final results:")
        print(final_pandas_t5[['Date', 'Time', 'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].head(3))
    else:
        print(f"\nNo pandas results found matching all criteria")

    if len(final_numpy_t5) > 0:
        print(f"\nFirst 3 numpy final results:")
        for n in range(min(3, len(final_numpy_t5))):
            print(f"Date: {final_numpy_t5['Date'][n]}, Time: {final_numpy_t5['Time'][n]}, "
                  f"Power: {final_numpy_t5['Global_active_power'][n]}, "
                  f"Sub1: {final_numpy_t5['Sub_metering_1'][n]}, Sub2: {final_numpy_t5['Sub_metering_2'][n]}, "
                  f"Sub3: {final_numpy_t5['Sub_metering_3'][n]}")
    else:
        print(f"\nNo numpy results found matching all criteria")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Висновки до лабораторної роботи №4
    ## Структури для роботи з великими обсягами даних у Python

    ### Результати вимірювання часу виконання

    | Завдання | Pandas (сек) | NumPy (сек) | Переможець | Коефіцієнт |
    |----------|--------------|-------------|------------|------------|
    | 1. Фільтрація потужності > 5 кВт | 0.009411 | 0.024173 | Pandas | 2.57x |
    | 2. Фільтрація напруги > 235 В | 0.121248 | 0.144495 | Pandas | 1.19x |
    | 3. Складна фільтрація (інтенсивність + підлічильники) | 0.009109 | 0.023421 | Pandas | 2.57x |
    | 4. Випадкова вибірка 500,000 записів | 0.188926 | 0.207076 | Pandas | 1.10x |
    | 5. Комплексна багатоетапна обробка | 0.139466 | 0.094223 | NumPy | 1.48x |

    ### Оцінка зручності виконання операцій (5-бальна шкала)

    #### >>>**Pandas DataFrame**

    **Простота синтаксису:** 3.4/5 - відносно інтуїтивні методи `.sample()`, `.dropna()`, логічна індексація, але я всеодно не можу обійтися без допомоги інтернету та нейромереж, якщо заглиблюватися у решту синтаксису.

    **Читабельність коду:** 3.9/5 - код можливо розуміти та підтримувати

    **Гнучкість операцій:** 5/5 - вбудовані методи для більшості завдань (але завжди доводиться гуглити, щоб про них дізнатися)

    **Продуктивність:** 4/5 для комплексних, 5/5 для решти - швидше для простих і середніх за складністю операцій

    #### >>**NumPy Array**

    **Простота синтаксису:** 3/5 - потребує більше "ручного" написання коду, але загалом часто схожий за своєю суттю

    **Читабельність коду:** 3.5/5 - більш, так би мовити, вербозний код, особливо для певних операцій

    **Гнучкість операцій:** 4/5 - потужні можливості, але реалізація інколи потребує трохи більше думання

    **Продуктивність:** 5/5 для комплексних операцій, 3/5 для менш комлексних - ефективніше, але зазвичай лише для комплексних багатоетапних операцій

    ### Загальні висновки

       - Різниця у продуктивності залежить від типу операції
       - Pandas переважає у простоті використання
       - NumPy ефективніше для складних алгоритмічних задач
       - Обидві структури забезпечують однакові результати обчислень

    Для більшості задач аналізу даних варто починати з Pandas через його зручність, переходячи до NumPy лише у випадках, коли потрібна максимальна продуктивність для складних операцій. Проте, універсального лідера не існує.
    """
    )
    return


if __name__ == "__main__":
    app.run()
