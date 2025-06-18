import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as ps
    import os
    from pyuca import Collator
    import datetime
    import urllib.request
    return Collator, datetime, mo, os, ps, urllib


@app.cell
def _():
    path = 'C:\\KPI\\Data_Analysis\\DS-labs\\lab2\\datasets\\'
    return (path,)


@app.cell
def _(os, path):
    def list_datasets():
        datasets_list = os.listdir(path)
        datasets_list.sort(key=lambda x: int(x.split("_")[1].lstrip("ID")))
        return datasets_list
    return (list_datasets,)


@app.cell
def _(datetime, path, urllib):
    def get_datasets():
        now = datetime.datetime.now()
        date_and_time = now.strftime("%d-%m-%Y-%H_%M_%S")
        for i in range(1,28):
            url='https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID=' + str(i) + '&year1=1981&year2=2024&type=Mean'
            wp = urllib.request.urlopen(url)
            text = wp.read()
            filename = f'NOAA_ID{str(i)}_{date_and_time}.csv'
            out = open(path + filename,'wb')
            out.write(text)
            out.close()
        print('CSV files imported successfully.')
    return (get_datasets,)


@app.cell
def _(list_datasets, os, path):
    def clear_datasets():
        try:
            for dataset in list_datasets():
                dataset_path = os.path.join(path, dataset)
                if os.path.isfile(dataset_path):
                    os.remove(dataset_path)
            print("All datasets removed successfully.")
        except OSError:
            print("something something error.")
    return (clear_datasets,)


@app.cell
def _(clear_datasets, get_datasets):
    def update_datasets():
        clear_datasets()
        get_datasets()
    return (update_datasets,)


@app.cell
def _():
    # def get_df():
    #     df = {}
    #     for file in list_datasets():
    #         headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI']
    #         test_df = ps.read_csv(path + file, header=1, names=headers, index_col=False)
    #         test_df['ID'] = range(1, len(test_df) + 1)
    #         test_df.set_index('ID', inplace=True)
    #         test_df.drop(len(test_df), inplace=True)
    #         test_df.at[1,'Year'] = test_df.at[2,'Year']
    #         test_df.drop(test_df.loc[test_df['VHI'] == -1].index, inplace=True)
    #         with open(path + file) as f:
    #             key = f.readline().split(" ")[7].rstrip(",") #use oblast name as a key
    #         df[key] = test_df
    #     return df
    return


@app.cell
def _(list_datasets, path):
    def en_keys():
        en_keys= {}
        for file in list_datasets():
            with open(path + file) as f:
                line = f.readline()
                oblID = int(line.split(" ")[6].rstrip(":"))
                oblName = line.split(" ")[7].rstrip(",")
                key = (oblID, oblName)
            en_keys[oblID] = oblName
        en_keys[12] = 'Kiev_City'
        return en_keys
    return (en_keys,)


@app.cell
def _():
    # def ukr_keys():
    #     original_dict = en_keys().copy()
    #     english_to_ukrainian = {
    #         "Cherkasy": "Черкаська область",
    #         "Chernihiv": "Чернігівська область",
    #         "Chernivtsi": "Чернівецька область",
    #         "Crimea": "Республіка Крим",
    #         "Dnipropetrovs'k": "Дніпропетровська область",
    #         "Donets'k": "Донецька область",
    #         "Ivano-Frankivs'k": "Івано-Франківська область",
    #         "Kharkiv": "Харківська область",
    #         "Kherson": "Херсонська область",
    #         "Khmel'nyts'kyy": "Хмельницька область",
    #         "Kiev": "Київська область",
    #         "Kiev_City": "Київ",
    #         "Kirovohrad": "Кіровоградська область",
    #         "Luhans'k": "Луганська область",
    #         "L'viv": "Львівська область",
    #         "Mykolayiv": "Миколаївська область",
    #         "Odessa": "Одеська область",
    #         "Poltava": "Полтавська область",
    #         "Rivne": "Рівненська область",
    #         "Sevastopol'": "Севастополь",
    #         "Sumy": "Сумська область",
    #         "Ternopil'": "Тернопільська область",
    #         "Transcarpathia": "Закарпатська область",
    #         "Vinnytsya": "Вінницька область",
    #         "Volyn": "Волинська область",
    #         "Zaporizhzhya": "Запорізька область",
    #         "Zhytomyr": "Житомирська область"
    #     }

    #     translated = [(id, english_to_ukrainian[name]) for id, name in original_dict.items()]

    #     translated_dict = {id: name for id, name in translated}

    #     return translated_dict
    return


@app.cell
def ukr_keys():
    def ukr_keys():
        translated_dict = {
          1: "Черкаська область",
          2: "Чернігівська область",
          3: "Чернівецька область",
          4: "Республіка Крим",
          5: "Дніпропетровська область",
          6: "Донецька область",
          7: "Івано-Франківська область",
          8: "Харківська область",
          9: "Херсонська область",
          10: "Хмельницька область",
          11: "Київська область",
          12: "Київ",
          13: "Кіровоградська область",
          14: "Луганська область",
          15: "Львівська область",
          16: "Миколаївська область",
          17: "Одеська область",
          18: "Полтавська область",
          19: "Рівненська область",
          20: "Севастополь",
          21: "Сумська область",
          22: "Тернопільська область",
          23: "Закарпатська область",
          24: "Вінницька область",
          25: "Волинська область",
          26: "Запорізька область",
          27: "Житомирська область"
        }

        return translated_dict
    return (ukr_keys,)


@app.cell
def _(en_keys):
    def oblast(oblID):
        oblast = en_keys()[oblID]
        return oblast
    return (oblast,)


@app.cell
def _(oblast):
    def col_name(oblID):
        col_name = str(oblID) + "." + oblast(oblID)
        return col_name
    return (col_name,)


@app.cell
def _(col_name, list_datasets, path, ps):
    def get_df2():
        i = 1
        headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI']
        df = ps.read_csv(path + list_datasets()[0], header=1, names=headers, index_col=False)
        df['ID'] = range(1, len(df) + 1)
        df.set_index('ID', inplace=True)
        df.drop(len(df), inplace=True)
        df.at[1,'Year'] = df.at[2,'Year']
        df.drop(df.loc[df['VHI'] == -1].index, inplace=True)
        df.drop(['SMN', 'SMT', 'VCI', 'TCI'], axis=1, inplace=True)
        df.rename(columns={'VHI': col_name(i)}, inplace=True)
        i += 1

        for file in list_datasets()[1 : ]:
            headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI']
            test_df = ps.read_csv(path + file, header=1, names=headers, index_col=False)
            test_df['ID'] = range(1, len(test_df) + 1)
            test_df.set_index('ID', inplace=True)
            test_df.drop(len(test_df), inplace=True)
            test_df.at[1,'Year'] = test_df.at[2,'Year']
            test_df.drop(test_df.loc[test_df['VHI'] == -1].index, inplace=True)
            test_df.drop(['Year', 'Week', 'SMN', 'SMT', 'VCI'], axis=1, inplace=True)
            test_df
            df[col_name(i)] = test_df['VHI']
            i += 1

        return df
    return (get_df2,)


@app.cell
def _(Collator, ukr_keys):
    def ukr_columns(dataframe):
        ukr_df = dataframe.copy()
        region_columns = ukr_df.columns[2:]

        for i, col in enumerate(region_columns):
            ukr_df.rename(columns={col: ukr_keys()[i+1]}, inplace=True)

        collator = Collator()
        sorted_cols = list(ukr_df.columns[:2]) + sorted(ukr_df.columns[2:], key=collator.sort_key)
        ukr_df=ukr_df[sorted_cols]

        for i, col in enumerate(ukr_df.columns[2:]):
            ukr_df.rename(columns={col: str(i+1) + "." + col}, inplace=True)

        return ukr_df
    return (ukr_columns,)


@app.cell
def index_by_year():
    def index_by_year(dataframe):
        dataframe.set_index('Year', inplace=True)
        return dataframe
    return (index_by_year,)


@app.cell
def get_vhi_by_province_and_year():
    def get_vhi_by_province_and_year(dataframe, province, year):
        # reset index to include Year as a column
        filtered_df = dataframe.loc[year].reset_index()

        result = filtered_df[['Year', 'Week', province]]

        return result
    return (get_vhi_by_province_and_year,)


@app.cell
def get_vhi_statistics(get_vhi_by_province_and_year):
    def get_vhi_statistics(dataframe, province, year):
        eggs = get_vhi_by_province_and_year(dataframe, province, year)
        min = eggs[province].min()
        max = eggs[province].max()
        mean = eggs[province].mean()
        median = eggs[province].median()
        return min, max, mean, median
    return (get_vhi_statistics,)


@app.cell
def get_extreme_drought():
    def get_extreme_drought(dataframe, percentage, vhi_threshold=15):

        region_columns = dataframe.columns[1:]
        total_regions = len(region_columns)
        min_regions = round((percentage / 100) * total_regions)

        # count of regions with VHI less than or equal to vhi_threshold for each row
        drought_count = (dataframe[region_columns] <= vhi_threshold).sum(axis=1)

        # true where count >= min_regions
        condition = drought_count >= min_regions

        # filter the dataframe using the condition
        filtered_dataframe = dataframe[condition]

        return filtered_dataframe
    return (get_extreme_drought,)


@app.cell
def _():
    def filter_vhi_data(df, provinces, start_year, end_year):
        """
        Filters the VHI DataFrame for selected provinces and a specific range of years.

        Parameters:
            df (pd.DataFrame): Input DataFrame with 'Week' column and province columns. 
                               The index of df should be years.
            provinces (list): List of province names (as they appear in the DataFrame headers) to retain.
            start_year (int): The starting year (inclusive) for filtering.
            end_year (int): The ending year (inclusive) for filtering.

        Returns:
            pd.DataFrame: A DataFrame that includes only the 'Week' column and the selected provinces,
                          filtered by the specified year range.
        """
        # filter the df for the specified year range using the index.
        df_year_filtered = df[(df.index >= start_year) & (df.index <= end_year)]

        columns_to_keep = ['Week'] + provinces

        # filter columns based on what exists in the df
        columns_to_keep = [col for col in columns_to_keep if col in df_year_filtered.columns]

        return df_year_filtered[columns_to_keep]
    return (filter_vhi_data,)


@app.cell
def _(get_df2, ukr_columns):
    def task1():
        df = get_df2()
        potuzhni_df = ukr_columns(df)
        print('Columns were ukrainizowani... (opatriocheni? upotuzhneni?)')
        return potuzhni_df.head(3)
    return (task1,)


@app.cell
def task2(get_vhi_by_province_and_year):
    def task2(task3_df, real_province, real_year):
        return get_vhi_by_province_and_year(task3_df, real_province, real_year)
    return (task2,)


@app.cell
def task3(get_vhi_statistics):
    def task3(task3_df, real_province, real_year):
        min, max, mean, median = get_vhi_statistics(task3_df, real_province, real_year)
        stats = {
            'min' : min,
            'max' : max,
            'mean' : mean,
            'median' : median
        }
        return stats
    return (task3,)


@app.cell
def task4(get_extreme_drought):
    def task4(task4_df, percentage):
        return get_extreme_drought(task4_df, percentage)
    return (task4,)


@app.cell
def _(filter_vhi_data):
    def task5(task5_df, regions, year4_start, year4_end):
        return filter_vhi_data(task5_df, regions, year4_start, year4_end)
    return (task5,)


@app.cell
def _(mo):
    mo.md(
        """
        # **Task 0**
        > ## *Load data and stuff*
        """
    )
    return


@app.cell
def _(mo):
    button = mo.ui.run_button(label="Reload Datasets")
    button
    return (button,)


@app.cell
def _(button, update_datasets):
    if button.value:
        update_datasets()
    return


@app.cell
def _(mo):
    mo.md(
        """
        #**Task 1**
        > ##Реалізувати окрему процедуру, яка змінить індекси областей, які використані на порталі NOAA (за англійською абеткою) на наступні, за українською (виключно старі індекси на нові).
        """
    )
    return


@app.cell
def _(task1):
    task1()
    return


@app.cell
def _(mo):
    mo.md(
        """
        #**Task 2**
        >##Ряд VHI для області за вказаний рік
        """
    )
    return


@app.cell
def _(mo):
    ukrainizacia2 = mo.ui.checkbox(label="Check if you want to use ukrainian labels for columns ")
    ukrainizacia2
    return (ukrainizacia2,)


@app.cell
def _(get_df2, index_by_year, ukr_columns, ukrainizacia2):
    if (ukrainizacia2.value==True):
        task2_df = index_by_year(ukr_columns(get_df2()))
    else:
        task2_df = index_by_year(get_df2())
    return (task2_df,)


@app.cell
def _(mo, task2_df):
    province2 = mo.ui.dropdown(options=task2_df.columns[1:], value=task2_df.columns[1:][0], label="Select province", searchable=True)
    province2
    return (province2,)


@app.cell
def _(province2):
    real_province2 = str(province2.value)
    return (real_province2,)


@app.cell
def _(mo):
    year2 = mo.ui.slider(start=1982, stop=2024, step=1)
    year2
    return (year2,)


@app.cell
def _(year2):
    real_year2 = str(year2.value)
    return (real_year2,)


@app.cell
def _(real_province2, real_year2, task2, task2_df):
    task2(task2_df, real_province2, real_year2)
    return


@app.cell
def _(mo):
    mo.md(
        """
        #**Task 3**
        >##Пошук екстремумів (min та max) для вказаних областей та років, середнього, медіани;
        """
    )
    return


@app.cell
def _(mo):
    ukrainizacia3 = mo.ui.checkbox(label="Check if you want to use ukrainian labels for columns ")
    ukrainizacia3
    return (ukrainizacia3,)


@app.cell
def _(get_df2, index_by_year, ukr_columns, ukrainizacia3):
    if (ukrainizacia3.value==True):
        task3_df = index_by_year(ukr_columns(get_df2()))
    else:
        task3_df = index_by_year(get_df2())
    return (task3_df,)


@app.cell
def _(mo, task3_df):
    province3 = mo.ui.dropdown(options=task3_df.columns[1:], value=task3_df.columns[1:][0], label="Select province", searchable=True)
    province3
    return (province3,)


@app.cell
def _(province3):
    real_province3 = str(province3.value)
    return (real_province3,)


@app.cell
def _(mo):
    year3 = mo.ui.slider(start=1982, stop=2024, step=1, label='Select year')
    year3
    return (year3,)


@app.cell
def _(year3):
    real_year3 = str(year3.value)
    return (real_year3,)


@app.cell
def _(real_province3, real_year3, task3, task3_df):
    task3(task3_df, real_province3, real_year3)
    return


@app.cell
def _(mo):
    mo.md(
        """
        #**Task 4**
        >##Для всього набору даних виявити роки, протягом яких екстремальні посухи торкнулися більше вказаного відсотка областей по Україні (20% областей - 5 областей з 25). Повернути роки, назви областей з екстремальними посухами та значення VHI;
        """
    )
    return


@app.cell
def _(mo):
    ukrainizacia4 = mo.ui.checkbox(label="Check if you want to use ukrainian labels for columns ")
    ukrainizacia4
    return (ukrainizacia4,)


@app.cell
def _(get_df2, index_by_year, ukr_columns, ukrainizacia4):
    if (ukrainizacia4.value==True):
        task4_df = index_by_year(ukr_columns(get_df2()))
    else:
        task4_df = index_by_year(get_df2())
    return (task4_df,)


@app.cell
def _(mo):
    percentage = mo.ui.slider(start=0, stop=100, step=1, label="something something percentage")
    percentage
    return (percentage,)


@app.cell
def _(percentage, task4, task4_df):
    task4(task4_df, percentage.value)
    return


@app.cell
def _(mo):
    mo.md(
        """
        #**Task 5** *(well, this wasn't supposed to be an exactly 5th task but...)*
        >##Ряд VHI за вказаний діапазон років для вказаних областей
        """
    )
    return


@app.cell
def _(mo):
    ukrainizacia5 = mo.ui.checkbox(label="Check if you want to use ukrainian labels for columns ")
    ukrainizacia5
    return (ukrainizacia5,)


@app.cell
def _(get_df2, index_by_year, ukr_columns, ukrainizacia5):
    if (ukrainizacia5.value==True):
        task5_df = index_by_year(ukr_columns(get_df2()))
    else:
        task5_df = index_by_year(get_df2())
    return (task5_df,)


@app.cell
def _(mo, task5_df):
    multiselect = mo.ui.multiselect(options=task5_df.columns[1:], label='Select regions')
    return (multiselect,)


@app.cell
def _(mo, multiselect):
    mo.hstack([multiselect, mo.md(f"Currently selected regions: {multiselect.value}")])
    return


@app.cell
def _(multiselect):
    regions = multiselect.value
    return (regions,)


@app.cell
def _(mo):
    year4_start = mo.ui.slider(start=1982, stop=2024, step=1, label='Select starting year')
    year4_start
    return (year4_start,)


@app.cell
def _(year4_start):
    real_year4_start = str(year4_start.value)
    return (real_year4_start,)


@app.cell
def _(mo, year4_start):
    year4_end = mo.ui.slider(start=year4_start.value, stop=2024, step=1, label='Select ending year')
    year4_end
    return (year4_end,)


@app.cell
def _(year4_end):
    real_year4_end = str(year4_end.value)
    return (real_year4_end,)


@app.cell
def _(real_year4_end, real_year4_start, regions, task5, task5_df):
    task5(task5_df, regions, real_year4_start, real_year4_end)
    return


if __name__ == "__main__":
    app.run()
