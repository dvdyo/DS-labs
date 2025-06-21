# Репозиторій з лабами із навчальної дисципліни: "Засоби підготовки та аналізу даних"

## Table of Contents
- [Загальна настройка](#загальна-настройка)
- [Lab 3: Interactive Dashboard](#lab-3-interactive-dashboard)
- [Lab 4: Data Analysis](#lab-4-data-analysis)
- [Lab 5: Data Processing](#lab-5-data-processing)
- [Lab 6: Advanced Analytics](#lab-6-advanced-analytics)

## Загальна настройка

### Крок 1: Клонування репозиторію
```bash
git clone https://github.com/dvdyo/DS-labs.git
```
*Альтернативно*: завантажте окрему директорію за допомогою розширення [GitZip](https://chromewebstore.google.com/detail/gitzip-for-github/ffabmkklhbepgcgfonabamgnfafbdlkn?pli=1).

### Крок 2: Встановлення залежностей
Рекомендується використовувати пакетний менеджер [**uv**](https://github.com/astral-sh/uv):

```bash
cd /lab{X}  # замініть {X} на номер лаби
uv sync
```

### Крок 3: Активація віртуального середовища
```bash
.venv\Scripts\activate  # Windows
# або
source .venv/bin/activate  # Linux/macOS
```

## Lab 3: Interactive Dashboard
**Технологія**: Streamlit

**Запуск**:
```bash
cd lab3
# Виконайте загальну настройку (кроки 2-3 вище)
streamlit run app.py
```

**Результат**: Відкриється веб-інтерфейс за адресою `http://localhost:8501`

## Lab 4: Data Analysis
**Технологія**: Marimo

**Запуск**:
```bash
cd lab4
# Виконайте загальну настройку (кроки 2-3 вище)
marimo run main.py
```

## Lab 5: Data Processing
**Технологія**: Python Script

**Запуск**:
```bash
cd lab5
# Виконайте загальну настройку (кроки 2-3 вище)
python main.py
```

## Lab 6: Advanced Analytics
**Технологія**: Python Script

**Запуск**:
```bash
cd lab6
# Виконайте загальну настройку (кроки 2-3 вище)
python main.py
```
