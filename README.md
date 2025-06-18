# Репозиторій з лабами із навчальної дисципліни: "Засоби підготовки та аналізу даних"
## Як запустити lab3?

Виконайте команду `git clone https://github.com/dvdyo/DS-labs.git`, або завантажте окрему директорію за допомогою розширення для браузера [GitZip](https://chromewebstore.google.com/detail/gitzip-for-github/ffabmkklhbepgcgfonabamgnfafbdlkn?pli=1).

`cd` в директорію **/lab3** та встановіть залежності, що вказані в **.toml** файлі.

Рекомендується це зробити за допомогою пакетного менеджера [**uv**](https://github.com/astral-sh/uv). Виконайте наступну команду, знаходячись у дирескорії **/lab3**:

`uv sync`

Ця команда встановить усі залежності та створить venv. Активуєте .venv за допомогою команди:

`.venv\Scripts\activate`

Потім, виконайте команду `streamlit run app.py`

Все повинно працювати.
