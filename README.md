Инструкция к запуску.
1. Откройте терминал 
2. Переместитесь в удобный вам путь
3. Пропишите строку в терминал `git clone 'https://github.com/OPPs360z/opopopo.git'`
4. Переместитесь в папку opopopo `cd opopopo`
5. Создайте виртуальное окружение `python -m venv myenv` (в Linux/macOS) или `python -m venv venv` (в Windows) (создается виртуальное окружение 1 раз)
6. Активируйте виртуальное окружение `.venv\Scripts\activate.bat` (Windows (Command Prompt)) `source myenv/bin/activate.csh` (macOS (оболочка csh/tcsh)) `source myenv/bin/activate` (Linux, macOS (оболочка bash/zsh))
7. Переместитесь на ветку backend `git checkout backend`
8. Пропишите команду в терминале `pip install -r requirements.txt` (установка зависимых билиотек)
9. Запустить проверку данных: `cd backend/mix && python Metrik.py`
10. Запустить сервер: `cd backend && python main.py`
11. Открыть в браузере `http://127.0.0.1:8000`
12. После использование, дабы прекратить работу с сервером в терминале нажимаете на горячую клавишу CTRL + C
13. Деактивируете виртуальное окружение `.venv\Scripts\activate.bat` (Windows (Command Prompt))
