import sqlite3

# Параметры базы данных
DB_PATH = "C:/Users/12967/diplom.db"
#TABLE_NAME = "ft_history_animal"
TABLE_NAME = "ft_boundary"


def execute_query(query, params=None, fetch=False):
    """Выполняет SQL-запрос к базе данных."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)

    if fetch:
        results = cursor.fetchall()
        conn.close()
        return results
    else:
        conn.commit()
        conn.close()

def get_all_rows():
    """Выводит все строки из таблицы."""
    query = f"SELECT * FROM {TABLE_NAME}"
    rows = execute_query(query, fetch=True)
    
    print("📋 Все записи в таблице:")
    for row in rows:
        print(row)

def delete_all_rows():
    """Удаляет все строки из таблицы."""
    query = f"DELETE FROM {TABLE_NAME}"
    execute_query(query)
    print("🗑️ Все записи удалены.")

def get_row_count():
    """Получает количество записей в таблице."""
    query = f"SELECT COUNT(*) FROM {TABLE_NAME}"
    count = execute_query(query, fetch=True)[0][0]
    
    print(f"📊 Количество записей: {count}")

if __name__ == "__main__":
    while True:
        print("\nВыберите действие:")
        print("1. Показать все записи")
        print("2. Удалить все записи")
        print("3. Показать количество записей")
        print("4. Выйти")

        choice = input("Введите номер действия: ")

        if choice == "1":
            get_all_rows()
        elif choice == "2":
            delete_all_rows()
        elif choice == "3":
            get_row_count()
        elif choice == "4":
            print("👋 Выход...")
            break
        else:
            print("❌ Неверный ввод, попробуйте снова.")

