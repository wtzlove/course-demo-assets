from modules.database import initialize_database


if __name__ == "__main__":
    db_path = initialize_database()
    print(f"SQLite database is ready: {db_path}")
