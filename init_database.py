from database import import_sample_data


def main():
    total_rows = import_sample_data(replace=True)
    print(f"Database initialized with {total_rows} crop yield records.")


if __name__ == "__main__":
    main()
