import ray

def perform_etl():
    # 1. Inicjalizacja klastra Ray (uruchomi się lokalnie wykorzystując wszystkie rdzenie)
    ray.init(ignore_reinit_error=True)
    

    print("Wczytywanie plików CSV...")
    ratings = ray.data.read_csv("data/ml-latest-small/ratings.csv")
    movies = ray.data.read_csv("data/ml-latest-small/movies.csv")

    print("Czyszczenie danych...")
    
    # Filmy: Odrzucamy wiersze, w których gatunek to "(no genres listed)"
    # Używamy .filter(), który na klastrze wykona się równolegle
    movies_cleaned = movies.filter(lambda row: row["genres"] != "(no genres listed)")
    
    # Oceny: Oceny są w skali od 0.5 do 5.0. 
    # Odrzucamy puste wartości (jeśli by istniały) i usuwamy kolumnę timestamp, by odchudzić zbiór
    ratings_cleaned = (
        ratings
        .drop_columns(["timestamp"])
        .filter(lambda row: row["rating"] is not None)
    )

    # Można też np. znormalizować oceny  za pomocą map:
    # ratings_cleaned = ratings_cleaned.map(lambda row: {**row, "rating_normalized": row["rating"] / 5.0})

    # Zapisujemy do formatu Parquet (świetnie kompresuje dane i wspiera typowanie)
    print("Zapisywanie przetworzonych danych do formatu Parquet...")
    
    # Ray podzieli dane na wiele mniejszych plików Parquet w podanych folderach
    movies_cleaned.write_parquet("data/processed/movies")
    ratings_cleaned.write_parquet("data/processed/ratings")

    print("Proces ETL zakończony sukcesem!")
    
    # Zamykamy sesję Ray
    ray.shutdown()

if __name__ == "__main__":
    perform_etl()