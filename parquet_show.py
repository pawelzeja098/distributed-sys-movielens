import pandas as pd

# Podajemy ścieżkę do folderu wygenerowanego przez Ray
# Pandas automatycznie sklei wszystkie małe pliki Parquet w tym folderze w jedną tabelę
movies_df = pd.read_parquet("data/processed/ratings")

# Wyświetlamy 5 pierwszych wierszy
print(movies_df.head())