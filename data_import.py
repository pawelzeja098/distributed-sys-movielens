import urllib.request
import zipfile
import os

def download_movielens_data(size="small"):
    """
    Pobiera zbiór danych MovieLens ze strony GroupLens.
    
    Parametry:
    size (str): "small" pobiera 'ml-latest-small.zip', 
                "full" pobiera 'ml-latest.zip'.
    """
    # Ustalanie nazwy pliku na podstawie wyboru
    if size == "small":
        zip_filename = "ml-latest-small.zip"
    elif size == "full":
        zip_filename = "ml-latest.zip"
    else:
        print("Błąd: Nieznany rozmiar. Wybierz 'small' lub 'full'.")
        return

    # Budowanie docelowego URL i ścieżek
    url = f"https://files.grouplens.org/datasets/movielens/{zip_filename}"
    extract_dir = "./data"

    # Tworzenie katalogu docelowego, jeśli nie istnieje
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Rozpoczynam pobieranie zbioru '{size}' z: {url}...")
    try:
        # Pobieranie pliku
        urllib.request.urlretrieve(url, zip_filename)
        print("Pobieranie zakończone sukcesem!")
    except Exception as e:
        print(f"Wystąpił błąd podczas pobierania: {e}")
        return

    print(f"Rozpakowywanie archiwum do folderu: {extract_dir}...")
    try:
        # Rozpakowywanie zawartości
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Rozpakowywanie zakończone.")
    except Exception as e:
        print(f"Wystąpił błąd podczas rozpakowywania: {e}")
        return
    finally:
        # Sprzątanie pobranego archiwum zip
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            print("Tymczasowy plik ZIP został usunięty.")

    # Archiwa GroupLens rozpakowują się do podfolderu o nazwie pliku zip (bez rozszerzenia)
    final_folder = os.path.join(extract_dir, zip_filename.replace('.zip', ''))
    print(f"\nGotowe! Pliki CSV znajdują się w folderze: {final_folder}")


if __name__ == "__main__":
    # Domyślnie pobieramy mały zbiór danych do testów.
    # Wystarczy zmienić argument na size="full", aby pobrać cały dataset.
    download_movielens_data(size="small")