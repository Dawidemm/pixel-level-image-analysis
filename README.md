# pixel-level-image-analysis

## README

### Opis plików z kodem:

1. **train_pipeline_mnist.py**

   - **Opis:**
   
     Plik `train_pipeline_mnist.py` zawiera skrypt do trenowania modelu przy użyciu zbioru danych MNIST. Skrypt wykorzystuje własny zbiór danych MNIST 8x8, importowany przez klasę (`myDataset`), a także implementacje modelu autoenkodera LBAE (`lbae.LBAE`) i modelu RBM (`rbm.RBM`). Całość jest zintegrowana w strukturę potoku (`pipeline.Pipeline`), który obejmuje trenowanie autoenkodera, trenowanie RBM i opcjonalnie trenowanie klasyfikatora.

   - **Parametry konfiguracyjne:**
     - `NUM_VISIBLE`: Liczba widocznych jednostek w autoenkoderze.
     - `NUM_HIDDEN`: Liczba ukrytych jednostek w RBM.
     - `MAX_EPOCHS`: Maksymalna liczba epok trenowania.
     - `RBM_STEPS`: Liczba kroków trenowania RBM.
     - `BATCH_SIZE`: Rozmiar paczki danych używanej w trakcie trenowania.

   - **Przykład użycia:**
     ```python
     python train_pipeline_mnist.py
     ```

2. **mnist_eval.py**

   - **Opis:**
   
     Plik `mnist_eval.py` zawiera skrypt do oceny wydajności wcześniej wytrenowanego modelu autoenkodera LBAE na zbiorze testowym MNIST. Skrypt korzysta z wcześniej wytrenowanego autoenkodera (`LBAE.load_from_checkpoint`) oraz implementacji RBM (`rbm.RBM`). Oceniana jest średnia odległość euklidesowa pomiędzy oryginalnymi obrazami a ich rekonstrukcjami, a także wyznaczany jest najlepszy próg binarizacji w kontekście miary zgodności Rand (Rand score).

   - **Przykład użycia:**
     ```python
     python mnist_eval.py
     ```

   - **Wymagania przed uruchomieniem:**
     - Model autoenkodera LBAE musi być wcześniej wytrenowany i dostępny pod odpowiednią ścieżką (w kodzie źródłowym podane ścieżki dostępu).

   - **Uwagi dotyczące konfiguracji:**
     - Zmienna `with_best` w funkcji `try_thresholds` decyduje, czy próbować różne progi binarizacji czy też użyć ustalonego najlepszego progu.

   - **Przykład użycia:**
     ```python
     python mnist_eval.py
     ```

   - **Wymagania przed uruchomieniem:**
     - Po wytrenowaniu modelu za pomocą `train_pipeline_mnist.py`, znajdź pliki `.ckpt` i `.yaml` w folderze `lightning_logs`, który został wygenerowany podczas treningu sieci. Skopiuj pełną ścieżkę do tych plików i zastąp odpowiednie ścieżki w kodzie źródłowym pliku `mnist_eval.py`.

   - **Uwagi dotyczące konfiguracji:**
     - Zmienna `with_best` w funkcji `try_thresholds` decyduje, czy próbować różne progi binarizacji czy też użyć ustalonego najlepszego progu.

   - **Przykład użycia:**
     ```python
     python mnist_eval.py
     ```
