# FootballScraper

Aplikacja do przewidywania **łącznej liczby bramek** w pojedynczym meczu piłkarskim oraz klasyfikacji **over/under 2.5 gola**. Pobiera dane historyczne z API [football-data.org](https://www.football-data.org/) (v4), buduje agregaty cech (forma, dom/wyjazd, head-to-head, siła rywali) i porównuje wiele modeli regresji (Linear, Polynomial+Ridge, DecisionTree, RandomForest, GradientBoosting, XGBoost, LightGBM, Poisson, NeuralNetwork) wraz z klasyfikatorem XGBoost dla over/under 2.5.

> Pełny opis stanu projektu i propozycje rozwoju znajdziesz w pliku [`KIERUNKI_ROZWOJU.md`](./KIERUNKI_ROZWOJU.md).

---

## Wymagania

- Python 3.10+ (zalecane 3.11)
- Klucz API z [football-data.org](https://www.football-data.org/client/register) (darmowy plan: 10 req/min)

## Instalacja

```bash
python -m venv .venv
.\.venv\Scripts\activate            # PowerShell na Windows
# source .venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

## Konfiguracja klucza API

Utwórz plik `config.cfg` w katalogu głównym projektu (plik jest w `.gitignore` i **nie trafi do repo**):

```ini
[KEYS]
API_KEY = twoj_klucz_api_z_football_data_org
```

---

## Sposób użycia

Pipeline składa się z dwóch kroków: **(1) pobranie statystyk meczu do JSON-a** → **(2) trening modeli i predykcja**.

### Krok 1. Pobranie statystyk dla meczu

```bash
python football-data.py <MATCH_ID> [<output_json>]
```

- `<MATCH_ID>` – identyfikator meczu w football-data.org (typowo z `football-data_getFutureMatchesPerCompetition.py`, patrz niżej)
- `<output_json>` – **opcjonalny** ścieżka pliku wyjściowego. Bez tego argumentu skrypt zapisze do pliku wskazanego przez `FOOTBALL_STATS_FILE` (env var) lub do `matchStats/matchStatsWesthamCrystal.json` (default). Folder zostanie utworzony automatycznie.

Przykład:

```bash
python football-data.py 497623 matchStats/matchStatsForestCity.json
```

Skrypt:
1. Pobiera mecze ligowe (domyślnie Premier League, `COMPETITION_ID=2021`, sezon 2024) z okresu `DATE_FROM`–`DATE_TO`.
2. Pobiera tabelę dla kolejki sprzed wskazanego meczu (rating drużyn po pozycji w tabeli).
3. Dla obu drużyn liczy agregaty z ostatnich `NUM_MATCHES=10` meczów + `NUM_H2H_MATCHES=6` H2H.
4. Zapisuje wynik do wskazanego JSON-a.

> Konfiguracja zakresu dat, ligi i sezonu jest na razie na sztywno w `football-data.py` (zmienne `DATE_FROM`, `DATE_TO`, `COMPETITION_ID`, `SEASON`). Zmień je ręcznie, jeśli chcesz analizować inny okres/ligę.

### Krok 2. Trening modeli i predykcja

```bash
python predictors.py [<input_json>]
```

- `<input_json>` – **opcjonalna** ścieżka do JSON-a wygenerowanego w kroku 1. Bez tego skrypt użyje pliku wskazanego przez `FOOTBALL_STATS_FILE` lub `matchStats/matchStatsWesthamCrystal.json`.

Przykład:

```bash
python predictors.py matchStats/matchStatsForestCity.json
```

Skrypt:
1. Wczytuje JSON, buduje DataFrame z 18 cechami.
2. Usuwa cechy mocno skorelowane (>0.85) i skaluje (`StandardScaler`).
3. Trenuje 10 modeli regresji (z `GridSearch`/`RandomizedSearch` tam, gdzie ma to sens).
4. Wypisuje tabelę porównawczą (MSE, R², CV-score, predykcja) — kolejność po MSE rosnąco.
5. Wypisuje **uśrednione predykcje** (średnia prosta i trzy ważone wg jakości modeli).
6. Trenuje dodatkowo klasyfikator XGBoost dla over/under 2.5 gola.

### Alternatywnie: zmienna środowiskowa zamiast argumentów

```powershell
$env:FOOTBALL_STATS_FILE = "matchStats/matchStatsForestCity.json"
python football-data.py 497623
python predictors.py
```

---

## Skrypty pomocnicze

### Lista nadchodzących meczów danej ligi

```bash
python football-data_getFutureMatchesPerCompetition.py <COMPETITION_ID>
```

Daty `DATE_FROM`/`DATE_TO` są zaszyte w pliku – edytuj wg potrzeb.

### Lista drużyn z ich ID dla danej ligi

```bash
python football-data_getTeamsIdPerCompetition.py <COMPETITION_ID>
```

Typowe ID lig: `2021` = Premier League, `2014` = La Liga, `2002` = Bundesliga, `2019` = Serie A, `2015` = Ligue 1.

---

## Struktura projektu

```
.
├── football-data.py                              # Krok 1: pobranie statystyk dla meczu
├── football-data_getFutureMatchesPerCompetition.py
├── football-data_getTeamsIdPerCompetition.py
├── predictors.py                                 # Krok 2: trening + predykcja
├── utilities.py                                  # Klucz API, I/O, throttling requestów
├── PredictorsFunctions/
│   ├── dataInitializer.py                        # JSON -> DataFrame + train/test split
│   ├── models.py                                 # Definicje 10 modeli regresji
│   ├── modelsPredictions.py                      # Predykcja per-model
│   ├── modelsUtilities.py                        # Wspólne helpery + rejestr metryk
│   ├── classificators.py                         # XGBoost over/under 2.5
│   ├── predictionSummary.py                      # Tabela + ensemble (ważone średnie)
│   └── predictionPlot.py                         # (niewywoływany) plot porównawczy
├── matchStats/                                   # Zapisane statystyki dla różnych meczów (gitignored)
│   └── matchStats*.json
├── config.cfg                                    # Klucz API (lokalny, NIE w repo)
├── requirements.txt
├── KIERUNKI_ROZWOJU.md                           # Propozycje dalszego rozwoju
└── README.md
```

---

## Znane ograniczenia (aktualnego podejścia)

1. **Bardzo mały zbiór treningowy**: ~18 wierszy treningowych na 18 cech — duże ryzyko overfittingu w XGBoost/RF/NN.
2. **Wyciek danych w sieci neuronowej**: NN trenuje z `validation_data=(X_test, y_test)` i wybierany jest run z najlepszym R² na teście (`PredictorsFunctions/models.py`, funkcja `neuralNetwork_multiple_runs`).
3. **Anachronizm w `rivalsRatingAvg`**: rating rywali liczony jest na podstawie tabeli sprzed przewidywanego meczu, nie tabeli z czasu rozegrania historycznego meczu.
4. **Brak time-series split**: `train_test_split(random_state=42)` losuje, może wziąć starszy mecz do testu niż do treningu.
5. **Hardkodowany sezon/liga/zakres dat** w `football-data.py`.
6. **Brak persystencji modeli**: każdy run trenuje od zera (długo).
7. **Klasyfikator over/under nie używa Poissona**: regresor Poissona daje liczby, ale nie liczy P(total > 2.5) z prawdziwego rozkładu Poissona.

Pełna analiza i propozycje napraw — patrz [`KIERUNKI_ROZWOJU.md`](./KIERUNKI_ROZWOJU.md).

---

## Licencja

[MIT](./LICENSE)
