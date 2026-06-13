# Kierunki rozwoju projektu

Dokument opisuje możliwe ścieżki rozwoju aplikacji `FootballScraper`. Każdy kierunek zawiera: czego dotyczy, co nam daje, co jest wymagane, jaka jest trudność i jakie są pułapki. Na końcu znajduje się **rekomendowana kolejność prac** prowadząca do docelowego modelu **over/under + 1X2 + dokładny wynik**.

---

## Punkt wyjścia – stan obecny

Aktualnie pipeline:

1. Pobiera dane z football-data.org dla **jednego konkretnego meczu**.
2. Buduje DataFrame z ~18 wierszy (po ~9 ostatnich meczów każdej z drużyn).
3. Trenuje 10 modeli regresji **na tych ~18 wierszach** i przewiduje `total_goals` dla 1 wiersza (przewidywany mecz).
4. Dorzuca klasyfikator XGBoost dla over/under 2.5.

**Kluczowy problem:** model nie uczy się z „piłki nożnej w ogóle” — uczy się tylko z 18 historycznych obserwacji konkretnych dwóch drużyn. Przy 18 cechach i 18 wierszach każdy bardziej złożony model (RF, XGBoost, NN) przeucza. Dlatego prawie wszystkie kierunki niżej mają **wspólny mianownik: zbudowanie dużego, globalnego zbioru treningowego**.

---

## Kierunek 0 — Porządki w obecnym kodzie (bez zmiany architektury)

**Cel.** Zachować obecne podejście, ale poprawić wycieki danych i porządek pracy.

**Co zrobić.**

- Usunąć wyciek w sieci neuronowej: zamiast `validation_data=(X_test, y_test)` użyć drugiego split-a z `X_train` (np. `train_test_split(X_train, y_train, test_size=0.2)`), a `X_test` zostawić tylko do końcowej oceny.
- Zastąpić `train_test_split(random_state=42)` **time-aware split-em** (np. `TimeSeriesSplit` po kolejce / dacie meczu).
- Wyciągnąć hardkody (`DATE_FROM`, `DATE_TO`, `COMPETITION_ID`, `SEASON`, `NUM_MATCHES`) do argumentów CLI / `config.cfg`.
- Dodać cache odpowiedzi z API (np. plik JSON per endpoint) — oszczędność requestów (limit 10/min).
- Dodać `--quiet` / `--only <model>` w `predictors.py`, żeby móc szybko iterować po jednym modelu.
- Dorzucić prosty test sanity: czy `predict_df` ma dokładnie 1 wiersz, czy `total_goals` ma sensowny zakres itp.

**Co nam to daje.** Bardziej wiarygodne metryki (bez sztucznie zawyżonego R² w NN), powtarzalność, łatwiejsza praca. **Nie naprawia jednak ograniczenia 18 wierszy.**

**Wymagania danych.** Brak nowych.

**Trudność.** Niska (1–2 dni).

**Pułapki.** Nawet po naprawie metryki nadal będą szumne — na 4–5 wierszowym teście pojedyncza obserwacja drastycznie zmienia R².

---

## Kierunek 1 — Globalny zbiór z wielu sezonów i lig (FUNDAMENT pod wszystko dalej)

**Cel.** Przejść z modelu „per-mecz, 18 wierszy” na model trenowany **raz** na tysiącach historycznych meczów, a potem szybko predykujący dowolny nadchodzący mecz.

**Co zrobić.**

- Zbudować zbiór historycznych meczów (np. 5–10 sezonów × 5 lig top = 8000–15000 meczów). Źródła:
  - **football-data.co.uk** (CSV-y od ~1993, darmowe, w komplecie kursy bukmacherskie pre-match) — najlepsze źródło na start.
  - football-data.org (które już używasz) — pre-match data, dobrze do live, ale gorzej do masowego pobrania historii (limity).
  - **understat.com / fbref.com** — dodatkowo dają **xG/xGA**, co znacząco poprawia modele goli.
- Dla każdego historycznego meczu policzyć cechy **tak, jak były znane przed jego rozegraniem** (ważne: forma drużyn liczona z meczów ściśle wcześniejszych — żadnych spojlerów).
- Trenować na N-1 sezonach, walidować na sezonie wyłączonym (out-of-time validation). Inaczej overfit się ukryje.
- Cache + persystencja: zapisać wytrenowane modele do `joblib`/`pickle`, żeby predykcja konkretnego meczu trwała sekundy.

**Co nam to daje.** Dopiero przy ~kilku tysiącach meczów modele typu XGBoost/NN zaczynają sensownie generalizować. R² na out-of-time validation będzie realistyczny (typowe dla goli: 0.05–0.15 — to dużo, w piłce jest mnóstwo szumu). Modele będą porównywalne z benchmarkami z literatury.

**Wymagania danych.**

- Pobrać i sparsować ~10 sezonów CSV z football-data.co.uk (lub odpowiednik).
- Zaprojektować schemat: każdy wiersz = 1 mecz, cechy obliczone wyłącznie z meczów wcześniejszych.
- Pamiętać o transferze drużyn między ligami (awanse/spadki) — drużyna grająca w Championship ma inną siłę niż w Premier League.

**Wymagania techniczne.**

- `pandas`, `requests`, ewentualnie `polars` dla szybkości.
- Skrypt budujący zbiór (czasochłonny, ale uruchamiany rzadko).
- Mała baza/parquet (np. `matches.parquet`) zamiast wielu JSON-ów.

**Trudność.** Średnia — najwięcej pracy w **feature engineeringu** i ścisłym pilnowaniu „brak danych z przyszłości”.

**Pułapki.**

- **Wyciek czasowy** to klasyczny błąd #1: jeśli pomylisz indeksy meczów, model „widzi przyszłość” i metryki idą do nieba. Sprawdzaj rygorystycznie sortowanie po dacie.
- **Promocje/relegacje** — drużyna może mieć tylko 1 sezon w Premier League. Trzeba albo kodować `team_id` jako embedding, albo (lepiej) usunąć tożsamość i opisać drużynę cechami (forma, xG, rating).
- **Survivorship**: 0-0 i niezakończone mecze trzeba traktować osobno.

**Szacowany nakład.** 1–2 tygodnie solo, jeśli liczyć też feature engineering i porządne walidacje.

---

## Kierunek 2 — Over/Under z prawdziwymi prawdopodobieństwami (Poisson bivariate / dixon-coles)

**Cel.** Wyjść poza „przewidzieć liczbę goli” i zwracać **P(total > 2.5)**, **P(BTTS = tak)**, itd.

**Co zrobić.**

- Zamiast jednego modelu na `total_goals` trenować **dwa modele Poissona**: jeden na `home_goals`, drugi na `away_goals`. Cechy: forma, xG, rating obu drużyn, przewaga gospodarza.
- Z dwóch oszacowanych intensywności λ_home, λ_away wyliczyć macierz 8×8 wyników P(home_goals=i, away_goals=j) = Poisson(i; λ_h) · Poisson(j; λ_a).
- Z macierzy wyliczyć:
  - P(home + away > 2.5) → over 2.5
  - P(home > 0 AND away > 0) → BTTS
  - P(home > away), P(home == away), P(home < away) → 1X2 (!)
- **Korekta Dixon-Coles** (mała poprawka na wyniki 0:0, 1:0, 0:1, 1:1, gdzie czysty Poisson niedoszacowuje korelacji) — to klasyczny artykuł akademicki z 1997 r., implementacja w ~50 liniach kodu.
- Alternatywnie: **Bivariate Poisson** (Karlis & Ntzoufras), który modeluje korelację bramek obu drużyn jawnie.

**Co nam to daje.**

- Prawdziwe prawdopodobieństwa, nie tylko punktowe predykcje.
- **Jednym modelem dostajesz over/under, BTTS, 1X2 i ranking dokładnych wyników** — to bardzo eleganckie podejście, standard w branży.
- Da się porównać z kursami bukmacherskimi (1/odds = implied probability) i liczyć value bets.

**Wymagania danych.** Wymaga **Kierunku 1** (globalny zbiór z wielu sezonów). Dodatkowo:

- **xG / xGA** drastycznie poprawia Poissona (np. z fbref/understat).
- Cechy kontekstowe: dni odpoczynku, podróż (h2h dystans, krajowy vs europejski mecz), wcześniejsze rotacje składów.

**Wymagania techniczne.**

- `statsmodels` (`GLM` z `family=Poisson`) lub `scikit-learn` (`PoissonRegressor`, już używany).
- Dla Dixon-Coles: implementacja własna lub gotowe pakiety: [`penaltyblog`](https://github.com/martineastwood/penaltyblog) (Python, ma D-C, bivariate Poisson, Elo, value betting).
- `scipy.stats.poisson` do macierzy wyników.

**Trudność.** Średnia. Sama matematyka jest prosta, najwięcej kodu to robi feature engineering.

**Pułapki.**

- Czysty Poisson niedoszacowuje wyników typu 1:1 — dlatego Dixon-Coles.
- Zakłada niezależność bramek (BVP/D-C to częściowo łagodzą).
- Przy wysokich λ (mecze z 5+ bramkami średnio) macierz 8×8 nie wystarczy — przejdź na 12×12.

**Szacowany nakład.** 3–5 dni po zbudowaniu Kierunku 1.

---

## Kierunek 3 — Klasyfikacja 1X2 jako osobny model multinomial

**Cel.** Bezpośredni model na wynik 1/X/2 (zwycięstwo gospodarza / remis / zwycięstwo gości).

**Co zrobić.**

- Target: `match_result ∈ {H, D, A}`.
- Model: regresja logistyczna multinomialna, XGBoost/LightGBM z `objective='multi:softprob'`, sieć neuronowa z softmax-3.
- Cechy: te same co przy Kierunku 1+2.
- Output: 3 prawdopodobieństwa sumujące się do 1, ocena: `log-loss`, `Brier score`, kalibracja (`reliability diagram`).

**Co nam to daje.** Dedykowany model 1X2, często **trochę lepszy** niż wyciąganie 1X2 z modelu Poissona bramek, bo uczy się bezpośrednio na sygnale „kto wygrał”.

**Wymagania danych.** Wymaga Kierunku 1.

**Wymagania techniczne.** `scikit-learn` / `xgboost` / `lightgbm` (już w `requirements.txt`).

**Trudność.** Niska–średnia (1–3 dni po Kierunku 1).

**Pułapki.**

- **Class imbalance**: remisów jest ~25%, ale ich predykcja jest najtrudniejsza (modele rzadko stawiają X jako top-1).
- **Kalibracja** jest ważniejsza niż accuracy. Bez kalibracji prawdopodobieństwa są nieużywalne do value bettingu. Użyj `CalibratedClassifierCV` (sigmoid / isotonic).
- Sprawdź czy model bije baseline „home odds” (rynek bukmacherski po removal'u marży to mocny benchmark — bez przewagi informacyjnej ciężko go pobić).

**Szacowany nakład.** 2–3 dni po Kierunku 1.

---

## Kierunek 4 — Predykcja dokładnego wyniku (scoreline)

**Cel.** Zwracać prawdopodobieństwo każdego konkretnego wyniku: 1:0, 2:0, 2:1, 1:1, …

**Co zrobić.**

- **Opcja A (najprostsza, najbardziej spójna): wyciągnąć z Kierunku 2** — macierz Poissona/Dixon-Coles bezpośrednio zawiera P(home_goals=i, away_goals=j). Most prawdopodobny scoreline = `argmax(matrix)`.
- **Opcja B (osobny model klasyfikacyjny)**: target = wynik jako klasa (`"2:1"`, `"1:1"`, ...). Klas jest dużo (top ~20–30 wyników pokrywa ~95%+), więc trzeba ograniczyć słownik (`other` jako dodatkowa klasa).
- **Opcja C (ordinal regression / dwie głowy NN)**: jedna głowa predykuje `home_goals`, druga `away_goals`, każda z `objective='count:poisson'` (XGBoost obsługuje).

**Co nam to daje.** Predykcja dokładnego wyniku to **najtrudniejszy target w piłce** — modelowo ciekawy, ale top-1 trafność rzadko przekracza 10–14%. Bardziej użyteczne jest top-3 albo top-5 wyników (czy nasz „prawdziwy” mieści się w naszej top-5?).

**Wymagania danych.** Kierunek 1 + najlepiej xG (Kierunek 2).

**Trudność.** Średnia–wysoka, jeśli oczekujesz dobrych wyników. Sama implementacja Opcji A to dosłownie kilka linii nad Kierunkiem 2.

**Pułapki.**

- Top-1 accuracy będzie niska (8–14%), nie zniechęcaj się.
- Modele scoreline trzeba oceniać przez **log-loss / RPS (Ranked Probability Score)**, nie przez accuracy.
- Wyniki o niskim prawdopodobieństwie (np. 5:4) zawsze będą losowe — modeluj je jako „long tail”.

**Szacowany nakład.** Opcja A: 1 dzień nad Kierunkiem 2. Opcja B/C: 3–5 dni.

---

## Kierunek 5 — Dodatkowe źródła danych (transversal)

Każdy z kierunków 2–4 zyskuje, jeśli wzbogacimy cechy. Najwartościowsze:

| Źródło | Co daje | Trudność pozyskania |
|---|---|---|
| **xG / xGA per match** (understat.com, fbref.com) | Lepszy sygnał niż same gole — wygładza pecha/szczęścia | Średnia (scraping, brak oficjalnego API) |
| **Kursy pre-match** (football-data.co.uk, Betfair Historical Data) | Implied probability rynku — silny benchmark + feature | Niska (CSV) |
| **Składy / kontuzje** (api-football.com, payed) | Top-scorer nie gra → λ niżej; ważne | Średnia–wysoka (płatne API) |
| **Elo / Glicko rating drużyn** (np. clubelo.com) | Dynamiczny rating drużyn, lepszy niż pozycja w tabeli | Niska–średnia (CSV / scraping) |
| **Strefa czasowa / dni odpoczynku** | Drużyna po Lidze Mistrzów w środę gra gorzej w sobotę | Niska (z danych meczów) |
| **Pogoda** | Marginalna poprawa, ale czasem widoczna (śnieg, wichura) | Średnia |

**Rekomendacja:** zaczynaj od **xG + kursy + Elo**. Te trzy zmienne dają największy zwrot na zainwestowany czas.

---

## Kierunek 6 — Inżynieria modeli i ewaluacja

Po zbudowaniu Kierunków 1–2 warto:

- **Backtest** na całym wyłączonym sezonie: predykcja każdego meczu używając tylko danych sprzed tego meczu. Metryki: log-loss, Brier, RPS, accuracy, kalibracja, ROI vs kursy bukmacherskie.
- **Walk-forward validation**: rolować okno treningowe co N kolejek, retrenować, predykować — symuluje realne użycie.
- **Kalibracja prawdopodobieństw**: `CalibratedClassifierCV` (sigmoid/isotonic).
- **Stacking / blending** wielu modeli (Poisson + XGB-1X2 + LightGBM-scoreline) — średnio daje ~2–5% poprawy log-loss.
- **Bayesowskie modele dynamiczne** (np. `PyMC` + Dixon-Coles z time-varying attack/defence ratings) — najwyższa półka, dużo pracy, najlepsze wyniki w literaturze (poziom: papier akademicki).

---

## Rekomendowana kolejność prac

Skoro celem są **over/under + 1X2 + dokładny wynik**, sugeruję następującą ścieżkę:

```
[0] Porządki (1-2 dni)
    │
    ▼
[1] Globalny zbiór z wielu sezonów (1-2 tyg)   ← bez tego nic więcej nie ruszy z miejsca
    │
    ▼
[2] Poisson bivariate / Dixon-Coles (3-5 dni)  ← daje OVER/UNDER + BTTS + 1X2 + scoreline naraz
    │
    ├─► (opcjonalnie) [3] osobny model 1X2 jako klasyfikacja — porównaj z 1X2 z Poissona
    │
    ├─► (opcjonalnie) [4] osobny model scoreline — porównaj z Poissonem
    │
    └─► [5] xG + Elo + kursy jako cechy — największe pojedyncze ulepszenie
            │
            ▼
       [6] Backtest, kalibracja, stacking
```

**Logika:** Kierunek 2 jest tu kluczowy, bo **jeden poprawnie wytrenowany model Poissona** z bramek home/away zwraca naraz:

- P(over/under X) — dla dowolnego X,
- P(BTTS),
- P(1X2),
- macierz scoreline → top-N wyników i dokładny argmax.

Czyli przy ścieżce [0] → [1] → [2] dostajesz **wszystkie trzy targety, których chcesz** (over/under, 1X2, dokładny wynik) z jednego modelu, spójnie. Potem warto sprawdzić w [3]/[4], czy dedykowane modele biją Poissona na poszczególnych targetach.

---

## Czego NIE robić

- Nie wrzucaj XGBoost / RandomForest / NN na 18-wierszowy zbiór i nie traktuj uzyskanych R² serio — to nie jest miarodajne.
- Nie używaj `train_test_split` losowo w danych czasowych — zawsze split po czasie.
- Nie zostawiaj `validation_data=(X_test, y_test)` w NN.
- Nie próbuj naraz pisać 5 osobnych modeli (over/under, 1X2, scoreline) bez wcześniejszego zbioru globalnego — utkniesz na metrykach.
- Nie używaj accuracy do ewaluacji modeli prawdopodobieństwowych — używaj log-loss / Brier / RPS.
- Nie zapominaj o **kalibracji** — surowe probabilities z lasów/xgboosta są często źle skalibrowane.

---

## Przydatne biblioteki / lektura

- **Pakiety:**
  - [`penaltyblog`](https://github.com/martineastwood/penaltyblog) — Dixon-Coles, Bivariate Poisson, Elo, value betting (Python)
  - [`statsmodels`](https://www.statsmodels.org/) — GLM Poisson z ładnym summary
  - [`scikit-learn`](https://scikit-learn.org/) — `PoissonRegressor`, `CalibratedClassifierCV`
  - [`shap`](https://github.com/shap/shap) — wyjaśnialność modeli drzewiastych
- **Lektura:**
  - Dixon & Coles (1997) "Modelling Association Football Scores and Inefficiencies in the Football Betting Market" — klasyk, ~10 stron
  - Karlis & Ntzoufras (2003) "Analysis of sports data by using bivariate Poisson models"
  - Constantinou et al. "pi-football" — model bayesowski na żywych meczach
- **Źródła danych:**
  - [football-data.co.uk](https://www.football-data.co.uk/) — historyczne CSV z kursami
  - [understat.com](https://understat.com/) — xG od sezonu 2014/15
  - [fbref.com](https://fbref.com/) — szeroki zakres statystyk
  - [clubelo.com](http://clubelo.com/) — ratingi Elo klubów
