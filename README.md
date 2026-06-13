<div align="center">
  <h1 align="center">Football Scraper</h1>  
  <br/>
</div>

## Table of Contents
* [About The Project](#about-the-project)
* [Built with](#built-with)
* [Features](#features)
* [Usage](#usage)
* [Room for Improvement](#room-for-improvement)
* [Contact with author](#contact-with-author)

## About The Project

Fun project to play around with football data. It's an app consisting of scripts that allow you to download data about football matches from various competitions
via an API, and then use this data to try to predict the number of goals in the next matches using various statistical models. 

API used in this project is [api.football-data](https://api.football-data.org).

During the project, there was a lot of testing of ChatGPT, DeepSeek and DeepAI for coding.

## Built With

- Python
- Data from [api.football-data](https://api.football-data.org)
- Lot of python packages, like:
    - packages from sklearn
    - packages from keras
    - pandas & numpy & matplotlib
    - xgboost, lightgbm

## Features

* Getting teams ID per competition
  - [GetTeamsPerCompetition](https://github.com/MikeTrusky/FootballScraper/blob/main/football-data_getTeamsIdPerCompetition.py) python script
  - using API to get IDs and print in console
* Getting future matches per competition
  - [GetFutureMatches](https://github.com/MikeTrusky/FootballScraper/blob/main/football-data_getFutureMatchesPerCompetition.py) python script
  - using API to get future matches and print in console
* Preparing data for match's goals prediction
  - [Main data script](https://github.com/MikeTrusky/FootballScraper/blob/main/football-data.py)
  - using API to get previous matches in competition
  - preparing a file with detailed data between two teams based on their previous matches
* Predict goals in future matches
  - [Main predictor script](https://github.com/MikeTrusky/FootballScraper/blob/main/predictors.py) use to call chosen models
  - possible models:
      - Naive
      - Linear regression
      - Polynomial regression
      - Decision Tree
      - Random Forest
      - Gradient Boosting
      - XGB Regressor
      - Neural Network
      - Poisson regressor
      - Light GBM
      - XGB Classificator

## Usage

1. Python packages
   * Use for example `pip install package_name` to install necessary packages
2. API key
   * To use API you need an account on [api.football-data](https://api.football-data.org) and create a "config.cfg" file,
     where you put your API key in a format:
    ```
    [KEYS]
    API_KEY="your_api_key"
    ```
3. Prepare a file with detailed data
   * Use `python football-data.py MATCH_ID` with specific MATCH_ID
   * Script save a data to matchStats.json file (you can change this file name in utilities.py file)
4. Try to predict goals
   * Run `python predictors.py`
   * Predictions will be print in a console

## Room for Improvement

- [ ] Added more models: for example CatBoost, LSTM, Transformery
- [ ] Add more data to matchStats.json data file

## Contact with author

Maciej Trzuskowski

- ![Gmail][gmail-icon] - maciej.trzuskowski@gmail.com
- ![LinkedIn][linkedin-icon] - https://www.linkedin.com/in/maciej-trzuskowski/

[gmail-icon]: https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white
[linkedin-icon]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
