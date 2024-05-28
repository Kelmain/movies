# Movies

## Overview

This folder contains a movie recommandation app to try different ML models. It's for school projet from the Wild Code School. 


# Case Study

You are a freelance data analyst. A cinema in decline located in the Creuse contacts you. They've decided to go digital by creating a website tailored to local audiences. To take things a step further, they've asked you to create a film recommendation recommendation engine that will eventually send notifications to customers via the the Internet. For the moment, no customer has entered their preferences, so you're in a cold start. But fortunately, the customer gives you a database of database of films based on the IMDb platform. You start by proposing a complete analysis of the database (Which actors are most present? At what period? Is the average length of get longer or shorter over the years? Are series actors the same as the same as in films? What is the average age of actors? What are the best-rated best rated? Do they share common characteristics? etc.) Following this initial analysis, you can decide to specialise your cinema, for example on the "period "90s", or "action and adventure films", to refine your exploration. After this analytical stage, at the end of the project, you will use machine learning algorithms to recommend films based on films that have been enjoyed by the viewer. The client will also provide you with a complementary database from TMDB, containing data on the countries of the production companies, budget, box office and a path to the film posters. You will be asked to retrieve the images of the films to display them in your recommendation interface. WARNING, the aim is not to show the recommended films in the cinema. The final objective is to have an application with KPIs on the one hand and the recommendation system with a film name input box for the user. This application will be made available to cinema customers in order to offer them an additional online service, in addition to the traditional cinema

## Content Description

- **sql/**: This subfolder contains the schema for the duckdb database
- **src/**: Contains Python scripts for downloading the data, cleaning it and loading it into the database.
- **Home.py**: Contains the streamlit app to display the recommendations
- **utilities.py**: Contains the functions to train the ML models and make the recommendations

## Usage

### SQL

The SQL files are the schema for the duckdb database. You can run them to create the database and the tables.   

### SRC

The api_conn.py file contains the functions to connect to the API and download the data. 
The make_dataset.py file contains the functions to clean the data and load it into the database.


Replace `<script_name>` with the name of the script you wish to run. These scripts are used to perform data cleaning and analysis tasks on the movie datasets.



## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. You can also open an issue if you find bugs or have suggestions for improvements.



## Contact

For any additional questions or comments, please email [neumann.arminpro@gmail.com](mailto:neumann.arminpro@gmail.com).
