# 🎬 AI Movie Recommender

**[🟢 Try the Live App Here!](https://movie-recommender-nipun.streamlit.app/)**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)

A lightweight, content-based movie recommendation engine built with Python and Streamlit. This application uses Natural Language Processing (TF-IDF) and Cosine Similarity to find and recommend movies based on a combination of genres, descriptions, directors, and theme keywords.

> **💡 Live Demo Note:** The live Streamlit deployment currently processes a limited dataset of **10,000 movies** to optimize performance and prevent out-of-memory errors on free cloud tiers. The full algorithm and local code support the entire 400MB dataset.

---

## ✨ Features

- **Content-Based Filtering:** Recommends movies by analyzing textual metadata (genre, description, tagline, director, and keywords) rather than relying purely on user ratings. 
- **Smart Search & Disambiguation:** Quickly search for movies in the database with typo-tolerant matching. If multiple movies match, it presents the top 5 (by rating) for the user to choose from.
- **Dynamic Cloud Data Loading:** Uses Pandas to stream the dataset directly from a cloud-hosted URL, bypassing GitHub's strict file size limits.
- **TV Show Filtering:** Automatically filters out TV series, sitcoms, and episodes using Regex to ensure only feature films are recommended.
- **Optimized Performance:** Uses Streamlit caching (`@st.cache_data` and `@st.cache_resource`) to compute the complex TF-IDF mathematical matrix only once, keeping the user experience lightning fast.

---

## 🛠️ Tools Used

* **[Streamlit](https://streamlit.io/):** The web framework used for building the responsive UI and handling state management.
* **[Pandas](https://pandas.pydata.org/):** Used for data ingestion, manipulation, missing value handling, and cloud-streaming.
* **[Scikit-Learn](https://scikit-learn.org/):** Powers the core AI using `TfidfVectorizer` for Natural Language Processing and `cosine_similarity` for match calculation.
* **[Kaggle](https://www.kaggle.com/):** Primary source for the comprehensive movie datasets, metadata, and future feature engineering (like extended crew data).[https://www.kaggle.com/datasets/gsimonx37/letterboxd?select=crew.csv]

---

## 🧠 How the Algorithm Works

1. **Feature Engineering:** The app concatenates text fields into a single string, artificially increasing the weight of highly important features like genre, description, and keywords by duplicating them.
2. **Vectorization:** A `TfidfVectorizer` converts these strings into a mathematical matrix, capping vocabulary at 5,000 features and removing common English stop words.
3. **Similarity Calculation:** When a user selects a movie, the app retrieves its vector and calculates the **Cosine Similarity** between that specific movie and all others.
4. **Filtering & Ranking:** The app filters out the original movie, removes TV shows, ensures a minimum rating of `3.0`, and returns the top 5 matches.
5. 
---

## 🚀 How to Setup Locally

Want to run the app locally with the full 400MB dataset? Follow these steps:

### 1. Clone the repository
Open your terminal and run:
```bash
git clone [https://github.com/reddypnipun/AI-Movie-Recommender.git](https://github.com/reddypnipun/AI-Movie-Recommender.git)
cd AI-Movie-Recommender
```
### 2. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```
### 3. Provide the Dataset
The live app streams data from a public Dropbox link. If you want to use your massive local file:
  * Place your master_movie_data.csv in the root directory.
  * Open app.py and update the load() function to read the local file instead of the URL:
  * df = pd.read_csv('master_movie_data.csv')
  * (Note: Because movie datasets can be very large, master_movie_data.csv and *.zip files are ignored by .gitignore in this repository to prevent push errors).
### 4. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py

