import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="AI Movie Recommender", layout="wide")
def set_movie(name):
    st.session_state.sel = name
def reset():
    st.session_state.sel = None
    st.session_state.res = []
if 'sel' not in st.session_state:
    st.session_state.sel = None
if 'res' not in st.session_state:
    st.session_state.res = []
@st.cache_data
def load():
    url='https://www.dropbox.com/scl/fi/2kewaxup7w4jq5rissi7a/master_movie_data.csv?rlkey=1zytk3770hckquwn1z8qbmt5l&st=14hi9ulm&dl=1'
    df = pd.read_csv(url, nrows=10000)
    cols = ['genre', 'name_y', 'description', 'tagline', 'director', 'theme_keywords']
    for c in cols:
        df[c] = df[c].fillna('')
    df['text'] = df['genre'] + ' ' + df['genre'] + ' ' + df['name_y'] + ' ' + df['description'] + ' ' + df['description'] + ' ' + df['description'] + ' ' + df['tagline'] + ' ' + df['director'] + ' ' + df['theme_keywords'] + ' ' + df['theme_keywords']
    return df
@st.cache_resource
def get_vecs(df):
    v = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.30)
    return v.fit_transform(df['text'])
df = load()
vecs = get_vecs(df)
st.title("AI Movie Recommender")
st.caption("Sample Portfolio Project")
st.info("⭐⭐**Note:** This live demo is processing a limited dataset of 10,000 movies to optimize performance within cloud memory constraints.⭐⭐")
if st.session_state.sel is None:
    st.write("### Step 1: Find your movie")
    q = st.text_input("Type a movie name:")
    if st.button("Search"):
        if q:
            t = str(q).lower().replace(" ", "")
            titles = df['name_x'].astype(str).str.lower().str.replace(" ", "")
            idx = df[titles.str.contains(t, na=False)].index
            if len(idx) > 0:
                st.session_state.res = df.loc[idx].sort_values(by='rating', ascending=False).head(5)['name_x'].tolist()
            else:
                st.error("No movies found.")
    if st.session_state.res:
        st.write("### Step 2: Which one did you mean?")
        cols = st.columns(len(st.session_state.res))
        for i, name in enumerate(st.session_state.res):
            with cols[i]:
                p = df[df['name_x'] == name]['link'].values[0]
                if pd.isna(p) or p == '':
                    p = "https://via.placeholder.com/500x750?text=No+Poster"
                st.image(p, use_container_width=True)
                st.markdown(f"**{name}**")
                st.button("Select", key=f"s_{i}", on_click=set_movie, args=(name,))
if st.session_state.sel:
    st.success(f" Recommendations for: **{st.session_state.sel}**")
    idx = df[df['name_x'] == st.session_state.sel].index[0]
    score = cosine_similarity(vecs[idx], vecs)[0]
    dfs = df.copy()
    dfs['score'] = score
    k = 'TV|Series|Show|Season|Episode|Sitcom|Miniseries'
    m = ~(dfs['genre'].str.contains(k, na=False, case=False) | dfs['theme_keywords'].str.contains(k, na=False, case=False) | dfs['description'].str.contains(k, na=False, case=False))
    dfs = dfs[m & (dfs['rating'] >= 3.0) & (dfs.index != idx)]
    top = dfs.sort_values(by='score', ascending=False).head(5)
    st.markdown("### Top 5 Recommendations:")
    cols = st.columns(5)
    for i, (r_idx, r) in enumerate(top.iterrows()):
        with cols[i]:
            p = r['link']
            if pd.isna(p) or p == '':
                p = "https://via.placeholder.com/500x750?text=No+Poster"
            st.image(p, use_container_width=True)
            st.markdown(f"**{r['name_x']}**")
            st.caption(f" {round(r['rating'], 1)}/5 | Match: {round(r['score'] * 100, 1)}%")
    st.write("---")
    st.button(" Search again", on_click=reset)