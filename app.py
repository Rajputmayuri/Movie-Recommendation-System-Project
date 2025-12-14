import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommender System", page_icon="🍿", layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
<style>
body {
    background-color: #020617;
}
h1, h2, h3 {
    color: #f8fafc;
}
.genre {
    background-color: #1e293b;
    padding: 6px 10px;
    border-radius: 14px;
    margin-right: 6px;
    margin-bottom: 6px;
    display: inline-block;
    color: #e0e7ff;
    font-size: 13px;
}
.card {
    background-color: #020617;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 10px;
    box-shadow: 0 0 12px rgba(99,102,241,0.35);
}
.footer {
    text-align:center;
    color: gray;
    margin-top: 40px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎬 Smart Movie Recommender System 🍿</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#cbd5f5;'>Discover movies similar to your favorites</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
vectorizer = joblib.load("vectorizer.pkl")
movies = joblib.load("movies.pkl")

ratings = pd.read_csv(
    "u.data", sep="\t", names=["userId", "movieId", "rating", "timestamp"]
)


# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def clean_text(title):
    return re.sub("[^a-zA-Z0-9]", " ", title)


tfidf = vectorizer.fit_transform(movies["clean_text"])


def search(title):
    title = clean_text(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -8)[-8:]
    return movies.iloc[indices][::-1]


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)][
        "userId"
    ].unique()

    similar_user_recs = ratings[
        (ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)
    ]["movieId"]

    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]

    all_users = ratings[
        (ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)
    ]

    all_users_recs = all_users["movieId"].value_counts() / len(
        all_users["userId"].unique()
    )

    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]

    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[
        ["score", "title", "genres"]
    ]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🎯 Search Controls")
movie_name = st.sidebar.text_input("🎥 Enter movie name", placeholder="Toy Story")

min_score = st.sidebar.slider("🔥 Minimum recommendation strength", 0.0, 2.0, 0.4)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Try popular movies for better recommendations")

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if movie_name:
    st.subheader("🔍 Search Results")

    search_results = search(movie_name)

    if not search_results.empty:
        st.dataframe(search_results[["title", "genres"]], use_container_width=True)

        selected_index = st.selectbox(
            "🎬 Select a movie",
            search_results.index,
            format_func=lambda x: search_results.loc[x, "title"],
        )

        if selected_index is not None:
            selected_movie_id = search_results.loc[selected_index, "movieId"]
            selected_title = search_results.loc[selected_index, "title"]

            st.subheader(f"🍿 Recommended Movies for **{selected_title}**")

            with st.spinner("Finding best recommendations..."):
                st.progress(60)
                recommendations = find_similar_movies(selected_movie_id)

            recommendations = recommendations[recommendations["score"] >= min_score]

            if not recommendations.empty:
                # Download CSV
                csv = recommendations.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Recommendations (CSV)",
                    csv,
                    "recommended_movies.csv",
                    "text/csv",
                )

                # Display cards
                for _, row in recommendations.iterrows():
                    with st.expander(f"🎞️ {row['title']}"):
                        st.markdown("<div class='card'>", unsafe_allow_html=True)

                        # Genres
                        for g in row["genres"].split("|"):
                            st.markdown(
                                f"<span class='genre'>{g}</span>",
                                unsafe_allow_html=True,
                            )

                        score = row["score"]
                        st.markdown(f"**Recommendation Strength:** 🔥 {score:.2f}")
                        st.progress(min(score / 2, 1.0))

                        if score > 1.2:
                            st.success("⭐ Highly Recommended")
                        elif score > 0.6:
                            st.info("👍 Good Match")
                        else:
                            st.warning("🙂 Decent Choice")

                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No recommendations found with selected filters.")
    else:
        st.error("❌ No movies found. Try another title.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
<hr>
<div class='footer'>
🚀 Built with Streamlit | MovieLens 100K Dataset <br>
🐙 <a href="https://github.com/your-username/your-repo" target="_blank">
View Source Code on GitHub
</a>
</div>
""",
    unsafe_allow_html=True,
)
