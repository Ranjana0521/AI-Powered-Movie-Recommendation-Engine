import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class MovieRecommender:

    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)

        self.preprocess()
        self.build_collaborative()
        self.build_content_based()

    # ----------------------------
    # Data Preprocessing
    # ----------------------------
    def preprocess(self):
        self.data = pd.merge(self.ratings, self.movies, on="movieId")

    # ----------------------------
    # Collaborative Filtering
    # ----------------------------
    def build_collaborative(self):
        self.user_movie_matrix = self.data.pivot_table(
            index="userId",
            columns="title",
            values="rating"
        ).fillna(0)

        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

    # ----------------------------
    # Content-Based Filtering
    # ----------------------------
    def build_content_based(self):
        self.movies["genres"] = self.movies["genres"].str.replace("|", " ")
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies["genres"])
        self.content_similarity = cosine_similarity(tfidf_matrix)

    # ----------------------------
    # Recommend by Movie Name
    # ----------------------------
    def recommend_by_movie(self, movie_name, top_n=5):
        if movie_name not in self.movies["title"].values:
            return ["Movie not found in dataset"]

        idx = self.movies[self.movies["title"] == movie_name].index[0]

        similarity_scores = list(enumerate(self.content_similarity[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        similar_movies = similarity_scores[1:top_n+1]
        recommendations = [self.movies.iloc[i[0]]["title"] for i in similar_movies]

        return recommendations

    # ----------------------------
    # Recommend by User ID
    # ----------------------------
    def recommend_by_user(self, user_id, top_n=5):
        if user_id not in self.user_movie_matrix.index:
            return ["User ID not found"]

        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False)
        most_similar_user = similar_users.index[1]

        user_movies = self.user_movie_matrix.loc[user_id]
        similar_user_movies = self.user_movie_matrix.loc[most_similar_user]

        recommendations = similar_user_movies[
            (similar_user_movies > 0) & (user_movies == 0)
        ].sort_values(ascending=False)

        return recommendations.head(top_n).index.tolist()

    # ----------------------------
    # Hybrid Recommendation
    # ----------------------------
    def hybrid_recommend(self, user_id, movie_name, top_n=5):
        content_recs = set(self.recommend_by_movie(movie_name, top_n=10))
        collaborative_recs = set(self.recommend_by_user(user_id, top_n=10))

        # Combine both recommendations (simple union)
        hybrid_recs = list(content_recs.union(collaborative_recs))

        # Remove the movie already liked
        if movie_name in hybrid_recs:
            hybrid_recs.remove(movie_name)

        # Return top N
        return hybrid_recs[:top_n]