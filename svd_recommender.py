import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from audiocompression import AudioCompression


class SVDRecommender(object):

    def __init__(self) -> None:
        """
        Initialize with AudioCompression object for SVD purposes
        """
        self.audiocompression = AudioCompression()

    def load_movie_data(self, filepath: str = "./data/movies.csv") -> None:
        """
        PROVIDED TO STUDENTS:
        Load movie data and create mappings from movie name to movie ID and vice versa
        """
        movies_df = pd.read_csv(filepath)
        self.movie_names_dict = dict(zip(movies_df.movieId, movies_df.title))
        self.movie_id_dict = dict(zip(movies_df.title, movies_df.movieId))

    def load_ratings_datasets(
        self,
        train_filepath: str = "./data/ratings_train.csv",
        test_filepath: str = "./data/ratings_test.csv",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        PROVIDED TO STUDENTS: Load train and test user-movie ratings datasets
        """
        train = pd.read_csv(train_filepath)
        test = pd.read_csv(test_filepath)
        return train, test

    def get_movie_name_by_id(self, movie_id: int) -> str:
        """
        PROVIDED TO STUDENTS: Get movie name for corresponding movie id
        """
        return self.movie_names_dict[movie_id]

    def get_movie_id_by_name(self, movie_name: str) -> int:
        """
        PROVIDED TO STUDENTS: Get movie id for corresponding movie name
        """
        return self.movie_id_dict[movie_name]

    def recommender_svd(self, R: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the matrix of Ratings (R) and number of features (k), build the singular
        value decomposition of R with numpy's SVD.
        
        (Reverting to the standard "economy" SVD implementation, as the
         audiocompression hint seems to cause the floating point errors)

        Args:
            R: (NxM) numpy array the train dataset upon which we'll try to predict / fill in missing predictions
            k: (int) number of important features we would like to use for our prediction
        Return:
            U_k: (Nxk) numpy array containing k features for each user
            V_k: (kXM) numpy array containing k features for each movie
        """
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
        
        U_k_slice = U[:, :k]  
        S_k_slice = S[:k]   
        Vt_k_slice = Vt[:k, :]
        
        sqrt_S_k = np.sqrt(S_k_slice)
        
        U_k = U_k_slice * sqrt_S_k
        
        V_k = sqrt_S_k[:, np.newaxis] * Vt_k_slice
        
        return U_k, V_k

    def predict(
        self,
        R: np.ndarray,
        U_k: np.ndarray,
        V_k: np.ndarray,
        users_index: dict,
        movies_index: dict,
        user_id: int,
        movies_pool: list,
        top_n: int = 3,
    ) -> np.ndarray:
        """
        Given a user specified by `user_id`, recommend the `top_n` movies that the user would want to watch among a list of movies in `movies_pool`.
        Use the compressed SVD user matrix `U_k` and movie matrix `V_k` in your prediction.

        Args:
            R: (NxM) numpy array the train dataset containing only the given user-movie ratings
            U_k: (Nxk) numpy array containing k features for each user
            V_k: (kXM) numpy array containing k features for each movie
            users_index: (N,) dictionary containing a mapping from actual `userId` to the index of the user in R (or) U_k
            movies_index: (M,) dictionary containing a mapping from actual `movieId` to the movie of the user in R (or) V_k
            user_id: (str) the user we want to recommend movies for
            movies_pool: List(str) numpy array of movie_names from which we want to select the `top_n` recommended movies
            top_n: (int) number of movies to recommend

        Return:
            recommendation: (top_n,) numpy array of movies the user with user_id would be
                            most interested in watching next and hasn't watched yet.
                            Must be a subset of `movies_pool`
        """
        mask = np.isnan(R)
        masked_array = np.ma.masked_array(R, mask)
        r_means = np.array(np.mean(masked_array, axis=0))

        u_idx = users_index[user_id]
        u_vector = U_k[u_idx, :] 
        predicted_centered_ratings = u_vector @ V_k
        predicted_full_ratings = predicted_centered_ratings + r_means
        
        recommendations = []
        for movie_name in movies_pool:
            try:
                m_id = self.get_movie_id_by_name(movie_name)
                
                if m_id not in movies_index:
                    continue
                    
                m_idx = movies_index[m_id]

                if not np.isnan(R[u_idx, m_idx]):
                    continue
                    
                rating = predicted_full_ratings[m_idx]
                recommendations.append((rating, movie_name))
                
            except KeyError:
                continue

        recommendations.sort(key=lambda x: (-x[0], x[1]))
        
        top_movies = [name for rating, name in recommendations[:top_n]]
        
        return np.array(top_movies)

    def create_ratings_matrix(
        self, ratings_df: pd.DataFrame
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        FUNCTION PROVIDED TO STUDENTS - MODIFIED FOR DETERMINISM

        Given the pandas dataframe of ratings for every user-movie pair,
        this method returns the data in the form a N*M matrix where,
        M[i][j] is the rating provided by user:(i) for movie:(j).

        Args:
            ratings_df: (pd.DataFrame) containing (userId, movieId, rating)
        """
        userList = ratings_df.iloc[:, 0].tolist()
        movieList = ratings_df.iloc[:, 1].tolist()
        ratingList = ratings_df.iloc[:, 2].tolist()
        
        users = sorted(list(set(ratings_df.iloc[:, 0])))
        movies = sorted(list(set(ratings_df.iloc[:, 1])))
        
        users_index = {users[i]: i for i in range(len(users))}
        pd_dict = {movie: [np.nan for i in range(len(users))] for movie in movies}
        for i in range(0, len(ratings_df)):
            movie = movieList[i]
            user = userList[i]
            rating = ratingList[i]
            if movie in pd_dict:
                pd_dict[movie][users_index[user]] = rating
        
        X = pd.DataFrame(pd_dict)
        X.index = users
        itemcols = list(X.columns)
        movies_index = {itemcols[i]: i for i in range(len(itemcols))}
        return np.array(X), users_index, movies_index