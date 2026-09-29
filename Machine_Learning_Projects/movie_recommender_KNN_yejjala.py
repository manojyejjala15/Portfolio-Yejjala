# Import necessary libraries for data handling, machine learning, and evaluation
import csv

from KNN import euclidean_distance, knn


def recommend_movies(
    movie_query,
    k_recommendations,
    file_path="E:/workspace/Manoj_Portfolio/data_sets/movies.csv",
):
    """
    Recommend movies similar to a query based on KNN.

    Args:
        movie_query (list): Feature vector [rating, Action, Adventure, Comedy, Drama, Horror, Romance, SciFi, Thriller]
        k_recommendations (int): Number of movies to recommend.
        file_path (str): Path to the movies CSV file.

    Returns:
        list: List of recommended movie data [movieId, title].
    """
    # Load raw data from CSV
    raw_movies_data = []
    try:
        with open(
            file_path, "r", encoding="utf-8"
        ) as md:  # MovieLens uses UTF-8
            reader = csv.reader(md)
            next(reader)  # Skip header: movieId,title,genres
            raw_movies_data = [row for row in reader]
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    if not raw_movies_data:
        print("Error: No data found in the CSV.")
        return []

    # Define genres to match your query
    genre_list = [
        "Action",
        "Adventure",
        "Comedy",
        "Drama",
        "Horror",
        "Romance",
        "SciFi",
        "Thriller",
    ]

    # Preprocess data into feature vectors
    movies_recommendation_data = []
    movie_info = []  # Store movieId and title for output
    for row in raw_movies_data:
        movie_id, title, genres = row
        genre_set = set(genres.split("|"))
        # Create binary genre vector
        genre_vector = [1 if genre in genre_set else 0 for genre in genre_list]
        # Use a dummy rating (since movies.csv lacks it)
        rating = (
            7.0  # Placeholder; could merge with ratings.csv for real values
        )
        feature_vector = [rating] + genre_vector
        movies_recommendation_data.append(feature_vector)
        movie_info.append([movie_id, title])

    # Validate query length
    if len(movie_query) != len(movies_recommendation_data[0]):
        print(
            f"Error: Query length ({len(movie_query)}) does not match feature length ({len(movies_recommendation_data[0])})."
        )
        return []

    # Use KNN to find k nearest neighbors
    recommendation_indices, _ = knn(
        movies_recommendation_data,
        movie_query,
        k=k_recommendations,
        distance_fn=euclidean_distance,
        choice_fn=lambda x: None,
    )

    # Gather recommendations
    movie_recommendations = [
        movie_info[index] for _, index in recommendation_indices
    ]
    return movie_recommendations


if __name__ == "__main__":
    # Feature vector for "The Post": [rating, Action, Adventure, Comedy, Drama, Horror, Romance, SciFi, Thriller]
    the_post = [7.2, 1, 1, 0, 0, 0, 0, 1, 0]

    # Get 5 movie recommendations
    recommended_movies = recommend_movies(
        movie_query=the_post, k_recommendations=5
    )

    # Print results
    if recommended_movies:
        print("\nRecommended Movies:")
        for i, (movie_id, title) in enumerate(recommended_movies, 1):
            print(f"{i}. {title} (MovieID: {movie_id})")
    else:
        print("No recommendations available.")
