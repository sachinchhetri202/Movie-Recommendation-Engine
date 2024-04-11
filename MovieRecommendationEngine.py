import pandas as pd
from imdb import IMDb
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


def recommend_movies(selected_movie, movies_df):
    movies_df['text'] = movies_df['title'] + ' ' + movies_df['genres'].fillna('')
    vectorizer = TfidfVectorizer(stop_words='english')  # Initializing TfidfVectorizer
    tfidf_matrix = vectorizer.fit_transform(movies_df['text'])  # Fitting and transforming the text data

    selected_text = selected_movie['title'] + ' ' + (
        selected_movie['genres'] if selected_movie['genres'] is not None else '')

    # Transforming selected movie's text into tf-idf representation
    selected_tfidf = vectorizer.transform([selected_text])
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(tfidf_matrix)
    distances, indices = knn.kneighbors(selected_tfidf)

    print("\nRecommended Movies:")
    for idx in indices[0]:
        movie = movies_df.iloc[idx]
        print(f"{movie['title']} -- {movie['genres']}")


def main():
    print("\nAssignment Movie Recommendation Engine : Sachin Chhetri\n")
    movies_df = pd.read_csv('movies.csv')
    random_movies = movies_df.sample(n=10)
    print("Random Movies at Start: ")
    print(random_movies[['title', 'genres']])

    user_selection = []

    ia = IMDb()

    while True:
        print("\nOptions:")
        print("1. Search Movies")
        print("2. View Selected Movies")
        print("3. Show Recommendations for Selected Movies")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        if choice == '1':
            search_query = input("Enter movie title or year: ")
            search_results = movies_df[movies_df['title'].str.contains(search_query, case=False)]

            if len(search_results) > 0:
                print("----------------------------------------- Search Results:")
                print(search_results[['title', 'genres']])

                selection = input("\nEnter the movieId of the movie you want to select (or type 'back' to go back): ")
                if selection.lower() == 'back':
                    continue

                selected_movies = search_results.loc[int(selection)]
                user_selection.append(selected_movies)
                print(f"Selected: {selected_movies['title']}")
                print("Recommendations based on your selection:")
                recommend_movies(selected_movies, movies_df)
                print()

                movie_title = selected_movies['title']
                search_results_imdb = ia.search_movie(movie_title)
                if search_results_imdb:
                    movie = search_results_imdb[0]
                    ia.update(movie)
                    imdb_rating = movie.get('rating', 'N/A')
                    imdb_poster = movie.get('full-size cover url', 'N/A')
                    print(f"Movie Title: {movie_title}")
                    print(f"IMDb Rating: {imdb_rating}")
                    print(f"IMDb Poster: {imdb_poster}")
                else:
                    print("IMDb information is not found.")
            else:
                print("No result found based on your search.")
        elif choice == '2':
            if user_selection:
                print("Selected Movies:")
                for idx, movie in enumerate(user_selection):
                    print(f"{idx + 1}. {movie['title']}\t\t-- {movie['genres']}")
            else:
                print("No movies selected yet.")
        elif choice == '3':
            if user_selection:
                print("\nRecommendations for Selected Movies:")
                for selected_movie in user_selection:
                    print(f"\nRecommendations for '{selected_movie['title']}':")
                    recommend_movies(selected_movie, movies_df)
            else:
                print("No movies selected yet.")
        elif choice == '4':
            print("Thank you!")
            break
        else:
            print("Invalid choice. Please enter a valid option.")


if __name__ == "__main__":
    main()
