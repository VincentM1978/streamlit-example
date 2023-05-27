import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests

# Définition de la classe SessionState
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Vérification si la session d'état existe déjà
if 'session_state' not in st.session_state:
    # Création d'une instance de SessionState
    st.session_state.session_state = SessionState(df=None, count_matrix=None, cosine_sim=None)

# Chargement des données si elles ne sont pas déjà enregistrées dans la session d'état
if st.session_state.session_state.df is None:
    df = pd.read_pickle("df_fr.pkl")

    df = df.reset_index(drop=True)
    df['genres'] = df['genres'].apply(lambda x: ','.join(x))
    df['cast'] = df['cast'].apply(lambda x: ','.join(x))
    df['directors'] = df['directors'].apply(lambda x: ','.join(x))

    features = ['genres', 'title', 'cast', 'directors', 'numVotes']

    def combine_features(row):
        return row['title'] + ' ' + row['genres'] + ' ' + row['directors'] + ' ' + row['cast']

    for feature in features:
        df[feature] = df[feature].fillna('')

    df['combined_features'] = df.apply(combine_features, axis=1)

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    df['index'] = df.index

    # Enregistrement des données dans la session d'état
    st.session_state.session_state.df = df
    st.session_state.session_state.count_matrix = count_matrix
    st.session_state.session_state.cosine_sim = cosine_sim
else:
    # Récupération des données depuis la session d'état
    df = st.session_state.session_state.df
    count_matrix = st.session_state.session_state.count_matrix
    cosine_sim = st.session_state.session_state.cosine_sim

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

def get_movie_details(tconst):
    omdb_api_key = "1f7e7751"  # Replace with your OMDB API key
    url = f"http://www.omdbapi.com/?apikey={omdb_api_key}&i={tconst}"
    response = requests.get(url)
    imdb_url = "https://www.imdb.com/title/" + tconst
    data = response.json()

    if data.get("Response") == "True":
        director = data.get("Director")
        actors = data.get("Actors")
        plot = data.get("Plot")
        rating = data.get("imdbRating")
        votes = data.get("imdbVotes")
        year = data.get("Year")  # New line to retrieve the year
        return director, actors, plot, rating, votes, year, imdb_url  # Include imdb_url in the return statement
    else:
        return None  # Return None if data is not available


def get_movie_poster(tconst):
    omdb_api_key = "1f7e7751"  # Replace with your OMDB API key
    url = f"http://www.omdbapi.com/?apikey={omdb_api_key}&i={tconst}"
    response = requests.get(url)
    data = response.json()

    if data.get("Response") == "True":
        poster_url = data.get("Poster")
        return poster_url
    else:
        return None

def movie_recommender(movie_user_likes, num_recommendations):
    st.write("--------------------------")

    
    try:
        movie_index = get_index_from_title(movie_user_likes)

        # Calculate movie similarities with genre and rating weighting
        genres_user_likes = set(df.loc[movie_index, 'genres'].split(','))
        average_rating_user_likes = df.loc[movie_index, 'averageRating']
        cast_user_likes = set(df.loc[movie_index, 'cast'].split(','))
        directors_user_likes = set(df.loc[movie_index, 'directors'].split(','))
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        weighted_similar_movies = []

        # Add the weight of actors and directors
        for index, sim in similar_movies:
            movie_genres = set(df.loc[index, 'genres'].split(','))
            genre_weight = len(genres_user_likes.intersection(movie_genres))
            average_rating = df.loc[index, 'averageRating']
            rating_weight = (average_rating - average_rating_user_likes) * 0.3
            movie_cast = set(df.loc[index, 'cast'].split(','))
            cast_weight = len(cast_user_likes.intersection(movie_cast))
            movie_directors = set(df.loc[index, 'directors'].split(','))
            director_weight = len(directors_user_likes.intersection(movie_directors))
            weighted_similar_movies.append((index, sim * (1 + genre_weight * 0.7) + rating_weight + cast_weight * 0.5 + director_weight * 0.5))

        sorted_similar_movies = sorted(weighted_similar_movies, key=lambda x: x[1], reverse=True)

        for index, sim in sorted_similar_movies[1:num_recommendations+1]:
            recommended_movie_title = get_title_from_index(index)
            tconst = df.loc[index, 'tconst']
            director, actors, plot, rating, votes, year, imdb_url = get_movie_details(tconst)  # Retrieve imdb_url
            poster_url = get_movie_poster(tconst)

            col1, col2 = st.columns([2, 4])

            with col1:
                if poster_url:
                    st.image(poster_url, caption=recommended_movie_title, width=150)
                else:
                    st.write("Poster not available for", recommended_movie_title)

            with col2:
                st.markdown(f"**Title:**<a href='{imdb_url}' target='_blank'><h3>{recommended_movie_title}</h3></a>", unsafe_allow_html=True)
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Director:** {director}")
                st.markdown(f"**Actors:** {actors}")
                st.markdown(f"**Plot:** {plot}")
                st.markdown(f"**Rating:** {'★' * int((float(rating)+1) / 2)} ({rating})")
                st.markdown(f"**Votes:** {votes}")

            st.write("---")  # Add horizontal line between films

    except IndexError:
        st.write("Movie not found in the database.")


def movie_recommender_2(actor_names, num_recommendations):
    st.write("--------------------------")

    actor_names = [actor.lower() for actor in actor_names]  # Convert actor_names to lowercase

    # Find movies with matching actor names
    movies_with_actors = df[df['cast'].apply(lambda x: any(actor in x.lower() for actor in actor_names))]

    if not movies_with_actors.empty:
        sorted_movies = movies_with_actors.sort_values(by='averageRating', ascending=False).head(num_recommendations)

        for index, row in sorted_movies.iterrows():
            recommended_movie_title = row['title']
            tconst = row['tconst']
            director, actors, plot, rating, votes, year, imdb_url = get_movie_details(tconst)  # Retrieve imdb_url
            poster_url = get_movie_poster(tconst)

            col1, col2 = st.columns([2, 4])

            with col1:
                if poster_url:
                    st.image(poster_url, caption=recommended_movie_title, width=150)
                else:
                    st.write("Poster not available for", recommended_movie_title)

            with col2:
                st.markdown(f"**Title:**<a href='{imdb_url}' target='_blank'><h3>{recommended_movie_title}</h3></a>", unsafe_allow_html=True)
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Director:** {director}")
                st.markdown(f"**Actors:** {actors}")
                st.markdown(f"**Plot:** {plot}")
                st.markdown(f"**Rating:** {'★' * int((float(rating)+1) / 2)} ({rating})")
                st.markdown(f"**Votes:** {votes}")

            st.write("---")  # Add horizontal line between films

    else:
        st.write("No movies found with the actors in the database.")

# Add CSS style for sidebar background color
# Add CSS style for sidebar background color
# Add CSS style for sidebar background color and title alignment
# Add CSS style for sidebar background color and title alignment
# Add CSS style for sidebar background color and title alignment
st.markdown(
    """
    <style>
    .sidebar {
        padding: 30px;
        background-color: #D6AA57;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<h1 style='text-align: center; background-color: #0E1117; color: #D6AA57; padding: 10px; border-radius: 5px;'>★ CinéMatrix ★</h1>", unsafe_allow_html=True)
st.title("Movie Recommendation System")

# Sidebar for user input
st.sidebar.title("Search Criteria")
choice = st.sidebar.radio("Search Movies", ("By Title", "By Cast"))

#choice = "By Title"

if choice == "By Title":
    movie_titles = df['title'].tolist()
    movie_titles.insert(0,"")
    input_movie = st.sidebar.selectbox("Select a movie:", movie_titles, index=0)
    num_recommendations = st.sidebar.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    if st.sidebar.button("Get Recommendations"):
        st.sidebar.write("Movie Searched:", input_movie)
        # Retrieve movie details
        movie_index = get_index_from_title(input_movie)
        tconst = df.loc[movie_index, 'tconst']
        director, actors, plot, rating, votes, year, imdb_url = get_movie_details(tconst)
        poster_url = get_movie_poster(tconst)

        # Display movie image and information
        if poster_url:
            st.sidebar.image(poster_url, caption=input_movie, width=150)
        else:
            st.sidebar.write("Poster not available for", input_movie)
        st.sidebar.markdown(f"**Title:** {input_movie}")
        st.sidebar.markdown(f"**Year:** {year}")
        st.sidebar.markdown(f"**Director:** {director}")
        st.sidebar.markdown(f"**Actors:** {actors}")
        st.sidebar.markdown(f"**Plot:** {plot}")
        st.sidebar.markdown(f"**Rating:** {'★' * int((float(rating) + 1) / 2)} ({rating})")
        st.sidebar.markdown(f"**Votes:** {votes}")

        movie_recommender(input_movie, num_recommendations)

elif choice == "By Cast":
    # Get a list of unique actor names from the DataFrame
    actor_names = set([actor for actors in df['cast'].str.split(',') for actor in actors])

    # Autocomplete widget for actor names
    selected_actors = st.sidebar.multiselect("Select actors:", list(actor_names))
    num_recommendations = st.sidebar.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if st.sidebar.button("Get Recommendations"):
        #st.sidebar.write("Actors Selected:", selected_actors)
        movie_recommender_2(selected_actors, num_recommendations)
