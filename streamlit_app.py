"""
Cross-Content Recommender — Streamlit App (Cloud entry point)

Bidirectional recommendations:
  📚 Book → Movies & Shows
  🎬 Movie/Show → Books

Local usage:
    streamlit run app/main.py

Cloud (Streamlit Cloud entry point):
    streamlit run streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Set up paths before any project imports — must come first
_root = Path(__file__).parent
sys.path.insert(0, str(_root / "src"))   # from models.x / from utils.x
sys.path.insert(0, str(_root))           # project root (project_config, etc.)

import ast

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from models.contrastive_recommender import ContrastiveRecommender

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cross-Content Recommender",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = _root


# ---------------------------------------------------------------------------
# Cached resource: cloud data setup
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading data — first run may take ~60s…")
def _setup_cloud_data() -> bool:
    """
    Download data from HuggingFace Hub on first cloud run.

    Skipped automatically when running locally (local CSV already present).

    Returns:
        True if cloud download ran, False if local data was used.
    """
    try:
        from utils.hf_data_loader import load_all_data, build_faiss_indices_if_needed
        from project_config import IS_CLOUD, EMBEDDINGS_DIR, PROJECTION_HEAD_DIR

        if not IS_CLOUD:
            return False  # Local dev — nothing to do

        load_all_data(force_refresh=False)
        build_faiss_indices_if_needed(EMBEDDINGS_DIR, PROJECTION_HEAD_DIR)
        return True

    except ImportError:
        return False  # huggingface_hub not installed; assume local
    except Exception as e:
        st.warning(f"⚠️ Data download issue: {e}")
        return False


# ---------------------------------------------------------------------------
# Cached resource: recommender
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation system…")
def load_recommender():
    """
    Load ContrastiveRecommender with projection head and both-direction indices.

    Triggers cloud data download on first run if needed.

    Returns:
        (recommender, has_projection: bool)
    """
    # Cloud: download data from HuggingFace (no-op locally)
    _setup_cloud_data()

    recommender = ContrastiveRecommender()
    recommender.load_data()

    has_projection = False
    try:
        recommender.load_projection_head()
        recommender.load_projected_index()        # 128-dim movie index
        recommender.load_projected_book_index()   # 128-dim book index
        has_projection = True
    except Exception as e:
        st.warning(f"⚠️ Contrastive model unavailable: {e}")
        st.info("ℹ️ Using baseline (embedding similarity) model.")

    # Always ensure baseline book index is available for fallback
    try:
        if recommender.book_baseline_index is None:
            recommender.load_book_baseline_index()
    except Exception as e:
        st.warning(f"⚠️ Could not load baseline book index: {e}")

    return recommender, has_projection


# ---------------------------------------------------------------------------
# Cached: Claude explanation
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate_explanation(
    direction: str,
    input_title: str,
    input_genres: str,
    result_title: str,
    result_genres: str,
    result_author: str = "",
    api_key: str = "",
) -> str:
    """
    Generate a 2-sentence cross-content explanation via Claude API.

    Cached by (direction, input_title, result_title) for 1 hour.

    Args:
        direction: "book_to_movie" or "movie_to_book"
        input_title: Query item title
        input_genres: Query item genres
        result_title: Recommended item title
        result_genres: Recommended item genres
        result_author: Author (only used in movie_to_book direction)
        api_key: Anthropic API key

    Returns:
        Explanation string, or empty string if API key absent.
    """
    if not api_key:
        return ""

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        system = (
            "You generate concise, insightful explanations for cross-content "
            "recommendations. Always respond in 2 sentences maximum."
        )

        if direction == "book_to_movie":
            user = (
                f"Book: {input_title} (genres: {input_genres}).\n"
                f"Movie/Show: {result_title} (genres: {result_genres}).\n"
                "In 2 sentences, explain why a reader of this book would enjoy "
                "this movie or show, focusing on shared themes, tone, or emotional experience."
            )
        else:  # movie_to_book
            user = (
                f"Movie/Show: {input_title} (genres: {input_genres}).\n"
                f"Book: {result_title} by {result_author} (genres: {result_genres}).\n"
                "In 2 sentences, explain why a fan of this movie or show would enjoy "
                "this book, focusing on shared themes, tone, or emotional experience."
            )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    except Exception as e:
        return f"_Could not generate explanation: {e}_"


# ---------------------------------------------------------------------------
# Genre helpers
# ---------------------------------------------------------------------------
def get_movie_genres(movie_df: pd.DataFrame) -> list:
    """Return sorted unique genres from the movies/shows catalog."""
    genres = (
        movie_df["genres"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .unique()
    )
    return sorted(g for g in genres if g and g != "\\N")


def get_book_genres(book_df: pd.DataFrame) -> list:
    """Return sorted unique genres from the books catalog."""
    genres = set()
    for val in book_df["Genres"].dropna():
        try:
            for g in ast.literal_eval(str(val)):
                genres.add(g.strip())
        except Exception:
            pass
    return sorted(genres)


# ---------------------------------------------------------------------------
# Result card renderers
# ---------------------------------------------------------------------------
def render_movie_card(row, book_title: str, book_genres: str, api_key: str):
    """Render a movie/show recommendation card."""
    expanded = row["rank"] <= 3
    rating = row.get("rating", 0.0)
    year = row.get("year", "N/A")
    try:
        rating_str = f"⭐ {float(rating):.1f}"
    except Exception:
        rating_str = ""

    with st.expander(
        f"**#{row['rank']} · {row['movie_title']} ({year})** {rating_str}",
        expanded=expanded,
    ):
        st.progress(
            min(float(row["final_score"]), 1.0),
            text=f"Match Score: {float(row['final_score']):.2%}",
        )
        st.markdown(f"**Genres:** {row['genres']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Similarity", f"{row['similarity_score']:.3f}")
        with col_b:
            st.metric("Genre Bonus", f"{row['genre_bonus']:.3f}")

        if row.get("description"):
            st.markdown(f"**Plot:** {row['description']}")

        if api_key:
            with st.spinner("Generating explanation…"):
                explanation = generate_explanation(
                    direction="book_to_movie",
                    input_title=book_title,
                    input_genres=book_genres,
                    result_title=row["movie_title"],
                    result_genres=str(row["genres"]),
                    api_key=api_key,
                )
            if explanation and not explanation.startswith("_Could not"):
                st.info(f"💡 {explanation}")
            elif explanation.startswith("_Could not"):
                st.warning(explanation)


def render_book_card(row, movie_title: str, movie_genres: str, api_key: str):
    """Render a book recommendation card."""
    expanded = row["rank"] <= 3

    with st.expander(
        f"**#{row['rank']} · {row['book_title']}** — {row['author']}",
        expanded=expanded,
    ):
        st.progress(
            min(float(row["final_score"]), 1.0),
            text=f"Match Score: {float(row['final_score']):.2%}",
        )
        st.markdown(f"**Genres:** {row['genres']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Similarity", f"{row['similarity_score']:.3f}")
        with col_b:
            st.metric("Genre Bonus", f"{row['genre_bonus']:.3f}")

        if row.get("description"):
            st.markdown(f"**About:** {row['description']}")

        if api_key:
            with st.spinner("Generating explanation…"):
                explanation = generate_explanation(
                    direction="movie_to_book",
                    input_title=movie_title,
                    input_genres=movie_genres,
                    result_title=row["book_title"],
                    result_genres=str(row["genres"]),
                    result_author=str(row["author"]),
                    api_key=api_key,
                )
            if explanation and not explanation.startswith("_Could not"):
                st.info(f"💡 {explanation}")
            elif explanation.startswith("_Could not"):
                st.warning(explanation)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    recommender, has_projection = load_recommender()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    banner_path = PROJECT_ROOT / "app" / "assets" / "header_banner.png"
    if banner_path.exists():
        st.image(str(banner_path))

    st.title("Cross-Content Recommender")
    st.markdown(
        "_Discover movies & shows from books, or find books behind your favourite screen stories._"
    )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model toggle
        if has_projection:
            model_choice = st.radio(
                "Recommendation Model",
                ["Baseline (Embeddings Only)", "Contrastive (Trained Alignment)"],
                index=1,
                help=(
                    "Baseline: raw content similarity. "
                    "Contrastive: trained projection for better cross-domain alignment."
                ),
            )
            use_projection = model_choice == "Contrastive (Trained Alignment)"
        else:
            use_projection = False
            st.info("ℹ️ Contrastive model unavailable. Using baseline.")

        # Results count
        k = st.slider("Number of results", min_value=5, max_value=20, value=10)

        st.divider()

        # Claude API key
        st.subheader("🤖 AI Explanations")
        api_key_input = st.text_input(
            "Anthropic API Key (optional)",
            type="password",
            placeholder="sk-ant-…",
            help="AI-generated 2-sentence explanations for each recommendation.",
        )
        api_key = api_key_input or os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            st.success("✅ Explanations enabled")
        else:
            st.info("💡 Add API key for AI explanations")

        st.divider()

        # Catalog stats
        st.caption(f"📚 {len(recommender.book_df):,} books")
        st.caption(f"🎬 {len(recommender.movie_df):,} movies/shows")

        st.divider()

        # Data disclaimer
        st.caption(
            "**Dataset note:** Covers titles up to 2022. "
            "Recent movies/shows may not appear. "
            "Includes both films and TV series sourced from IMDB."
        )
        st.caption(
            "Data hosted on [HuggingFace](https://huggingface.co/datasets/"
            "ishijo/cross-content-recommender-data) — "
            "ishijo/cross-content-recommender-data"
        )

    # ------------------------------------------------------------------
    # Direction toggle
    # ------------------------------------------------------------------
    direction = st.radio(
        "Choose direction",
        ["📚 Book → Movies & Shows", "🎬 Movie/Show → Books"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_book_to_movie = direction == "📚 Book → Movies & Shows"

    # Clear stale results when direction changes
    if st.session_state.get("_direction") != direction:
        for key in ["recommendations", "input_info", "input_title"]:
            st.session_state.pop(key, None)
        st.session_state["_direction"] = direction

    st.divider()

    # ------------------------------------------------------------------
    # Genre filter (post-filter on results)
    # ------------------------------------------------------------------
    if is_book_to_movie:
        all_genres = get_movie_genres(recommender.movie_df)
        genre_label = "Filter by genre (movies/shows)"
    else:
        all_genres = get_book_genres(recommender.book_df)
        genre_label = "Filter by genre (books)"

    selected_genres = st.multiselect(genre_label, all_genres)

    st.divider()

    # ------------------------------------------------------------------
    # API key warning (once, not per-card)
    # ------------------------------------------------------------------
    if not api_key:
        st.info(
            "ℹ️ No Anthropic API key detected — recommendations will display "
            "without AI explanations. Add your key in the sidebar to enable them."
        )

    # ------------------------------------------------------------------
    # Direction A: Book → Movies & Shows
    # ------------------------------------------------------------------
    if is_book_to_movie:
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("📖 Enter a Book")
            book_title = st.text_input(
                "Book title",
                placeholder="e.g. Gone Girl, Dune, The Girl on the Train",
                label_visibility="collapsed",
            )
            search_btn = st.button(
                "🔍 Find Movies & Shows", type="primary", use_container_width=True
            )

            # Show book info if available
            if "input_info" in st.session_state and st.session_state["input_info"] is not None:
                info = st.session_state["input_info"]
                st.markdown("---")
                st.markdown("**Selected Book:**")
                st.markdown(f"📚 _{info['Book']}_")
                if pd.notna(info.get("Author")):
                    st.markdown(f"✍️ {info['Author']}")
                if pd.notna(info.get("Avg_Rating")):
                    st.markdown(f"⭐ {float(info['Avg_Rating']):.1f}/5.0")
                try:
                    genres_list = ast.literal_eval(str(info.get("Genres", "[]")))
                    if genres_list:
                        st.markdown(f"🏷️ {', '.join(genres_list[:5])}")
                except Exception:
                    pass
                if info.get("Description"):
                    st.markdown(
                        f"_{str(info['Description'])[:200]}…_"
                    )

        with col2:
            st.subheader("🎬 Recommended Movies & Shows")

            if search_btn:
                if not book_title:
                    st.error("⚠️ Please enter a book title.")
                else:
                    with st.spinner("Finding movies & shows…"):
                        try:
                            recs = recommender.recommend_movies(
                                book_title,
                                k=k,
                                genre_filter=True,
                                use_projection=use_projection,
                            )

                            if len(recs) == 0:
                                st.error(f"❌ Book '{book_title}' not found.")
                                similar = recommender.book_df[
                                    recommender.book_df["Book"].str.contains(
                                        book_title.split()[0], case=False, na=False
                                    )
                                ].head(3)
                                if len(similar):
                                    st.info("💡 Did you mean:")
                                    for _, r in similar.iterrows():
                                        st.write(f"• {r['Book']}")
                            else:
                                st.session_state["recommendations"] = recs
                                st.session_state["input_title"] = book_title
                                st.session_state["api_key"] = api_key
                                book_info = recommender.get_book_info(book_title)
                                st.session_state["input_info"] = book_info
                                st.success(f"✅ Found {len(recs)} movies & shows!")

                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            # Display results
            if "recommendations" in st.session_state and len(st.session_state["recommendations"]) > 0:
                recs = st.session_state["recommendations"]
                stored_title = st.session_state.get("input_title", "")
                stored_key = st.session_state.get("api_key", "")
                book_info = st.session_state.get("input_info")

                # Parse book genres for explanations
                book_genres_str = ""
                if book_info is not None and "Genres" in book_info:
                    try:
                        book_genres_str = ", ".join(
                            ast.literal_eval(str(book_info["Genres"]))
                        )
                    except Exception:
                        book_genres_str = str(book_info.get("Genres", ""))

                # Apply genre post-filter
                display_recs = recs
                if selected_genres:
                    mask = display_recs["genres"].apply(
                        lambda g: any(sg in str(g) for sg in selected_genres)
                    )
                    display_recs = display_recs[mask]

                if len(display_recs) == 0:
                    st.warning("No movies/shows match the selected genre filter.")
                else:
                    for _, row in display_recs.iterrows():
                        render_movie_card(row, stored_title, book_genres_str, stored_key)
            else:
                st.info("👆 Enter a book title and click 'Find Movies & Shows'.")

    # ------------------------------------------------------------------
    # Direction B: Movie/Show → Books
    # ------------------------------------------------------------------
    else:
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("🎬 Enter a Movie or Show")
            movie_title = st.text_input(
                "Movie or show title",
                placeholder="e.g. Inception, Breaking Bad, The Crown",
                label_visibility="collapsed",
            )
            search_btn = st.button(
                "🔍 Find Books", type="primary", use_container_width=True
            )

            # Show movie info if available
            if "input_info" in st.session_state and st.session_state["input_info"] is not None:
                info = st.session_state["input_info"]
                st.markdown("---")
                st.markdown("**Selected Movie/Show:**")
                st.markdown(f"🎬 _{info.get('primaryTitle', '')}_ ({info.get('startYear', 'N/A')})")
                if info.get("genres"):
                    st.markdown(f"🏷️ {info['genres']}")
                if info.get("Description"):
                    st.markdown(f"_{str(info['Description'])[:200]}…_")

        with col2:
            st.subheader("📚 Recommended Books")

            if search_btn:
                if not movie_title:
                    st.error("⚠️ Please enter a movie or show title.")
                else:
                    with st.spinner("Finding books…"):
                        try:
                            recs = recommender.recommend_books_from_movie(
                                movie_title,
                                k=k,
                                use_projection=use_projection,
                            )

                            # Fetch movie info for display
                            try:
                                movie_row, _ = recommender._find_movie_by_title(movie_title)
                                st.session_state["input_info"] = movie_row.to_dict()
                                movie_genres_str = str(movie_row.get("genres", ""))
                            except Exception:
                                st.session_state["input_info"] = None
                                movie_genres_str = ""

                            st.session_state["recommendations"] = recs
                            st.session_state["input_title"] = movie_title
                            st.session_state["api_key"] = api_key
                            st.session_state["movie_genres"] = movie_genres_str
                            st.success(f"✅ Found {len(recs)} books!")

                        except ValueError as e:
                            st.error(f"❌ {e}")
                            # Suggest similar titles
                            first_word = movie_title.split()[0] if movie_title else ""
                            if first_word:
                                similar = recommender.movie_df[
                                    recommender.movie_df["primaryTitle"].str.contains(
                                        first_word, case=False, na=False
                                    )
                                ].head(3)
                                if len(similar):
                                    st.info("💡 Did you mean:")
                                    for _, r in similar.iterrows():
                                        st.write(
                                            f"• {r['primaryTitle']} ({r.get('startYear', 'N/A')})"
                                        )
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            # Display results
            if "recommendations" in st.session_state and len(st.session_state["recommendations"]) > 0:
                recs = st.session_state["recommendations"]
                stored_title = st.session_state.get("input_title", "")
                stored_key = st.session_state.get("api_key", "")
                movie_genres_str = st.session_state.get("movie_genres", "")

                # Apply genre post-filter
                display_recs = recs
                if selected_genres:
                    mask = display_recs["genres"].apply(
                        lambda g: any(sg in str(g) for sg in selected_genres)
                    )
                    display_recs = display_recs[mask]

                if len(display_recs) == 0:
                    st.warning("No books match the selected genre filter.")
                else:
                    for _, row in display_recs.iterrows():
                        render_book_card(row, stored_title, movie_genres_str, stored_key)
            else:
                st.info("👆 Enter a movie or show title and click 'Find Books'.")

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Built with Streamlit · Powered by Sentence Transformers & Contrastive Learning"
    )


if __name__ == "__main__":
    main()
