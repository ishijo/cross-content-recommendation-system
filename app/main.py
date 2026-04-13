"""
Cross-Content Movie Recommendations - Streamlit App

Interactive web application for getting movie recommendations based on books.
Supports both baseline and contrastive learning models with optional Claude API explanations.

Usage:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import streamlit as st
import pandas as pd
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.contrastive_recommender import ContrastiveRecommender

# Page config
st.set_page_config(
    page_title="Cross-Content Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource(show_spinner="Loading recommendation system...")
def load_recommender():
    """
    Load contrastive recommender with projection head.

    Returns:
        (recommender, has_projection_flag)
    """
    recommender = ContrastiveRecommender()

    # Load baseline data
    recommender.load_data()

    # Try to load projection head
    has_projection = False
    try:
        recommender.load_projection_head()
        recommender.load_projected_index()
        has_projection = True
    except Exception as e:
        st.warning(f"⚠️ Could not load projection head: {e}")
        st.info("ℹ️ Using baseline model only.")

    return recommender, has_projection


@st.cache_data(ttl=3600, show_spinner=False)
def generate_explanation(book_title: str, book_genres: str,
                        movie_title: str, movie_genres: str,
                        api_key: str) -> str:
    """
    Generate 2-sentence explanation for recommendation using Claude API.

    Cached for 1 hour to avoid redundant API calls.

    Args:
        book_title: Book title
        book_genres: Book genres
        movie_title: Movie title
        movie_genres: Movie genres
        api_key: Anthropic API key

    Returns:
        Explanation text or error message
    """
    if not api_key:
        return ""

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        prompt = f"""Explain in exactly 2 sentences why someone who enjoyed the book "{book_title}" ({book_genres}) might also enjoy the movie "{movie_title}" ({movie_genres}). Focus on shared themes, tone, or emotional experience."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    except Exception as e:
        return f"_Could not generate explanation: {str(e)}_"


def main():
    """Main app"""

    # Load recommender
    recommender, has_projection = load_recommender()

    # Title
    st.title("🎬 Cross-Content Movie Recommendations")
    st.markdown("_Find movies based on books you love using AI-powered recommendations_")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model toggle
        if has_projection:
            model_choice = st.radio(
                "Recommendation Model",
                ["Baseline (Content Similarity)", "Contrastive Learning (Advanced)"],
                index=1,
                help="Baseline uses raw content similarity. Contrastive uses trained projection for better cross-domain alignment."
            )
            use_projection = model_choice == "Contrastive Learning (Advanced)"
        else:
            use_projection = False
            st.info("ℹ️ Contrastive model not available. Using baseline.")

        # K slider
        k = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            help="How many movie recommendations to show"
        )

        # Genre filter
        genre_filter = st.checkbox(
            "Enable genre filtering",
            value=True,
            help="Re-rank recommendations based on genre overlap with the book"
        )

        st.divider()

        # Claude API key
        st.subheader("🤖 AI Explanations")
        api_key_input = st.text_input(
            "Anthropic API Key (optional)",
            type="password",
            placeholder="sk-ant-...",
            help="Add your Anthropic API key to get AI-generated explanations for each recommendation"
        )

        # Use env variable if available
        api_key = api_key_input or os.getenv("ANTHROPIC_API_KEY", "")

        if api_key:
            st.success("✅ Explanations enabled")
        else:
            st.info("💡 Add API key for AI explanations")

        st.divider()

        # Stats
        st.caption(f"📚 {len(recommender.book_df):,} books")
        st.caption(f"🎬 {len(recommender.movie_df):,} movies")

    # Main content
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("📖 Enter a Book")

        book_title = st.text_input(
            "Book Title",
            placeholder="e.g., Harry Potter and the Sorcerer's Stone",
            label_visibility="collapsed"
        )

        search_button = st.button(
            "🔍 Find Movies",
            type="primary",
            use_container_width=True
        )

        # Display book info if recommendations exist
        if 'book_info' in st.session_state and st.session_state['book_info'] is not None:
            book_info = st.session_state['book_info']
            st.markdown("---")
            st.markdown("**Selected Book:**")
            st.markdown(f"📚 _{book_info['Book']}_")
            if 'Author' in book_info and pd.notna(book_info['Author']):
                st.markdown(f"✍️ {book_info['Author']}")
            if 'Avg_Rating' in book_info and pd.notna(book_info['Avg_Rating']):
                rating = float(book_info['Avg_Rating'])
                st.markdown(f"⭐ {rating:.1f}/5.0")

    with col2:
        st.subheader("🎬 Recommended Movies")

        # Handle search
        if search_button:
            if not book_title:
                st.error("⚠️ Please enter a book title")
            else:
                with st.spinner("Searching for recommendations..."):
                    try:
                        # Get recommendations
                        recommendations = recommender.recommend_movies(
                            book_title,
                            k=k,
                            genre_filter=genre_filter,
                            use_projection=use_projection
                        )

                        if len(recommendations) == 0:
                            # Book not found - suggest alternatives
                            st.error(f"❌ Book '{book_title}' not found")

                            # Find 3 similar titles
                            similar = recommender.book_df[
                                recommender.book_df['Book'].str.contains(
                                    book_title.split()[0],
                                    case=False,
                                    na=False
                                )
                            ].head(3)

                            if len(similar) > 0:
                                st.info("💡 Did you mean:")
                                for _, row in similar.iterrows():
                                    st.write(f"• {row['Book']}")
                        else:
                            # Store in session state
                            st.session_state['recommendations'] = recommendations
                            st.session_state['book_title'] = book_title
                            st.session_state['api_key'] = api_key

                            # Get book info
                            book_info = recommender.get_book_info(book_title)
                            st.session_state['book_info'] = book_info

                            st.success(f"✅ Found {len(recommendations)} recommendations!")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # Display recommendations
        if 'recommendations' in st.session_state and len(st.session_state['recommendations']) > 0:
            recs = st.session_state['recommendations']
            book_title = st.session_state['book_title']
            api_key = st.session_state.get('api_key', '')
            book_info = st.session_state.get('book_info')

            # Get book genres for explanations
            book_genres = ""
            if book_info is not None and 'Genres' in book_info:
                try:
                    import ast
                    book_genres = ", ".join(ast.literal_eval(str(book_info['Genres'])))
                except:
                    book_genres = str(book_info.get('Genres', ''))

            for _, row in recs.iterrows():
                # Expandable card (top 3 expanded by default)
                expanded = row['rank'] <= 3

                with st.expander(
                    f"**#{row['rank']} · {row['movie_title']} ({row['year']})** ⭐ {row['rating']:.1f}",
                    expanded=expanded
                ):
                    # Similarity bar
                    st.progress(
                        min(row['final_score'], 1.0),
                        text=f"Match Score: {row['final_score']:.2%}"
                    )

                    # Genres
                    st.markdown(f"**Genres:** {row['genres']}")

                    # Scores breakdown
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Similarity", f"{row['similarity_score']:.3f}")
                    with col_b:
                        st.metric("Genre Bonus", f"{row['genre_bonus']:.3f}")

                    # Description
                    st.markdown(f"**Plot:** {row['description']}")

                    # AI explanation
                    if api_key:
                        with st.spinner("Generating explanation..."):
                            explanation = generate_explanation(
                                book_title,
                                book_genres,
                                row['movie_title'],
                                row['genres'],
                                api_key
                            )

                            if explanation and not explanation.startswith("_Could not"):
                                st.info(f"💡 {explanation}")
                            elif explanation.startswith("_Could not"):
                                st.warning(explanation)

        else:
            st.info("👆 Enter a book title and click 'Find Movies' to see recommendations")

    # Footer
    st.markdown("---")
    st.caption("Built with Streamlit • Powered by Sentence Transformers & Contrastive Learning")


if __name__ == "__main__":
    main()
