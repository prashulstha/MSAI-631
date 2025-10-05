import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import re

class FIFAPlayerRecommendationSystem:
    def __init__(self, csv_file_path):
        """Initialize the recommendation system with FIFA player data"""
        self.df = pd.read_csv(csv_file_path)
        self.scaler = StandardScaler()
        self.similarity_matrix = None
        self.feature_columns = []
        self.prepare_data()
    
    def get_position_weights(position_category):
        """Return feature weights based on position"""
        if position_category == 'Forward':
            return {
                'pace': 1.5,
                'shooting': 2.0,
                'dribbling': 1.5,
                'passing': 1.0,
                'defending': 0.3,
                'physic': 1.0
            }
        elif position_category == 'Defender':
            return {
                'pace': 1.0,
                'shooting': 0.3,
                'dribbling': 0.5,
                'passing': 1.0,
                'defending': 2.0,
                'physic': 1.5
            }
        
    def prepare_data(self):
        """Prepare and clean the data for recommendations"""
        # Remove players with missing essential data
        self.df = self.df.dropna(subset=['short_name', 'player_positions', 'overall'])
        
        # Define feature columns for similarity calculation
        # Focus on key stats that define player abilities
        self.feature_columns = [
            'overall', 'potential', 'age', 'height_cm', 'weight_kg',
            'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
            'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
            'attacking_short_passing', 'skill_dribbling', 'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
            'power_shot_power', 'power_stamina', 'power_strength',
            'mentality_positioning', 'mentality_vision', 'defending_marking_awareness'
        ]
        
        # Fill missing values with median for numerical columns
        for col in self.feature_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Create normalized feature matrix
        feature_data = self.df[self.feature_columns].values
        self.normalized_features = self.scaler.fit_transform(feature_data)
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.normalized_features)
        
    def extract_positions(self, position_string):
        """Extract primary positions from position string"""
        if pd.isna(position_string):
            return []
        # Split by comma and clean up
        positions = [pos.strip() for pos in str(position_string).split(',')]
        return positions
    
    def get_position_category(self, positions):
        """Categorize positions into broader groups"""
        if not positions:
            return "Unknown"
            
        position_str = str(positions[0]).upper()
        
        # Goalkeepers
        if 'GK' in position_str:
            return 'Goalkeeper'
        
        # Defenders
        defender_positions = ['CB', 'LB', 'RB', 'LCB', 'RCB', 'LWB', 'RWB']
        if any(pos in position_str for pos in defender_positions):
            return 'Defender'
        
        # Midfielders
        midfielder_positions = ['CM', 'CDM', 'CAM', 'LM', 'RM', 'LCM', 'RCM', 'LAM', 'RAM', 'LDM', 'RDM']
        if any(pos in position_str for pos in midfielder_positions):
            return 'Midfielder'
        
        # Forwards
        forward_positions = ['ST', 'CF', 'LW', 'RW', 'LF', 'RF', 'LS', 'RS']
        if any(pos in position_str for pos in forward_positions):
            return 'Forward'
        
        return 'Unknown'
    
    def search_player(self, player_name):
        """Search for a player by name (case insensitive, partial match)"""
        matches = self.df[self.df['short_name'].str.contains(player_name, case=False, na=False)]
        
        if matches.empty:
            # Try long name if short name doesn't match
            matches = self.df[self.df['long_name'].str.contains(player_name, case=False, na=False)]
        
        return matches
    
    def get_similar_players(self, player_index, num_recommendations=5, same_position_only=True):
        """Get similar players based on cosine similarity"""
        if player_index >= len(self.similarity_matrix):
            return []
        
        # Get similarity scores for the player
        sim_scores = self.similarity_matrix[player_index]
        
        # Get the target player's position category
        target_positions = self.extract_positions(self.df.iloc[player_index]['player_positions'])
        target_position_category = self.get_position_category(target_positions)
        
        # Create list of (index, similarity_score) tuples
        similar_indices = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != player_index]
        
        # Filter by position if requested
        if same_position_only and target_position_category != 'Unknown':
            filtered_indices = []
            for idx, score in similar_indices:
                player_positions = self.extract_positions(self.df.iloc[idx]['player_positions'])
                player_category = self.get_position_category(player_positions)
                if player_category == target_position_category:
                    filtered_indices.append((idx, score))
            similar_indices = filtered_indices
        
        # Sort by similarity score (descending)
        similar_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N recommendations
        return similar_indices[:num_recommendations]
    
    def display_player_info(self, player_index):
        """Display detailed information about a player"""
        player = self.df.iloc[player_index]
        positions = self.extract_positions(player['player_positions'])
        position_category = self.get_position_category(positions)
        
        print(f"\n--- Player Information ---")
        print(f"Name: {player['short_name']} ({player['long_name']})")
        print(f"Age: {player['age']}")
        print(f"Overall Rating: {player['overall']}")
        print(f"Potential: {player['potential']}")
        print(f"Positions: {', '.join(positions) if positions else 'N/A'}")
        print(f"Position Category: {position_category}")
        print(f"Club: {player['club_name'] if pd.notna(player['club_name']) else 'N/A'}")
        print(f"Nationality: {player['nationality_name']}")
        print(f"Value: €{player['value_eur']:,}" if pd.notna(player['value_eur']) else "Value: N/A")
        print(f"Key Stats - Pace: {player['pace']:.0f}, Shooting: {player['shooting']:.0f}, "
              f"Passing: {player['passing']:.0f}, Dribbling: {player['dribbling']:.0f}, "
              f"Defending: {player['defending']:.0f}, Physical: {player['physic']:.0f}")
        print("-" * 50)
    
    def run_gui_app(self):
        """Run a Streamlit GUI for player recommendations (resembles fifa_rec_ui.py)"""
        import streamlit as st
        # Use short_name for selection, fallback to long_name if needed
        st.set_page_config(page_title="AI Adaptive Recommender", layout="wide")
        st.title("⚽ Adaptive Player Recommendation System")
        st.write("""
        This intelligent system uses **AI-based adaptive interaction** to recommend football players similar to your selection.  
        It adapts the interface, explanations, and visuals dynamically based on similarity confidence and user feedback.
        """)

        with st.sidebar:
            st.header("🔎 Choose Player")
            player_names = self.df['short_name'].dropna().unique()
            selected_player = st.selectbox("Select a player:", player_names)
            num_recommendations = st.slider("Number of Recommendations:", 3, 10, 5)

        def get_recommendations(player_name, top_n=5):
            matches = self.df[self.df['short_name'] == player_name]
            if matches.empty:
                return [], []
            idx = matches.index[0]
            recs = self.get_similar_players(idx, num_recommendations=top_n, same_position_only=True)
            if not recs:
                return [], []
            indices, scores = zip(*recs)
            return self.df.iloc[list(indices)], list(scores)

        if selected_player:
            recs, scores = get_recommendations(selected_player, num_recommendations)
            selected_idx = self.df[self.df['short_name'] == selected_player].index[0]
            selected_features = self.df.loc[selected_idx, self.feature_columns]

            # Adaptive Feedback Section
            import numpy as np
            avg_similarity = np.mean(scores) if scores else 0
            if avg_similarity > 0.8:
                st.success(f"✅ High confidence recommendations for **{selected_player}** (Avg. similarity: {avg_similarity:.2f})")
            elif 0.6 <= avg_similarity <= 0.8:
                st.warning(f"⚠️ Moderate similarity detected. Some results may differ in playstyle.")
            else:
                st.error(f"❌ Low similarity found. Consider adjusting feature weights.")

            # Show Recommendations
            st.subheader("🏆 Recommended Players")
            for i, (index, score) in enumerate(zip(recs.index if hasattr(recs, 'index') else [], scores)):
                player_info = self.df.loc[index]
                st.markdown(f"**{i+1}. {player_info['short_name']}** — Similarity: `{score:.2f}`")
                st.progress(float(score))
                st.caption(f"Position: {player_info.get('player_positions', 'N/A')} | Club: {player_info.get('club_name', 'Unknown')}")

            # Comparison Dashboard
            st.subheader("📊 Compare Attributes")
            if hasattr(recs, 'short_name') and not recs.empty:
                compare_player = st.selectbox("Compare selected player with:", recs['short_name'])
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {selected_player}")
                    st.bar_chart(selected_features)
                with col2:
                    st.markdown(f"### {compare_player}")
                    compare_features = self.df.loc[self.df['short_name'] == compare_player, self.feature_columns].T
                    st.bar_chart(compare_features)

            # User Feedback (Adaptive Learning Stub)
            st.subheader("💬 Provide Feedback")
            feedback = st.radio(
                "Are these recommendations useful?",
                ("Yes", "No", "Partially"),
                horizontal=True
            )
            if feedback:
                st.write("Thank you for your feedback. The system will adapt future results accordingly (simulated learning).")
            if feedback == "No":
                st.info("System would decrease the weight of selected features and re-tune similarity measures in future versions.")
            elif feedback == "Yes":
                st.success("System would reinforce feature weightings that contributed most to the recommendation success.")

        st.markdown("---")
        st.markdown("#### ⚖️ Ethical & Human-Centered AI Note")
        st.write("""
        This system demonstrates **AI-based Adaptive Human-Computer Interaction (HCI)**, where the interface adapts based on context, confidence, and feedback.
        Such systems must remain transparent, user-controllable, and explainable to maintain ethical integrity in AI-assisted recommendations.
        """)
    
def main():
    # Initialize the recommendation system
    try:
        # You'll need to update this path to point to your CSV file
        csv_file_path = "./data/FC26_20250921.csv"  # Update this path
        
        print("Loading FIFA player data...")
        recommender = FIFAPlayerRecommendationSystem(csv_file_path)
        
        # Run the GUI application
        recommender.run_gui_app()
        
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Please make sure 'FC26_20250921.csv' is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()