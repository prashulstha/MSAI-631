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
    
    def run_console_app(self):
        """Run the interactive console application"""
        print("=== FIFA Player Recommendation System ===")
        print(f"Loaded {len(self.df)} players from the database")
        print("Type 'quit' to exit the application\n")
        
        while True:
            # Get user input
            player_name = input("Enter a player name to search: ").strip()
            
            if player_name.lower() in ['quit', 'exit', 'q']:
                print("Thanks for using the FIFA Player Recommendation System!")
                break
            
            if not player_name:
                print("Please enter a valid player name.\n")
                continue
            
            # Search for the player
            matches = self.search_player(player_name)
            
            if matches.empty:
                print(f"No players found matching '{player_name}'. Try a different spelling or partial name.\n")
                continue
            
            # If multiple matches, let user choose
            if len(matches) > 1:
                print(f"\nFound {len(matches)} players matching '{player_name}':")
                for i, (idx, player) in enumerate(matches.iterrows()):
                    print(f"{i+1}. {player['short_name']} - {player['club_name']} - Overall: {player['overall']}")
                
                try:
                    choice = int(input(f"Select a player (1-{len(matches)}): ")) - 1
                    if 0 <= choice < len(matches):
                        selected_player = matches.iloc[choice]
                        player_index = matches.index[choice]
                    else:
                        print("Invalid selection.\n")
                        continue
                except ValueError:
                    print("Please enter a valid number.\n")
                    continue
            else:
                selected_player = matches.iloc[0]
                player_index = matches.index[0]
            
            # Display selected player info
            self.display_player_info(player_index)
            
            # Ask for recommendation preferences
            try:
                num_recommendations = int(input("How many similar players would you like to see? (default: 5): ") or "5")
                same_position = input("Only show players in the same position category? (y/n, default: y): ").lower()
                same_position_only = same_position != 'n'
            except ValueError:
                num_recommendations = 5
                same_position_only = True
            
            # Get and display recommendations
            similar_players = self.get_similar_players(player_index, num_recommendations, same_position_only)
            
            if not similar_players:
                print("No similar players found with the specified criteria.\n")
                continue
            
            print(f"\n=== Top {len(similar_players)} Similar Players ===")
            for i, (idx, similarity_score) in enumerate(similar_players):
                similar_player = self.df.iloc[idx]
                positions = self.extract_positions(similar_player['player_positions'])
                
                print(f"\n{i+1}. {similar_player['short_name']} (Similarity: {similarity_score:.3f})")
                print(f"   Club: {similar_player['club_name'] if pd.notna(similar_player['club_name']) else 'N/A'}")
                print(f"   Age: {similar_player['age']}, Overall: {similar_player['overall']}, Potential: {similar_player['potential']}")
                print(f"   Positions: {', '.join(positions) if positions else 'N/A'}")
                print(f"   Value: €{similar_player['value_eur']:,}" if pd.notna(similar_player['value_eur']) else "   Value: N/A")
            
            print("\n" + "="*60 + "\n")

def main():
    # Initialize the recommendation system
    try:
        # You'll need to update this path to point to your CSV file
        csv_file_path = "./data/FC26_20250921.csv"  # Update this path
        
        print("Loading FIFA player data...")
        recommender = FIFAPlayerRecommendationSystem(csv_file_path)
        
        # Run the console application
        recommender.run_console_app()
        
    except FileNotFoundError:
        print("Error: Could not find the CSV file. Please make sure 'FC26_20250921.csv' is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()