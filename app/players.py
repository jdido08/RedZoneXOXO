import pandas as pd
import numpy as np
from roboflow import Roboflow
import cv2
import supervision as sv
from scipy.spatial import cKDTree


import yaml
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    ROBOFLOW_KEY = config['roboflow_key']

#### DETECT PLAYERS ###
def detect_players(source_path):
    # Initialization
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace().project("player_dec")
    model = project.version(1).model

    byte_tracker = sv.ByteTrack()  # Initialize with default parameters
    annotator = sv.BoxAnnotator()

    # Initialize accumulator for all detections
    all_detections = []

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # Extract frame dimensions
        height, width, _ = frame.shape

        # Get predictions for the current frame
        predictions = model.predict(frame, confidence=60, overlap=60).json()["predictions"]
        
        # Convert predictions to the expected format for sv.Detections
        detections_list = [
            [
                pred["x"], 
                pred["y"], 
                pred["width"], 
                pred["height"], 
                pred["confidence"], 
                project.classes[pred["class"].lower()]  # Fetch the class ID using the class name
            ] 
            for pred in predictions
        ]

        # If there are no detections, return the original frame
        if not detections_list:
            return frame

        # Convert bounding box from center to top-left and bottom-right format
        xyxy = [[det[0] - det[2] / 2, det[1] - det[3] / 2, det[0] + det[2] / 2, det[1] + det[3] / 2] for det in detections_list]
        
        confidence = [detection[4] for detection in detections_list]
        class_id = [detection[5] for detection in detections_list]

        # Convert to numpy arrays
        xyxy = np.array(xyxy)
        confidence = np.array(confidence)
        class_id = np.array(class_id)

        # Create Detections object
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        detections = byte_tracker.update_with_detections(detections)

        # Append each detection data to all_detections
        for xyxy, _, conf, class_id_value, tracker_id in detections:
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            class_name = next(key for key, value in project.classes.items() if value == class_id_value)  # Map class ID back to class name
            all_detections.append({
                'frame': index,
                'x': x1,
                'y': y1,
                'width': w,
                'height': h,
                'confidence': conf,
                'class': class_name,
                'tracker_id': tracker_id,
                'frame_width': width,
                'frame_height':height
            })
        
        # Prepare labels for annotation
        labels = [
            f"#{tracker_id} {key} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id in detections
            for key, value in project.classes.items() if value == class_id  # Map class ID back to class name
        ]
        
        # Annotate the frame with detections and labels
        return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

    # Process the Video
    sv.process_video(
        source_path=source_path,
        target_path=source_path + "_annotated.mp4",
        callback=callback
    )

    # Convert accumulated detections to a DataFrame
    df = pd.DataFrame(all_detections)
    df['x'] = (df['x'] + (df['width']/2))
    df['y'] = (df['y'] + (df['height']/2))
    df['label'] = None
    df = df[['frame', 'tracker_id', 'x', 'y', 'label' ]]
    #df.to_csv('test_dataframe_players_9_30_2023.csv')
    return df

### TRANFORM PLAYERS ####
def compute_homography(df_field_points):
   
    # Extract image points and field points
    image_points = df_field_points[['x', 'y']].values.astype('float32')
    field_points = df_field_points[['field_x', 'field_y']].values.astype('float32')
    
    # Compute the homography matrix
    H, _ = cv2.findHomography(image_points, field_points)
    return H

def transform_player_positions(df_players, df_field):
    # Compute the homography matrix
    H = compute_homography(df_field)

    # Extract the center coordinates for homography transformation
    player_points = df_players[['x', 'y']].values.astype('float32')
    
    # Apply the homography transformation
    field_positions = cv2.perspectiveTransform(player_points.reshape(-1, 1, 2), H)
    
    # Split the transformed points into field_x and field_y columns
    df_players['field_x'] = field_positions[:, 0, 0]
    df_players['field_y'] = field_positions[:, 0, 1]
    
    return df_players[['frame', 'tracker_id', 'field_x', 'field_y', 'label']]

### TRACK PLAYERS ####


# Function to fill all gaps using linear interpolation for each tracker
def fill_all_gaps(data):
    # Detect where the gaps are
    gap_starts = np.where(data['frame'].diff() > 1)[0]
    
    # For each gap, interpolate
    for start in gap_starts:
        gap_size = int(data.iloc[start]['frame'] - data.iloc[start - 1]['frame'] - 1)
        
        # Interpolating field_x
        x_start = data.iloc[start - 1]['field_x']
        x_end = data.iloc[start]['field_x']
        x_interp = np.linspace(x_start, x_end, gap_size + 2)[1:-1]
        
        # Interpolating field_y
        y_start = data.iloc[start - 1]['field_y']
        y_end = data.iloc[start]['field_y']
        y_interp = np.linspace(y_start, y_end, gap_size + 2)[1:-1]
        
        # Interpolating frames
        frame_start = data.iloc[start - 1]['frame']
        frame_interp = np.arange(frame_start + 1, frame_start + gap_size + 1)
        
        # Creating the interpolated dataframe
        df_interp = pd.DataFrame({
            'frame': frame_interp,
            'tracker_id': [data.iloc[start]['tracker_id']] * gap_size,
            'field_x': x_interp,
            'field_y': y_interp,
            'label': [None] * gap_size
        })
        
        # Concatenating to the main dataframe
        data = pd.concat([data.iloc[:start], df_interp, data.iloc[start:]]).reset_index(drop=True)
    
    return data

def find_player_team(label):
    team = None
    if label in ('QB', 'WR', 'RB', 'FB', 'TE', 'C', 'G', 'T'):
        team = 'offense'
    elif label in ('DL', 'LB', 'DB'):
        team = 'defense'
    return team

def init_player_data(x, y, label):
    team = find_player_team(label)
    return {'positions': [(x, y)], 'label': label, 'team':team, 'associated_tracker_ids':[]}

def max_speed_in_ypf(position, fps=23):
    """
    Determine the max speed of a player in yards per frame based on their position and video fps.
    
    Args:
    - position (str): Player's position.
    - fps (int): Frames per second of the video.
    
    Returns:
    - float: Max speed in yards per frame.
    """
    
    # Updated player max speeds in mph based on typical estimates
    player_max_speeds_mph = {
        "QB": 16.5,  # Average of 15-18 mph
        "RB": 21,    # Average of 20-22 mph
        "WR": 21,    # Average of 20-22 mph
        "TE": 19,    # Average of 18-20 mph
        "FB": 17,    # Average of 16-18 mph
        "T":  13.5,  # Average of 12-15 mph
        "C":  13.5,  # Average of 12-15 mph
        "G":  13.5,  # Average of 12-15 mph
        "OL": 13.5,  # Average of 12-15 mph
        "DL": 13.5,  # Average of 12-15 mph
        "LB": 19,    # Average of 18-20 mph
        "DB": 21     # Average of 20-22 mph
    }
    
    # Convert from mph to ypf
    max_speed_mph = player_max_speeds_mph.get(position, 0)
    max_speed_yps = (max_speed_mph * 1760) / 3600  # Convert mph to yps
    max_speed_ypf = max_speed_yps / fps   # Convert yps to ypf
    
    return max_speed_ypf

#basic function used to predict field posistion when known player posistion was unknown 
def extrapolate_position(positions, label, fps=23.0, max_speed=0.5):
    dt = 1.0 / fps
    
    if len(positions) < 2:
        return positions[-1]  # No sufficient data, return the last known position

    x_coords, y_coords = zip(*positions)
    frames = np.arange(len(positions))

    # Fit linear regression model
    x_slope, x_intercept = np.polyfit(frames, x_coords, 1)
    y_slope, y_intercept = np.polyfit(frames, y_coords, 1)

    # Predict new position using the regression model
    new_x = x_slope * (len(positions)) + x_intercept
    new_y = y_slope * (len(positions)) + y_intercept

    # Limit the velocity
    dx = new_x - x_coords[-1]
    dy = new_y - y_coords[-1]
    speed = np.sqrt(dx**2 + dy**2) / dt

    max_speed = max_speed_in_ypf(label, fps)
    
    if speed > max_speed:
        scaling_factor = max_speed / speed
        dx *= scaling_factor
        dy *= scaling_factor
        #print("speed was over 10")

    # Compute new predicted position
    new_x = x_coords[-1] + dx
    new_y = y_coords[-1] + dy

    # Check for field boundaries and adjust if necessary
    predicted_x = max(0, min(new_x, 120))   # x boundary: [0, 120] yards
    predicted_y = max(0, min(new_y, 53.3))  # y boundary: [0, 53.3] yards

    return predicted_x, predicted_y

#function use to find closest matched between known locations of unidentified players and unknown locations of identified players
def find_closest_matches(known_df, predicted_df, threshold=1.5):
    known_coords = known_df[['field_x', 'field_y']].values
    predicted_coords = predicted_df[['field_x', 'field_y']].values
    
    tree = cKDTree(known_coords)
    distances, indices = tree.query(predicted_coords, k=1)
    
    # Creating a dictionary to store the best match for each known tracker
    best_matches = {}
    for pred_idx, (distance, known_idx) in enumerate(zip(distances, indices)):
        if distance <= threshold:
            known_id = known_df.iloc[known_idx]['tracker_id']
            pred_id = predicted_df.iloc[pred_idx]['tracker_id']
            
            # If the known player is already matched with another predicted player
            if known_id in best_matches:
                existing_pred_idx, existing_distance = best_matches[known_id]
                # Compare distances and choose the closer one
                if distance < existing_distance:
                    best_matches[known_id] = (pred_idx, distance)
            else:
                best_matches[known_id] = (pred_idx, distance)
                
    matches = [(known_id, predicted_df.iloc[pred_idx]['tracker_id']) for known_id, (pred_idx, _) in best_matches.items()]
    return matches

#interpolate between estimated posistions that are bookend by detections to avoid any jumps 
def interpolate_estimations(data):
    """Interpolate positions for sequences of 'Estimated' bookended by 'Detection' and label them as 'Interpolated'."""
    
    def interpolate_positions(group):
        """Interpolate positions for sequences of 'Estimated' bookended by 'Detection'."""
        # Identify start and end of estimation sequences
        group['start_estimation'] = (group['calc'] == 'Estimated') & (group['calc'].shift(1) == 'Detection')
        group['end_estimation'] = (group['calc'] == 'Estimated') & (group['calc'].shift(-1) == 'Detection')

        # Iterate over rows and interpolate positions for identified sequences
        i = 0
        while i < len(group):
            row = group.iloc[i]
            if row['start_estimation']:
                start_idx = i
                while i < len(group) and not group.iloc[i]['end_estimation']:
                    i += 1
                if i < len(group) and group.iloc[i]['end_estimation']:
                    end_idx = i
                    x_start, y_start = group.iloc[start_idx - 1]['field_x'], group.iloc[start_idx - 1]['field_y']
                    x_end, y_end = group.iloc[end_idx + 1]['field_x'], group.iloc[end_idx + 1]['field_y']
                    delta_x = (x_end - x_start) / (end_idx - start_idx + 2)
                    delta_y = (y_end - y_start) / (end_idx - start_idx + 2)
                    for j in range(start_idx, end_idx + 1):
                        group.at[group.index[j], 'field_x'] = x_start + delta_x * (j - start_idx + 1)
                        group.at[group.index[j], 'field_y'] = y_start + delta_y * (j - start_idx + 1)
                        group.at[group.index[j], 'calc'] = "Interpolated"
            i += 1
        return group

    # Apply the interpolation function to each player's data
    interpolated_and_labeled_data = data.groupby('tracker_id').apply(interpolate_positions)

    # Drop auxiliary columns used for interpolation
    interpolated_and_labeled_data.drop(columns=['start_estimation', 'end_estimation'], inplace=True)

    return interpolated_and_labeled_data

def get_video_fps(video_source):
        # Open the video file
        cap = cv2.VideoCapture()

        # Get FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print(f"FPS of the video: {fps}")
        return fps

def overwrite_estimated_positions(df):
    # Create a copy of the DataFrame to avoid modifying the original data
    corrected_df = df.copy()
    
    # Get unique tracker IDs (players)
    tracker_ids = df['tracker_id'].unique()
    
    for tracker_id in tracker_ids:
        # Filter rows corresponding to the current tracker ID
        player_data = df[df['tracker_id'] == tracker_id]
        
        # Check if the last position of the player is "Estimated"
        if player_data.iloc[-1]['calc'] == 'Estimated':
            # Find the last known position with "Detection"
            last_known_position = player_data[player_data['calc'] == 'Detection'].iloc[-1]
            
            # Identify indices of all "Estimated" positions at the end for the player
            estimated_end_indices = player_data[player_data['calc'] == 'Estimated'].index.tolist()
            
            # Update the "Estimated" positions with the last known position
            for index in estimated_end_indices:
                corrected_df.loc[index, 'field_x'] = last_known_position['field_x']
                corrected_df.loc[index, 'field_y'] = last_known_position['field_y']
                corrected_df.loc[index, 'calc'] = 'Last Known'
    
    return corrected_df

def track_players(fps, player_field_positions, refs_player_associations):

    NUM_POINTS_FOR_REGRESSION = 10
    player_dict = {} # Dictionary to hold player data for each tracker_id
    player_associations = {}

    #filter out all refs (and non-players)
    player_field_positions = player_field_positions[~player_field_positions['tracker_id'].isin(refs_player_associations['tracker_id'].tolist())]
    

    player_field_positions = player_field_positions.sort_values(by=['tracker_id', 'frame']) # Sorting data by tracker_id and frame
    player_field_positions = player_field_positions.groupby('tracker_id').apply(fill_all_gaps).reset_index(drop=True) # Applying the function to each tracker

    frame_results_df = pd.DataFrame(columns=['frame', 'tracker_id', 'field_x', 'field_y', 'label', 'team', 'calc', 'original_tracker_id'])

    # For each player in frame 0 
    for index, row in player_field_positions[player_field_positions['frame'] == 0].iterrows():
        tracker_id = row['tracker_id']
        x = row['field_x']
        y = row['field_y']
        label = row['label']
        team = find_player_team(label)
        player_dict[tracker_id] = init_player_data(x, y, label)
        new_row = pd.DataFrame([{'frame': 0, 'tracker_id': tracker_id, 'field_x': x, 'field_y': y, 'label': label, 'team':team, "calc": "Detection", "original_tracker_id": np.nan}])
        frame_results_df = pd.concat([frame_results_df, new_row], ignore_index=True)


    #compute estimates 
    for frame_num in range(1, player_field_positions['frame'].max() + 1):
        frame_df = player_field_positions[player_field_positions['frame'] == frame_num].copy()
        player_predictions_df = pd.DataFrame(columns=['frame', 'tracker_id', 'field_x', 'field_y', 'label', 'team'])

        for player_id, player_data in player_dict.items():
            if(player_id in player_associations.keys()):
                player_associated_id = player_associations[player_id]
            else:
                player_associated_id = None 
            
            # If original player tracker_id is in the current frame, then use it update the player data
            if player_id in frame_df['tracker_id'].values:
                
                #if player_dict already has 5 or more recent posistions then remove the oldest and add new one 
                if len(player_data['positions']) >= NUM_POINTS_FOR_REGRESSION:
                    player_data['positions'].pop(0)
                
                x = frame_df.loc[frame_df['tracker_id'] == player_id, 'field_x'].values[0]
                y = frame_df.loc[frame_df['tracker_id'] == player_id, 'field_y'].values[0]
                player_data['positions'].append((x, y))

                new_row = pd.DataFrame([{'frame': frame_num, 'tracker_id': player_id, 'field_x': x, 'field_y': y, 'label': player_data['label'], 'team':player_data['team'], "calc": "Detection", "original_tracker_id": player_id}])
                frame_results_df = pd.concat([frame_results_df, new_row], ignore_index=True)

            #if one of the original players isnt in current frame but a known assoicated tracker_id is
            elif (player_associated_id in frame_df['tracker_id'].values):

                if len(player_data['positions']) >= NUM_POINTS_FOR_REGRESSION:
                    player_data['positions'].pop(0)
                
                x = frame_df.loc[frame_df['tracker_id'] == player_associated_id, 'field_x'].values[0]
                y = frame_df.loc[frame_df['tracker_id'] == player_associated_id, 'field_y'].values[0]
                player_data['positions'].append((x, y))

                new_row = pd.DataFrame([{'frame': frame_num, 'tracker_id': player_id, 'field_x': x, 'field_y': y, 'label': player_data['label'], 'team':player_data['team'], "calc": "Detection", "original_tracker_id": player_associated_id}])
                frame_results_df = pd.concat([frame_results_df, new_row], ignore_index=True)
                    
            else: #cases when original player can't be mapped to any tracker_ids in current frame

                predicted_x, predicted_y = extrapolate_position(player_data['positions'], player_data['label'], fps)   # Predict the next position using linear extrapolation
                new_row = pd.DataFrame([{'frame': frame_num, 'tracker_id': player_id, 'field_x': predicted_x, 'field_y': predicted_y, 'label': "", 'team':""}])
                player_predictions_df = pd.concat([player_predictions_df, new_row], ignore_index=True)

        
        #unidenified players from current frame
        known_locations = frame_df[~frame_df.tracker_id.isin(player_dict.keys()|player_associations.values())]  
        
        if(frame_num == 7):
            known_locations.to_csv("known_locations_frame_7.csv")
            player_predictions_df.to_csv("player_predictions_frame_7.csv")

        #map predicted players to unidientifed posistions 
        tracker_id_matches = find_closest_matches(known_locations, player_predictions_df)
        #print(tracker_id_matches)
        
        for match in tracker_id_matches:
            #match[1] is one of OG players
            #match[0] is new players 
            player_associations[match[1]] = match[0] 
            player_dict[match[1]]['associated_tracker_ids'].append(match[0])
            print("Frame: ", frame_num, "  ", match[1], " mapped to " , match[0])

            x = known_locations.loc[known_locations['tracker_id'] == match[0], 'field_x'].values[0]
            y = known_locations.loc[known_locations['tracker_id'] == match[0], 'field_y'].values[0]

            #if player_dict already has 5 or more recent posistions then remove the oldest and add new one 
            if len(player_dict[match[1]]['positions']) >= NUM_POINTS_FOR_REGRESSION:
                player_dict[match[1]]['positions'].pop(0)
            player_dict[match[1]]['positions'].append((x, y))

            
            new_row = pd.DataFrame([{'frame': frame_num, 'tracker_id': match[1], 'field_x': x, 'field_y': y, 'label': player_dict[match[1]]['label'], 'team':player_dict[match[1]]['team'],"calc": "Detection", "original_tracker_id": match[0] }])
            #print(new_row)
            frame_results_df = pd.concat([frame_results_df, new_row])
        
        matched_players = [match[1] for match in tracker_id_matches]
        player_predictions_df = player_predictions_df[~player_predictions_df.tracker_id.isin(matched_players)] 
        
        for player_id, player_data in player_dict.items():
            if(player_id in player_predictions_df['tracker_id'].values):
                
                if len(player_data['positions']) >= NUM_POINTS_FOR_REGRESSION:
                    player_data['positions'].pop(0)
                
                x = player_predictions_df.loc[player_predictions_df['tracker_id'] == player_id, 'field_x'].values[0]
                y = player_predictions_df.loc[player_predictions_df['tracker_id'] == player_id, 'field_y'].values[0]
                player_data['positions'].append((x, y))

                new_row = pd.DataFrame([{'frame': frame_num, 'tracker_id': player_id, 'field_x': x, 'field_y': y, 'label': player_data['label'], 'team':player_data['team'],"calc": "Estimated", "original_tracker_id": np.nan }])
                frame_results_df = pd.concat([frame_results_df, new_row], ignore_index=True)

    #interpolate between estimated posistions that are bookend by detections to avoid any jumps 
    interpolate_estimations_df = interpolate_estimations(frame_results_df)
    interpolate_estimations_df.to_csv('interpolate_estimations_df.csv')
    print("interpolate_estimations_df")
    print(interpolate_estimations_df)

    #for players we lose track of player posistions at the end overwrite the estimates
    overwritten_estimations_df = overwrite_estimated_positions(interpolate_estimations_df)
    overwritten_estimations_df.to_csv('overwritten_estimations_df.csv')
    print("overwritten_estimations_df")
    print(overwritten_estimations_df)

    return overwritten_estimations_df

        


