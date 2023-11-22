import tkinter as tk
from tkinter import ttk
import pandas as pd

from players import detect_players, transform_player_positions, track_players

class AppGUI:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Controls")

        #set varibales 
        # self.paused = True
        # self.current_frame_number = 0  # Start with frame 0, you can update this number with each frame you process
        # self.selected_point = None
        # self.selected_player = None
        # self.point_edited = False
        # self.old_gray = None
        # self.video_active = False
        # self.lk_params = dict(winSize=(15, 15),
        #                 maxLevel=2,
        #                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # self.tracker_ids = None #used to make sure when adding self.tracker_ids they are unqiue 
        # self.frame_player = None
        # self.frame = None
        # self.p0 = None
        # self.cap = None

        #data frames
        # self.df_field = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y',  'label'])
        # self.df_players_frame_0 = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y', 'label'])
        # self.df_players = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y', 'label'])
        # self.df_transformed_players = pd.DataFrame(columns=['frame', 'tracker_id', 'field_x', 'field_y', 'label'])
        # self.df_tracked_players = None

        # Initialize GUI components
        self._init_gui_components()

    def _init_gui_components(self):

        #Section 1
        frame = ttk.Frame(self.root, padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        #video source
        self.video_source_label = ttk.Label(frame, text="Video Source")
        self.video_source_label.grid(row=1, column=1, pady=5, padx=5, sticky=tk.W)
        self.video_source_input = ttk.Entry(frame, width=65)
        self.video_source_input.grid(row=1, column=2, pady=5, padx=5, sticky=tk.W)

        #player detections
        self.player_detections_label = ttk.Label(frame, text="Player Detections")
        self.player_detections_label.grid(row=2, column=1, pady=5, padx=5, sticky=tk.W)
        self.player_detections_input = ttk.Entry(frame, width=65)
        self.player_detections_input.grid(row=2, column=2, pady=5, padx=5, sticky=tk.W)

        #field points
        self.field_points_label = ttk.Label(frame, text="Field Points")
        self.field_points_label.grid(row=3, column=1, pady=5, padx=5, sticky=tk.W)
        self.field_points_input = ttk.Entry(frame, width=65)
        self.field_points_input.grid(row=3, column=2, pady=5, padx=5, sticky=tk.W)

        #frame 0 players
        self.frame_0_players_label = ttk.Label(frame, text="Fame 0 Players")
        self.frame_0_players_label.grid(row=4, column=1, pady=5, padx=5, sticky=tk.W)
        self.frame_0_players_input = ttk.Entry(frame, width=65)
        self.frame_0_players_input.grid(row=4, column=2, pady=5, padx=5, sticky=tk.W)

        #transform players
        self.transformed_players_label = ttk.Label(frame, text="Transformed Players")
        self.transformed_players_label.grid(row=5, column=1, pady=5, padx=5, sticky=tk.W)
        self.transformed_players_input = ttk.Entry(frame, width=65)
        self.transformed_players_input.grid(row=5, column=2, pady=5, padx=5, sticky=tk.W)

        #player associations
        self.player_associations_label = ttk.Label(frame, text="Player Associations")
        self.player_associations_label.grid(row=6, column=1, pady=5, padx=5, sticky=tk.W)
        self.player_associations_input = ttk.Entry(frame, width=65)
        self.player_associations_input.grid(row=6, column=2, pady=5, padx=5, sticky=tk.W)

        #tracked players
        self.tracked_players_label = ttk.Label(frame, text="Tracked Players")
        self.tracked_players_label.grid(row=7, column=1, pady=5, padx=5, sticky=tk.W)
        self.tracked_players_input = ttk.Entry(frame, width=65)
        self.tracked_players_input.grid(row=7, column=2, pady=5, padx=5, sticky=tk.W)

        #diagram
        self.diagram_label = ttk.Label(frame, text="Diagram")
        self.diagram_label.grid(row=8, column=1, pady=5, padx=5, sticky=tk.W)
        self.diagram_input = ttk.Entry(frame, width=65)
        self.diagram_input.grid(row=8, column=2, pady=5, padx=5, sticky=tk.W)

        # Draw a solid black line between column 2 and 3
        line_canvas = tk.Canvas(frame, width=2, height=frame.winfo_reqheight(), bg='black')  # Set the background color
        line_canvas.grid(row=1, column=2, rowspan=7, sticky=tk.E+tk.N+tk.S, padx=(10, 5))  # Align it to the right with sticky=tk.E
        # Since the canvas itself is black, you don't even need to draw a line on it, but if you want to:
        line_canvas.create_line(1, 0, 1, line_canvas.winfo_reqheight(), fill='black')  # Draws a black vertical line


        #Detect Players 
        self.detect_players_btn = ttk.Button(frame, text="Detect Players", command=self.call_detect_players, width=25)
        self.detect_players_btn.grid(row=1, column=3, pady=5, padx=5, sticky=tk.W)

        #Review Detections 
        self.review_players_btn = ttk.Button(frame, text="Review Detections", command=self.review_detections, width=25)
        self.review_players_btn.grid(row=2, column=3, pady=5, padx=5, sticky=tk.W)

        #Map Field  
        self.start_mapping_btn = ttk.Button(frame, text="Map Field", command=self.start_field_mapping, width=25)
        self.start_mapping_btn.grid(row=3, column=3, pady=5, padx=5, sticky=tk.W)

        #Identify Players in Frame 0
        self.start_identify_btn = ttk.Button(frame, text="Identify Players", command=self.start_identify_players, width=25)
        self.start_identify_btn.grid(row=4, column=3, pady=5, padx=5, sticky=tk.W)

        # Transform Players to Posistions on the Field
        self.transform_players_btn = ttk.Button(frame, text="Transform Players", command=self.call_transform_players, width=25)
        self.transform_players_btn.grid(row=5, column=3, pady=5, padx=5, sticky=tk.W)
        
    
        # Track Players on the Field
        self.track_players_btn = ttk.Button(frame, text="Track Players", command=self.call_track_players, width=25)
        self.track_players_btn.grid(row=6, column=3, pady=5, padx=5, sticky=tk.W)
        
        # Create Diagram
        self.create_diagram_btn = ttk.Button(frame, text="Create Diagram", command=self.call_create_diagrams, width=25)
        self.create_diagram_btn.grid(row=7, column=3, pady=5, padx=5, sticky=tk.W)


    def run(self):
        self.root.mainloop()

    def set_field_mapping_output(self, csv_path):
        self.field_points_input.insert(0, csv_path)

    def start_field_mapping(self):
        video_source_path = self.video_source_input.get()
        from map_field_gui import MapFieldGUI
        mapField = MapFieldGUI(video_source_path, self.set_field_mapping_output)
        mapField.root.mainloop()

    def set_identify_players_output(self, csv_path):
         self.frame_0_players_input.insert(0, csv_path)

    def start_identify_players(self):
        video_source_path = self.video_source_input.get()
        players_csv_path = self.player_detections_input.get()
        from identify_players_gui import IdentifyPlayersGUI
        identifyPlayers = IdentifyPlayersGUI(video_source_path,players_csv_path, self.set_identify_players_output)
        identifyPlayers.root.mainloop()

    def call_detect_players(self):
        source_video_path = self.video_source_input.get() 
        df_players = detect_players(source_video_path)

        csv_path = source_video_path[:-4] + "_players.csv" #remove .mp4
        df_players.to_csv(csv_path)

        self.player_detections_input.insert(0,csv_path)
        
        print("DETECTION COMPLETE!")
        
    def call_transform_players(self):
        df_field = pd.read_csv(self.field_points_input.get())
        df_field[['field_x', 'field_y']] = df_field['label'].str.split(',', expand=True).astype(float) # Split the 'label' column into 'field_x' and 'field_y' columns
        
        df_players = pd.read_csv(self.player_detections_input.get())
        df_players_without_frame_0 = df_players[df_players['frame'] != 0] #drop frame 0

        df_players_frame_0 = pd.read_csv(self.frame_0_players_input.get())

        df_players_all_frames = pd.concat([df_players_frame_0,df_players_without_frame_0], ignore_index = True) #add in manually annotated frame 0 points 
        
        df_transformed_players = transform_player_positions(df_players_all_frames,df_field)

        csv_path = self.video_source_input.get()[:-4] + "_transformed_players.csv" #remove .mp4
        df_transformed_players.to_csv(csv_path)

        self.transformed_players_input.insert(0,csv_path)

        print("TRANSFORMED!")

    
    def set_track_players_output(self, csv_path_player_associations, csv_path_tracked_players):
        self.player_associations_input.insert(0,csv_path_player_associations)
        self.tracked_players_input.insert(0,csv_path_tracked_players)

    def call_track_players(self):
        
        
        from track_players_gui import TrackPlayersGUI
        trackPlayers = TrackPlayersGUI(
            self.video_source_input.get(),
            self.transformed_players_input.get(), 
            self.set_track_players_output)
        trackPlayers.root.mainloop()


        #self.df_tracked_players = track_players(self.df_transformed_players)
        #print(self.df_tracked_players)
        print("PLAYERS TRACKED!")
    
    def review_detections(self):
        print("REVIEW DETECTIONS")

    def call_create_diagrams(self):
        source_video_path = self.video_source_input.get()
        estimated_players = pd.read_csv(self.tracked_players_input.get())

        from diagrams import create_football_diagram_video
        diagram_path = create_football_diagram_video(estimated_players,source_video_path)
        self.diagram_input.insert(0,diagram_path)
        print("CREATE DIAGRAMS!")
    

