import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import pandas as pd
import numpy as np
import cv2


from players import detect_players, transform_player_positions, track_players

class AppGUI:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Controls")

        #set varibales 
        self.paused = True
        self.current_frame_number = 0  # Start with frame 0, you can update this number with each frame you process
        self.selected_point = None
        self.selected_player = None
        self.point_edited = False
        self.old_gray = None
        self.video_active = False
        self.lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.tracker_ids = None #used to make sure when adding self.tracker_ids they are unqiue 
        self.frame_player = None
        self.frame = None
        self.p0 = None
        self.cap = None

        #data frames
        self.df_field = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y',  'label'])
        self.df_players_frame_0 = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y', 'label'])
        self.df_players = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y', 'label'])
        self.df_transformed_players = pd.DataFrame(columns=['frame', 'tracker_id', 'field_x', 'field_y', 'label'])
        self.df_tracked_players = None

        # Initialize GUI components
        self._init_gui_components()

    def _init_gui_components(self):

        #Section 1
        top_frame = ttk.Frame(self.root, padding="5")
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        #directory
        self.directory_label = ttk.Label(top_frame, text="Directory")
        self.directory_label.grid(row=0, column=1, pady=5, padx=5, sticky=tk.W)
        self.directory_input = ttk.Entry(top_frame, width=65)
        self.directory_input.grid(row=0, column=2, pady=5, padx=5, sticky=tk.W)

        #video source
        self.video_source_label = ttk.Label(top_frame, text="Video Source")
        self.video_source_label.grid(row=1, column=1, pady=5, padx=5, sticky=tk.W)
        self.video_source_input = ttk.Entry(top_frame, width=65)
        self.video_source_input.grid(row=1, column=2, pady=5, padx=5, sticky=tk.W)

        # SECTION 2
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        #labels
        self.input_label = ttk.Label(frame, text="Input")
        self.input_label.grid(row=0, column=1, pady=5, padx=5, sticky=tk.W)
        self.output_label = ttk.Label(frame, text="Output")
        self.output_label.grid(row=0, column=3, pady=5, padx=5, sticky=tk.E)

    
        #Detect Players 
        self.detect_players_btn = ttk.Button(frame, text="Detect Players", command=self.call_detect_players, width=25)
        self.detect_players_btn.grid(row=1, column=2, pady=5, padx=5, sticky=tk.W)
        
        self.detect_players_output = ttk.Entry(frame, width=40)
        self.detect_players_output.grid(row=1, column=3, pady=5, padx=5, sticky=tk.W)

        #Review Detections 
        self.review_players_btn = ttk.Button(frame, text="Review Detections", command=self.review_detections, width=25)
        self.review_players_btn.grid(row=2, column=2, pady=5, padx=5, sticky=tk.W)
        self.review_players_input = ttk.Entry(frame, width=40)
        self.review_players_input.grid(row=2, column=1, pady=5, padx=5, sticky=tk.W)
        self.review_players_output = ttk.Entry(frame, width=40)
        self.review_players_output.grid(row=2, column=3, pady=5, padx=5, sticky=tk.W)

        #Map Field  
        self.start_mapping_btn = ttk.Button(frame, text="Start Field Mapping", command=self.start_field_mapping, width=25)
        self.start_mapping_btn.grid(row=3, column=2, pady=5, padx=5, sticky=tk.W)
        self.start_mapping_input = ttk.Entry(frame, width=40)
        self.start_mapping_input.grid(row=3, column=1, pady=5, padx=5, sticky=tk.W)
        self.start_mapping_output = ttk.Entry(frame, width=40)
        self.start_mapping_output.grid(row=3, column=3, pady=5, padx=5, sticky=tk.W)

        #Identify Players in Frame 0
        self.start_identify_btn = ttk.Button(frame, text="Start Identify Players", command=self.start_identify_players, width=25)
        self.start_identify_btn.grid(row=4, column=2, pady=5, padx=5, sticky=tk.W)
        self.start_identify_input = ttk.Entry(frame, width=40)
        self.start_identify_input.grid(row=4, column=1, pady=5, padx=5, sticky=tk.W)
        self.start_identify_output = ttk.Entry(frame, width=40)
        self.start_identify_output.grid(row=4, column=3, pady=5, padx=5, sticky=tk.W)
        

        # Transform Players to Posistions on the Field
        self.transform_players_btn = ttk.Button(frame, text="Transform Players", command=self.call_transform_players, width=25)
        self.transform_players_btn.grid(row=5, column=2, pady=5, padx=5, sticky=tk.W)
        self.transform_players_input = ttk.Entry(frame, width=40)
        self.transform_players_input.grid(row=5, column=1, pady=5, padx=5, sticky=tk.W)
        self.transform_players_output = ttk.Entry(frame, width=40)
        self.transform_players_output.grid(row=5, column=3, pady=5, padx=5, sticky=tk.W)

        # Track Players on the Field
        self.track_players_btn = ttk.Button(frame, text="Track Players", command=self.call_track_players, width=25)
        self.track_players_btn.grid(row=6, column=2, pady=5, padx=5, sticky=tk.W)
        self.track_players_input = ttk.Entry(frame, width=40)
        self.track_players_input.grid(row=6, column=1, pady=5, padx=5, sticky=tk.W)
        self.track_players_output = ttk.Entry(frame, width=40)
        self.track_players_output.grid(row=6, column=3, pady=5, padx=5, sticky=tk.W)

        # Create Diagram
        self.create_diagram_btn = ttk.Button(frame, text="Create Diagram", command=self.call_create_diagrams, width=25)
        self.create_diagram_btn.grid(row=7, column=2, pady=5, padx=5, sticky=tk.W)
        self.create_diagram_input = ttk.Entry(frame, width=40)
        self.create_diagram_input.grid(row=7, column=1, pady=5, padx=5, sticky=tk.W)
        self.create_diagram_output = ttk.Entry(frame, width=40)
        self.create_diagram_output.grid(row=7, column=3, pady=5, padx=5, sticky=tk.W)


    def run(self):
        self.root.mainloop()

    def set_field_mapping_output(self, csv_path):
        self.start_mapping_output.insert(0, csv_path)

    def start_field_mapping(self):
        video_source_path = self.video_source_input.get()
        from map_field_gui import MapFieldGUI
        mapField = MapFieldGUI(video_source_path, self.set_field_mapping_output)
        mapField.root.mainloop()

    def set_identify_players_output(self, csv_path):
         self.start_identify__output.insert(0, csv_path)

    def start_identify_players(self):
        video_source_path = self.video_source_input.get()
        players_csv_path = self.start_identify_input.get()
        from identify_players_gui import IdentifyPlayersGUI
        identifyPlayers = IdentifyPlayersGUI(video_source_path,players_csv_path, self.set_identify_players_output)
        identifyPlayers.root.mainloop()

    def call_detect_players(self):
        source_video_path = self.video_source_input.get() 
        self.df_players = detect_players(source_video_path)

        csv_path = source_video_path[:-4] + "_players.csv" #remove .mp4
        self.df_players.to_csv(csv_path)

        self.detect_players_output.insert(0,csv_path)
        self.start_identify_input.insert(0,csv_path)
        
        print("DETECTION COMPLETE!")
        
    def call_transform_players(self):
        self.df_field[['field_x', 'field_y']] = self.df_field['label'].str.split(',', expand=True).astype(float) # Split the 'label' column into 'field_x' and 'field_y' columns
        df_players_without_frame_0 = self.df_players[self.df_players['frame'] != 0] #drop frame 0
        df_players_all_frames = pd.concat([self.df_players_frame_0,df_players_without_frame_0], ignore_index = True) #add in manually annotated frame 0 points 
        self.df_transformed_players = transform_player_positions(df_players_all_frames,self.df_field)
        print("TRANSFORMED!")

    def call_track_players(self):
        self.df_tracked_players = track_players(self.df_transformed_players)
        print(self.df_tracked_players)
        print("PLAYERS TRACKED!")
    
    def review_detections(self):
        print("REVIEW DETECTIONS")

    def call_create_diagrams(self):
        print("CREATE DIAGRAMS!")
    

