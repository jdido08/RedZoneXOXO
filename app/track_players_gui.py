import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import pandas as pd
import numpy as np
import cv2


class TrackPlayersGUI:

    def __init__(self, source_video_path, transformed_players_csv_path, callback):

        self.root = tk.Tk()
        self.root.title("Controls")

        #set varibales 
        self.transformed_players_csv_path = transformed_players_csv_path
        self.selected_player_association = None 
        self.source_video_path = source_video_path
        
        #set data frames
        self.df_player_associations = pd.DataFrame(columns=['player_tracker_id', 'type', 'tracker_id', 'offset'])
        self.df_tracked_players = None

        # Initialize GUI components
        self._init_gui_components()

        #set callback
        self.callback = callback

    def _init_gui_components(self):
        
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        #Track Players Round 1
        track_players_btn = ttk.Button(frame, text="Track Players Round 1", command=self.track_players_round_1)
        track_players_btn.grid(row=0, column=1, pady=5, sticky=tk.W+tk.E)

        track_players_btn_2 = ttk.Button(frame, text="Track Players Round 2", command=self.track_players_round_2)
        track_players_btn_2.grid(row=0, column=3, pady=5, sticky=tk.W+tk.E)

        #Add Player Association
        add_association_btn = ttk.Button(frame, text="Add Player Association", command=self.add_player_association)
        add_association_btn.grid(row=1, column=3, pady=5, sticky=tk.W+tk.E)

        #Add Player Shadow Association
        add_shadow_association_btn = ttk.Button(frame, text="Add Shadow Player Association", command=self.add_shadow_association)
        add_shadow_association_btn.grid(row=3, column=3, pady=5, sticky=tk.W+tk.E)

        #Add Ref Association
        add_ref_association_btn = ttk.Button(frame, text="Add Ref Association", command=self.add_ref_association)
        add_ref_association_btn.grid(row=1, column=1, pady=5, sticky=tk.W+tk.E)

        #add tree
        self.tree_frame = ttk.Frame(frame)
        self.tree_frame.grid(row=4, column=1, padx=10, sticky=tk.W+tk.E+tk.N+tk.S)

        self.tree = ttk.Treeview(self.tree_frame)
        self.tree["columns"] = ("player_tracker_id", "type", "tracker_id", "offset")
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("player_tracker_id", anchor=tk.W, width=50)
        self.tree.column("type", anchor=tk.W, width=50)
        self.tree.column("tracker_id", anchor=tk.W, width=50)
        self.tree.column("offset", anchor=tk.W, width=50)

        self.tree.heading("#0", text="", anchor=tk.W)
        self.tree.heading("player_tracker_id", text="Player Tracker ID", anchor=tk.W)
        self.tree.heading("type", text="Type", anchor=tk.W)
        self.tree.heading("tracker_id", text="Tracker ID", anchor=tk.W)
        self.tree.heading("offset", text="Offset", anchor=tk.W)
        self.tree.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)

        self.player_tree_scroll = tk.Scrollbar(self.tree_frame)
        self.player_tree_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.player_tree_scroll.config(command=self.tree.yview)
        self.tree.config(yscrollcommand=self.player_tree_scroll.set)

        self.tree.bind('<ButtonRelease-1>', self.on_tree_select)

        self.delete_association_btn = ttk.Button(frame, text="Delete Association", command=self.delete_association)
        self.delete_association_btn.grid(row=5, column=1, pady=5, sticky=tk.W+tk.E)


        #add components for last unknown tree
        self.lu_tree_frame = ttk.Frame(frame)
        self.lu_tree_frame.grid(row=4, column=3, padx=10, sticky=tk.W+tk.E+tk.N+tk.S)

        self.lu_tree = ttk.Treeview(self.lu_tree_frame)
        
        self.lu_tree_scroll = tk.Scrollbar(self.lu_tree_frame)
        self.lu_tree_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.lu_tree_scroll.config(command=self.lu_tree.yview)
        self.lu_tree.config(yscrollcommand=self.lu_tree_scroll.set)

    def run(self):
        self.root.mainloop()


    ###### Track Players Functions ########
    def populate_last_unknown_tree(self, df):
        self.lu_tree["columns"] = df.columns.tolist()
        for col in df.columns:
            self.lu_tree.column(col, anchor=tk.W)
            self.lu_tree.heading(col, text=col, anchor=tk.W)
        for index, row in df.iterrows():
            self.lu_tree.insert("", "end", values=row.tolist())

    def get_video_fps(self, source_path):
        # Open the video file
        cap = cv2.VideoCapture(source_path)

        # Get FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print(f"FPS of the video: {fps}")
        return fps
        
    def calc_num_of_last_known_positions(self,df):
        
        #print(df)

        df = df.drop(columns=['tracker_id']).reset_index()

        # Step 2: Calculate the number of "Last Known" positions for each tracker_id
        last_known_counts = df[df['calc'] == 'Last Known'].groupby('tracker_id').size()
        
        # Step 3: Create a new DataFrame
        result_df = last_known_counts.reset_index(name='count_last_known')
        
        # Step 4: Sort the DataFrame based on the count, in descending order
        result_df = result_df.sort_values(by='count_last_known', ascending=False)

        print("result_df")
        print(result_df)
        
        # Step 5: Return the sorted DataFrame
        return result_df


    def track_players_round_1(self):
        #get transformed player posistions 
        df_transformed_players = pd.read_csv(self.transformed_players_csv_path)

        #get refs from df_player_associations
        refs_player_associations = self.df_player_associations[self.df_player_associations['type']=='ref_association']
        
        #csv_path_player_associations = self.source_video_path[:-4] + "_player_associations.csv" #remove .mp4
        #self.df_player_associations.to_csv(csv_path_player_associations)

        fps = self.get_video_fps(self.source_video_path)
        
        from players import track_players
        self.df_tracked_players = track_players(fps, df_transformed_players, refs_player_associations)
        tracker_ids_with_last_known_posistions = self.calc_num_of_last_known_positions(self.df_tracked_players)
        self.populate_last_unknown_tree(tracker_ids_with_last_known_posistions)

        # csv_path_tracked_players = self.source_video_path[:-4] + "_tracked_players.csv" #remove .mp4
        # df_track_players.to_csv(csv_path_tracked_players)
        # self.callback(csv_path_player_associations,csv_path_tracked_players)
        # self.root.destroy()
        
    def track_players_round_2(self):
        print("TRACK PLAYERS ROUND 2")

    def update_treeview(self):

        # Clear all current items in the self.tree
        for row in self.tree.get_children():
            self.tree.delete(row)

        for index, row in self.df_player_associations.iterrows():
            self.tree.insert(parent='', index='end', iid=index, text="", 
                        values=(row['player_tracker_id'], row['type'], row['tracker_id'], row['offset']))

    def on_tree_select(self, event):
        item = self.tree.selection()[0]
        self.selected_player_association = self.tree.item(item, "values")

    def delete_association(self):
        if self.selected_player_association:
            player_tracker_id, type, tracker_id, offset= self.selected_player_association
            self.df_player_associations = self.df_player_associations.drop(self.df_player_associations[(self.df_player_associations['player_tracker_id'] == player_tracker_id) & 
                                                (self.df_player_associations['type'] == type) & 
                                                (self.df_player_associations['tracker_id'] == tracker_id) &
                                                (self.df_player_associations['offset'] == offset)].index)
            self.update_treeview()
            self.selected_player_association = None  # Reset selected point

    def add_player_association(self):
        # Create a new window
        self.add_association_window = tk.Toplevel(self.root)
        self.add_association_window.title("Add Player Association")

        # Main Player ID
        main_player_id_label = ttk.Label(self.add_association_window, text="Main Player ID:")
        main_player_id_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.main_player_id_entry = ttk.Entry(self.add_association_window, width=30)
        self.main_player_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Associated Tracker ID
        associated_tracker_id_label = ttk.Label(self.add_association_window, text="Associated Tracker ID:")
        associated_tracker_id_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.associated_tracker_id_entry = ttk.Entry(self.add_association_window, width=30)
        self.associated_tracker_id_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Submit button
        submit_btn = ttk.Button(self.add_association_window, text="Submit", command=self.submit_player_association)
        submit_btn.grid(row=2, column=0, pady=10)

    def submit_player_association(self):
        main_player_id = self.main_player_id_entry.get()
        associated_tracker_id = self.associated_tracker_id_entry.get()

        # Add the data to the dataframe
        new_row = pd.DataFrame([{'player_tracker_id': main_player_id, 'type': "association", 'tracker_id': associated_tracker_id, 'offset': None}])
        self.df_player_associations = pd.concat([self.df_player_associations, new_row], ignore_index=True)

        self.update_treeview()  # Refresh the treeview to reflect the changes

        # Close the association window
        self.add_association_window.destroy()

    def add_shadow_association(self):
        # Create a new window
        self.add_shadow_window = tk.Toplevel(self.root)
        self.add_shadow_window.title("Add Shadow Association")

        # Main Player ID
        main_player_id_label = ttk.Label(self.add_shadow_window, text="Main Player ID:")
        main_player_id_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.main_player_id_entry_shadow = ttk.Entry(self.add_shadow_window, width=30)
        self.main_player_id_entry_shadow.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Associated Tracker ID
        associated_tracker_id_label = ttk.Label(self.add_shadow_window, text="Associated Tracker ID:")
        associated_tracker_id_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.associated_tracker_id_entry_shadow = ttk.Entry(self.add_shadow_window, width=30)
        self.associated_tracker_id_entry_shadow.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Offset
        offset_label = ttk.Label(self.add_shadow_window, text="Offset:")
        offset_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.offset_entry_shadow = ttk.Entry(self.add_shadow_window, width=30)
        self.offset_entry_shadow.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        # Submit button
        submit_btn = ttk.Button(self.add_shadow_window, text="Submit", command=self.submit_shadow_association)
        submit_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def submit_shadow_association(self):
        main_player_id = self.main_player_id_entry_shadow.get()
        associated_tracker_id = self.associated_tracker_id_entry_shadow.get()
        offset = self.offset_entry_shadow.get()

        # Add the data to the dataframe
        new_row = pd.DataFrame([{'player_tracker_id': main_player_id, 'type': "shadow_association", 'tracker_id': associated_tracker_id, 'offset': offset}])
        self.df_player_associations = pd.concat([self.df_player_associations, new_row], ignore_index=True)

        self.update_treeview()  # Refresh the treeview to reflect the changes

        # Close the shadow association window
        self.add_shadow_window.destroy()

    def add_ref_association(self):
        # Create a new window
        self.add_ref_window = tk.Toplevel(self.root)
        self.add_ref_window.title("Add Ref Association")

        # Associated Tracker ID
        tracker_id_label = ttk.Label(self.add_ref_window, text="Tracker ID:")
        tracker_id_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.tracker_id_entry_ref = ttk.Entry(self.add_ref_window, width=30)
        self.tracker_id_entry_ref.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Submit button
        submit_btn = ttk.Button(self.add_ref_window, text="Submit", command=self.submit_ref_association)
        submit_btn.grid(row=1, column=0, columnspan=2, pady=10)

    def submit_ref_association(self):
        tracker_id = self.tracker_id_entry_ref.get()

        # Add the data to the dataframe
        new_row = pd.DataFrame([{'player_tracker_id': None, 'type': "ref_association", 'tracker_id': tracker_id, 'offset': None}])
        self.df_player_associations = pd.concat([self.df_player_associations, new_row], ignore_index=True)

        self.update_treeview()  # Refresh the treeview to reflect the changes

        # Close the ref association window
        self.add_ref_window.destroy()
