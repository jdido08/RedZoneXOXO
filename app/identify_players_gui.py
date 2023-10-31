import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import pandas as pd
import numpy as np
import cv2


class IdentifyPlayersGUI:

    def __init__(self, source_video_path, players_csv_path, callback):

        self.root = tk.Tk()
        self.root.title("Controls")

        #set varibales 
        self.players_csv_path = players_csv_path
        self.tracker_ids = None #used to make sure when adding self.tracker_ids they are unqiue 
        self.frame_player = None
        self.source_video_path = source_video_path
        self.cap = cv2.VideoCapture(source_video_path)

        #set data frames
        self.df_players_frame_0 = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y', 'label'])

        # Initialize GUI components
        self._init_gui_components()

        #set callback
        self.callback = callback

    def _init_gui_components(self):
        
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Player Positions Treeview
        start_identify_btn = ttk.Button(frame, text="Start Identify Players", command=self.start_identify_players)
        start_identify_btn.grid(row=0, column=1, pady=5, sticky=tk.W+tk.E)

        end_identify_btn = ttk.Button(frame, text="End Identify Players", command=self.end_identify_players)
        end_identify_btn.grid(row=1, column=1, pady=5, sticky=tk.W+tk.E)


        self.player_tree_frame = ttk.Frame(frame)
        self.player_tree_frame.grid(row=3, column=1, padx=10, sticky=tk.W+tk.E+tk.N+tk.S)

        self.player_tree = ttk.Treeview(self.player_tree_frame)
        self.player_tree["columns"] = ("Frame", "ID", "X", "Y", "Label")
        self.player_tree.column("#0", width=0, stretch=tk.NO)
        self.player_tree.column("Frame", anchor=tk.W, width=50)
        self.player_tree.column("ID", anchor=tk.W, width=50)
        self.player_tree.column("X", anchor=tk.W, width=50)
        self.player_tree.column("Y", anchor=tk.W, width=50)
        self.player_tree.column("Label", anchor=tk.W, width=100)

        self.player_tree.heading("#0", text="", anchor=tk.W)
        self.player_tree.heading("Frame", text="Frame", anchor=tk.W)
        self.player_tree.heading("ID", text="ID", anchor=tk.W)
        self.player_tree.heading("X", text="X", anchor=tk.W)
        self.player_tree.heading("Y", text="Y", anchor=tk.W)
        self.player_tree.heading("Label", text="Label", anchor=tk.W)
        self.player_tree.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)

        self.player_tree_scroll = tk.Scrollbar(self.player_tree_frame)
        self.player_tree_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.player_tree_scroll.config(command=self.player_tree.yview)
        self.player_tree.config(yscrollcommand=self.player_tree_scroll.set)

        self.player_tree.bind('<ButtonRelease-1>', self.on_player_tree_select)

        self.delete_player_btn = ttk.Button(frame, text="Delete Player", command=self.delete_selected_player)
        self.delete_player_btn.grid(row=5, column=1, pady=5, sticky=tk.W+tk.E)

        self.delete_player_btn = ttk.Button(frame, text="Edit Player Label", command=self.edit_selected_player_label)
        self.delete_player_btn.grid(row=6, column=1, pady=5, sticky=tk.W+tk.E)

    def run(self):
        self.root.mainloop()


    ###### Identify Players Functions ########
        
    def start_identify_players(self):
        df_players = pd.read_csv(self.players_csv_path)
        self.tracker_ids = np.sort(df_players['tracker_id'].unique())
        self.df_players_frame_0 = df_players[df_players['frame'] == 0]
        self.update_player_treeview()  # Populate the player treeview

        cv2.namedWindow('Identify Players', cv2.WINDOW_NORMAL)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, self.frame_player = self.cap.read()
        self.draw_frame_players(self.frame_player)  # Draw players from self.df_players_frame_0
        
        cv2.setMouseCallback('Identify Players', self.on_mouse_identify)

    def update_player_treeview(self):

        # Clear all current items in the self.tree
        for row in self.player_tree.get_children():
            self.player_tree.delete(row)

        for index, row in self.df_players_frame_0.iterrows():
            self.player_tree.insert(parent='', index='end', iid=index, text="", 
                        values=(row['frame'], row['tracker_id'], row['x'], row['y'], row['label']))

    def on_player_tree_select(self, event):
        item = self.player_tree.selection()[0]
        self.selected_player = self.player_tree.item(item, "values")

        frame, tracker_id, old_x, old_y, label = self.selected_player

        frame_copy = self.frame_player.copy()
        cv2.circle(frame_copy, (int(float(old_x)), int(float(old_y))), 5, (0, 0, 255), -1)
        cv2.imshow('Identify Players', frame_copy)
        self.edit_selected_player_label()  # Note the use of 'self' here
        print(f"Selected point: {self.selected_player}")

    def end_identify_players(self):
        cv2.destroyWindow('Identify Players')

        csv_path = self.source_video_path[:-4] + "_frame_0_players.csv" #remove .mp4
        self.df_players_frame_0.to_csv(csv_path)
        self.callback(csv_path)

        self.root.destroy()

    def delete_selected_player(self):
        if self.selected_player:
            frame, tracker_id, old_x, old_y, label = self.selected_player
            self.df_players_frame_0 = self.df_players_frame_0.drop(self.df_players_frame_0[(self.df_players_frame_0['frame'] == int(0)) & 
                                                (self.df_players_frame_0['x'] == float(old_x)) & 
                                                (self.df_players_frame_0['y'] == float(old_y))].index)
            self.update_player_treeview()
            self.draw_frame_players(self.frame_player)  # Redraw the frame after deletion
            self.selected_player = None  # Reset selected point

    def edit_selected_player_label(self):
        frame, tracker_id, old_x, old_y, label = self.selected_player
        
        label = simpledialog.askstring("Input", f"Enter position for player:") 
                
        if label is None:  # If the user cancels the dialog
            print("point not added!")
            self.selected_player = False
            return 
        
        self.df_players_frame_0 = self.df_players_frame_0.drop(self.df_players_frame_0[(self.df_players_frame_0['frame'] == int(0)) & 
                                                (self.df_players_frame_0['x'] == float(old_x)) & 
                                                (self.df_players_frame_0['y'] == float(old_y))].index)
        
        # Add the new point with the same label
        new_row = pd.DataFrame([{'frame': 0, 'tracker_id':tracker_id, 'x': old_x, 'y': old_y, 'label': label}])
        self.df_players_frame_0 = pd.concat([self.df_players_frame_0, new_row], ignore_index=True)
        self.update_player_treeview()
        self.draw_frame_players(self.frame_player)  # Redraw the frame 
        self.selected_player = None 

    def on_mouse_identify(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selected_player:
                frame, tracker_id, old_x, old_y, label = self.selected_player
                self.df_players_frame_0 = self.df_players_frame_0.drop(self.df_players_frame_0[(self.df_players_frame_0['frame'] == int(0)) & 
                                                    (self.df_players_frame_0['x'] == float(old_x)) & 
                                                    (self.df_players_frame_0['y'] == float(old_y))].index)
                
                # Add the new point with the same label
                new_row = pd.DataFrame([{'frame': 0, 'tracker_id':tracker_id, 'x': x, 'y': y, 'label': label}])
                self.df_players_frame_0 = pd.concat([self.df_players_frame_0, new_row], ignore_index=True)
                self.update_player_treeview()
                self.draw_frame_players(self.frame_player)  # Redraw the frame  
            
                self.selected_player = None  # Reset selected point

            else:
                # Keep a copy of the current frame before drawing
                frame_copy = self.frame_player.copy()

                ############### FOR TESTING ##############################################
                pixel = frame_copy[y, x]  # Note: OpenCV represents images in BGR format
                # Convert from BGR to RGB
                pixel_rgb = (pixel[2], pixel[1], pixel[0])
                print(f"RGB Value at ({x}, {y}): {pixel_rgb}")
                ########################################################################

                # Mark the point and add label
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Identify Players', frame_copy)

                # Prompt for label
                label = simpledialog.askstring("Input", f"Enter position for player:")
                
                if label is None:  # If the user cancels the dialog
                    print("point not added!")
                    cv2.imshow('Identify Players', self.frame_player)
                    return 
                        
                
                tracker_id = self.tracker_ids[-1] + 1
                self.tracker_ids = np.append(self.tracker_ids, tracker_id)

                new_row = pd.DataFrame([{'frame': 0, 'tracker_id':tracker_id, 'x': x, 'y': y, 'label': label}])
                self.df_players_frame_0 = pd.concat([self.df_players_frame_0, new_row], ignore_index=True)
                self.update_player_treeview()
                self.draw_frame_players(self.frame_player)  # Redraw the frame after deletion}

    def draw_frame_players(self, frame):
        
        #frame_copy = frame_player.copy()
        #frame_copy = frame_player
        for _, row in self.df_players_frame_0.iterrows():
            x, y, label = round(float(row['x'])), round(float(row['y'])), row['label']
            cv2.circle(frame, (x, y), 5, (128, 0, 128), -1)
            if str(label) != "nan":
                cv2.putText(frame, str(label), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Identify Players', frame)

