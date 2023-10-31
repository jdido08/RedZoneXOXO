import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
import pandas as pd
import numpy as np
import cv2



class MapFieldGUI:

    def __init__(self, video_source_path, callback):

        self.root = tk.Tk()
        self.root.title("Map Field")

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
        self.source_video_path = video_source_path
        self.cap = cv2.VideoCapture(video_source_path)
        print(self.source_video_path)

        #data frames
        self.df_field = pd.DataFrame(columns=['frame', 'tracker_id', 'x', 'y',  'label'])

        #set callback function
        self.callback = callback
        
        # Initialize GUI components
        self._init_gui_components()

    def _init_gui_components(self):
        
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Buttons at the top
        self.start_mapping_btn = ttk.Button(frame, text="Start Field Mapping", command=self.start_field_mapping)
        self.start_mapping_btn.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)

        self.end_mapping_btn = ttk.Button(frame, text="End Field Mapping", command=self.end_field_mapping)
        self.end_mapping_btn.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)

        self.play_btn = ttk.Button(frame, text="Play/Pause", command=self.play_pause_video)
        self.play_btn.grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)

        # Field Points Treeview
        self.tree_frame = ttk.Frame(frame)
        self.tree_frame.grid(row=3, column=0, padx=10, sticky=tk.W+tk.E+tk.N+tk.S)

        self.tree = ttk.Treeview(self.tree_frame)
        self.tree["columns"] = ("Frame", "Type", "X", "Y", "Label")
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Frame", anchor=tk.W, width=50)
        self.tree.column("Type", anchor=tk.W, width=50)
        self.tree.column("X", anchor=tk.W, width=50)
        self.tree.column("Y", anchor=tk.W, width=50)
        self.tree.column("Label", anchor=tk.W, width=100)

        self.tree.heading("#0", text="", anchor=tk.W)
        self.tree.heading("Frame", text="Frame", anchor=tk.W)
        self.tree.heading("Type", text="Type", anchor=tk.W)
        self.tree.heading("X", text="X", anchor=tk.W)
        self.tree.heading("Y", text="Y", anchor=tk.W)
        self.tree.heading("Label", text="Label", anchor=tk.W)

        self.tree.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)

        self.tree_scroll = tk.Scrollbar(self.tree_frame, command=self.tree.yview)
        self.tree.config(yscrollcommand=self.tree_scroll.set)

        self.tree.bind('<ButtonRelease-1>', self.on_tree_select)

        self.delete_btn = ttk.Button(frame, text="Delete Point", command=self.delete_selected_point)
        self.delete_btn.grid(row=4, column=0, pady=5, sticky=tk.W+tk.E)

        # Slider below the treeviews
        self.max_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=10)

       # Configure the root to expand the column containing your frames
        self.root.grid_columnconfigure(0, weight=1)

        # Configure the slider frame to expand the column containing the frame_slider
        self.slider_frame.grid_columnconfigure(1, weight=1)

        self.frame_slider = tk.Scale(self.slider_frame, from_=0, to=self.max_frames-1, orient=tk.HORIZONTAL, label="Frame")
        self.frame_slider.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)

        self.increment_btn = ttk.Button(self.slider_frame, text=">", command=self.increment_frame)
        self.decrement_btn = ttk.Button(self.slider_frame, text="<", command=self.decrement_frame)
        self.decrement_btn.grid(row=1, column=0)  
        self.increment_btn.grid(row=1, column=2)  

    def run(self):
        self.root.mainloop()


    ###### Field Mapping Functions ########


    def draw_frame_points(self):
        frame_copy = self.frame.copy()
        current_frame_points = self.df_field[self.df_field['frame'] == self.current_frame_number]
        for _, row in current_frame_points.iterrows():
            x, y, label = int(row['x']), int(row['y']), row['label']
            cv2.circle(frame_copy, (x, y), 5, (128, 0, 128), -1)
            cv2.putText(frame_copy, label, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Football Video', frame_copy)

    def on_tree_select(self,event):
        item = self.tree.selection()[0]
        self.selected_point = self.tree.item(item, "values")

        frame_num, type, old_x, old_y, label = self.selected_point

        cv2.circle(self.frame, (int(float(old_x)), int(float(old_y))), 5, (0, 0, 255), -1)
        cv2.imshow('Football Video', self.frame)
        
        print(f"Selected point: {self.selected_point}")

    def update_treeview(self):
        # Clear all current items in the self.tree
        for row in self.tree.get_children():
            self.tree.delete(row)

        frame_points = self.df_field[self.df_field['frame'] == self.current_frame_number]
        for index, row in frame_points.iterrows():
            self.tree.insert(parent='', index='end', iid=index, text="", 
                        values=(row['frame'], row['type'], row['x'], row['y'], row['label']))
            
    def delete_selected_point(self):
        if self.selected_point:
            frame_num, type, old_x, old_y, label = self.selected_point
            self.df_field = self.df_field.drop(self.df_field[(self.df_field['frame'] == int(frame_num)) & 
                                                (self.df_field['type'] == type) &
                                                (self.df_field['x'] == float(old_x)) & 
                                                (self.df_field['y'] == float(old_y))].index)
            self.update_treeview()
            self.draw_frame_points()  # Redraw the frame after deletion
            self.selected_point = None  # Reset selected point
            self.point_edited = True
        
    def on_mouse(self, event, x, y, flags, param): 
        
        if event == cv2.EVENT_LBUTTONDOWN:

            if self.selected_point:
                frame_num, type, old_x, old_y, label = self.selected_point

                # Mark the new point and add label
                # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                # cv2.putText(frame, label, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.imshow('Football Video', frame)

                # Remove the original point from the dataframe
                self.df_field = self.df_field.drop(self.df_field[(self.df_field['frame'] == int(self.current_frame_number)) & 
                                                        (self.df_field['type'] == type) &
                                                        (self.df_field['x'] == float(old_x)) & 
                                                        (self.df_field['y'] == float(old_y))].index)
                print("removed")
                self.update_treeview()
                self.draw_frame_points()

                # Add the new point with the same label
                new_row = pd.DataFrame([{'frame': int(self.current_frame_number), 'type':'field', 'x': x, 'y': y, 'label': label}])
                self.df_field = pd.concat([self.df_field, new_row], ignore_index=True)
                self.draw_frame_points()  # Redraw the frame after editing
                self.update_treeview()

                self.selected_point = None  # Reset selected point
                self.point_edited = True


            else:

                # Keep a copy of the current frame before drawing
                frame_copy = self.frame.copy()

                # Mark the point and add label
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Football Video', self.frame)

                # Prompt for label
                label = simpledialog.askstring("Input", f"Enter label for point ({x},{y}):")
                
                if label is None:  # If the user cancels the dialog
                    print("point not added!")
                    self.frame = frame_copy  # Revert to the unmodified frame
                    cv2.imshow('Football Video', self.frame)
                    return 
                
                cv2.putText(self.frame, label, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('Football Video', self.frame)
                
                # update dataframe with new point 
                new_row = pd.DataFrame([{'frame': self.current_frame_number, 'type':'field', 'x': x, 'y': y, 'label': label}])
                self.df_field = pd.concat([self.df_field, new_row], ignore_index=True)
                self.update_treeview()
                
    def play_pause_video(self):
        if self.paused == True:
            self.paused = False
        elif self.paused == False:
            self.paused = True
        
    def start_field_mapping(self):
        self.video_active = True
        cv2.namedWindow('Football Video', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Football Video', self.on_mouse)
        self.video_loop()  # Start the video loop

    def end_field_mapping(self):
        self.video_active = False
        self.cap.release()
        cv2.destroyAllWindows()
        
        csv_path = self.source_video_path[:-4] + "_field_points.csv" #remove .mp4
        self.df_field.to_csv(csv_path)
        self.callback(csv_path)

        self.root.destroy()

    def increment_frame(self):
        current_value = self.frame_slider.get()
        if current_value < (int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1):
            self.frame_slider.set(current_value + 1)

    def decrement_frame(self):
        current_value = self.frame_slider.get()
        if current_value > 0:
            self.frame_slider.set(current_value - 1)

    # Video playback loop function for Tkinter's main loop
    def video_loop(self):

        if not self.video_active:
            return #retun immediately if video is not active

        # This part will ensure the first frame is displayed when the video is paused
        if self.old_gray is None:
            ret, self.frame = self.cap.read()
            if not ret:
                self.root.quit()
                return
            self.old_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Football Video', self.frame)
        
        elif not self.paused:  # If video is not paused
            ret, self.frame = self.cap.read()
            if not ret: #when we reach the end of the episode 
                self.paused = True  # Pause the video
            else:

                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # If old_gray is not None, we have a previous frame to compare to
                if self.old_gray is not None:

                    if not self.point_edited:
                        # Before processing a new frame:
                        last_frame_points = self.df_field[self.df_field['frame'] == self.current_frame_number - 1]
                        self.p0 = np.array(last_frame_points[['x', 'y']]).astype(np.float32).reshape(-1, 1, 2)

                        # Calculate optical flow using Lucas-Kanade method
                        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

                        frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                        # If points are found, update them
                        if p1 is not None:
                            labels = last_frame_points['label'].tolist()
                            for new_point, label in zip(p1.reshape(-1, 2).tolist(), labels):
                                new_x, new_y = new_point
                                
                                # Check if the point lies inside the frame boundaries 
                                if 0 <= new_x < frame_width and 0 <= new_y < frame_height:

                                    # Check if frame-label combo exists
                                    existing_row = self.df_field[(self.df_field['frame'] == self.current_frame_number) & (self.df_field['label'] == label)]

                                    if not existing_row.empty:  # If the combo exists
                                        row_index = existing_row.index[0]
                                        self.df_field.at[row_index, 'x'] = new_x
                                        self.df_field.at[row_index, 'y'] = new_y
                                    else:
                                        new_row = pd.DataFrame([{'frame': self.current_frame_number, 'type':"field", 'x': new_x, 'y': new_y, 'label': label}])
                                        self.df_field = pd.concat([self.df_field, new_row], ignore_index=True)

                    self.point_edited = False

                    self.draw_frame_points()
                    self.update_treeview()

                    self.frame_slider.set(self.current_frame_number) # Update the slider's position as the video plays
                    self.old_gray = frame_gray.copy()
                    self.current_frame_number += 1  # Increment frame number for the next frame
        
        elif self.paused: #if paused
            if self.old_gray is not None:

                # Fetch the desired frame number from the slider
                desired_frame_number = self.frame_slider.get()

                # If the desired frame is not the current frame, set the video capture position
                if desired_frame_number != self.current_frame_number:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, desired_frame_number)
                    ret, self.frame = self.cap.read()
                    if not ret:
                        self.root.quit()
                        return
                    self.old_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    self.current_frame_number = desired_frame_number

                    self.draw_frame_points() # Annotate points on the frame
                    self.update_treeview()


        self.root.after(10, self.video_loop)  # Call video_loop again after 10 ms

