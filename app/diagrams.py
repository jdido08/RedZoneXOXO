import numpy as np
import supervision as sv
from ultralytics import YOLO
from roboflow import Roboflow
import pandas as pd
import cv2
import numpy as np
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio_v2
import io
from IPython.display import display, Image
import imageio.v2 as imageio_v2
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def get_video_fps(source_path):
    # Open the video file
    cap = cv2.VideoCapture(source_path)

    # Get FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(f"FPS of the video: {fps}")
    return fps

def create_vertical_football_field(linenumbers=True,
                                         endzones=True,
                                         line_of_scrimage=None,
                                         first_down_line=None,
                                         yardlines=True,
                                         hashmarks=True,
                                         figsize=(6.33, 12),
                                         dpi=72,
                                         scale=1,
                                         border=True):
    """
    Function that plots the football field for viewing plays in Tecmo Super Bowl style.
    Allows for showing or hiding endzones. The field is vertical.
    """
    base_fontsize = 14
    base_markersize = 30
    base_linewidth = 2

    scaled_fontsize = base_fontsize * scale
    scaled_markersize = base_markersize * scale
    scaled_linewidth = base_linewidth * scale


    desired_width = 2048
    desired_height = 3888
    figsize_width = desired_width / dpi
    figsize_height = desired_height / dpi
    figsize = (figsize_width, figsize_height)
    
    #figsize = (figsize[0] * scale, figsize[1] * scale)
    
    rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=0.1*scale,
                             edgecolor='r', facecolor='lightgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    ax.add_patch(rect)
    
    if border:  # draw border if border is True
        border_rect = patches.Rectangle((0, 0), 53.3, 120, linewidth=scaled_linewidth, edgecolor='white', facecolor='none', zorder=1)
        ax.add_patch(border_rect)

    if yardlines:
        plt.plot([0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0],
                 [10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80, 80, 90, 90, 100, 100, 110, 110, 120, 120],
                 color='white', linewidth=scaled_linewidth)

    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 53.3, 10, linewidth=0.1*scale, edgecolor='r', facecolor='gray', alpha=0.2, zorder=0)
        ez2 = patches.Rectangle((0, 110), 53.3, 10, linewidth=0.1*scale, edgecolor='r', facecolor='gray', alpha=0.2, zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
        
        # Add diagonal lines in endzones
        for i in range(0, 10, 1):
            ax.plot([0, 53.3], [i, i+1], color='white', alpha=0.2, linewidth=scaled_linewidth)
            ax.plot([0, 53.3], [110+i, 111+i], color='white', alpha=0.2, linewidth=scaled_linewidth)
    
    plt.ylim(-10, 120)
    plt.xlim(-5, 58.3)
    plt.axis('off')
    
    if linenumbers:
        for y in range(20, 110, 10):
            numb = y
            if y > 50:
                numb = 120 - y
            plt.text(2, y, str(numb - 10),
                     verticalalignment='center',
                     fontsize=scaled_fontsize,  
                     color='white', fontweight='bold', rotation=-90)
            plt.text(53.3 - 2, y, str(numb - 10),
                     verticalalignment='center',
                     fontsize=scaled_fontsize, 
                     color='white', ha='right', fontweight='bold',rotation=-90)
    
    if hashmarks:
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        for y in hash_range:
            ax.plot([0.4, 0.7], [y, y], color='white', linewidth=scaled_linewidth)
            ax.plot([53.0, 52.5], [y, y], color='white', linewidth=scaled_linewidth)
            ax.plot([22.91, 23.57], [y, y], color='white', linewidth=scaled_linewidth)
            ax.plot([29.73, 30.39], [y, y], color='white', linewidth=scaled_linewidth)

    if line_of_scrimage is not None:
        hl = line_of_scrimage + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        
    if first_down_line is not None:
        hl = first_down_line + 10
        plt.plot([hl, hl], [0, 53.3], color='red')

    return fig, ax

def create_football_formations_diagrams(df, player_paths=False, player_boxes=None):
    markersize = 50
    fontsize = 20
    
        # Create Tecmo football field
    fig, ax = create_vertical_football_field(
                                                    linenumbers=True,
                                                    yardlines=True,
                                                    hashmarks=True,
                                                    scale=4.5)

    # Plotting players
    for idx, row in df.iterrows():
        try:
            #note that in vertical field alignment x becomes y and y becomes x
            #additionally because we need to change how y field is calced 
            
            marker = 'bo'
            if(row['team'] == 'offense' and row['calc'] == "Detection"):
                marker = 'bo'
            elif(row['team'] == 'offense' and row['calc'] == "Estimated"):
                marker = 'bs'
            elif(row['team'] == 'offense' and row['calc'] == "Interpolated"):
                marker = 'bv'
            elif(row['team'] == 'defense' and row['calc'] == "Detection"):
                marker = 'ro'
            elif(row['team'] == 'defense' and row['calc'] == "Estimated"):
                marker = 'rs'
            elif(row['team'] == 'defense' and row['calc'] == "Interpolated"):
                marker = 'rv'
            
            ax.plot((53-row['field_y']), row['field_x'], marker, markersize=markersize)
            plt.text((53-row['field_y']), row['field_x'], row['label'], ha='center', va='center', color='w', fontsize=fontsize)
        
        except:
            print("ERROR: ", row['frame'], " , ", row['field_x'], " , ", row['field_y'])

    # Additional plot settings
    ax.set_aspect('equal')
    plt.savefig('C:/Users/jdido/footballforfellas/diagram_tests/test.png')  # Saves the plot as a PNG image
    return fig, ax


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image to the specified width and height.

    Args:
    - image (ndarray): Image to be resized.
    - width (int, optional): Desired width after resizing.
    - height (int, optional): Desired height after resizing.
    - inter (int, optional): Interpolation method for resizing.

    Returns:
    - ndarray: Resized image.
    """
    # Get the image dimensions
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # If width is None, compute the ratio of the height and construct the dimensions
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # If height is None, compute the ratio of the width and construct the dimensions
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


import os
def create_football_diagram_video(df, source_video_path):
    
    fps  = get_video_fps(source_video_path)

    #downsample 
    # n = int(fps / 12)  # Calculate the downsampling rate
    # df = df[df['frame'] % n == 0]  # Filter the dataframe based on the downsampling rate

    frames_numbers = df['frame'].unique()

    temp_files = []  # List to store paths of temporary chunk files
    chunk_frames = []
    chunk_size = 50

    for frame_num in frames_numbers:
        print(frame_num)
        frame_df = df[df['frame'] == frame_num]
        fig, ax = create_football_formations_diagrams(frame_df)
        plt.tight_layout()
        plt.axis("off")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Read the image from the buffer
        image = imageio_v2.imread(buf)

        # Resize the image to the desired resolution
        resized_image = resize_image(image, width=1024)  # Adjust the width as per your needs

        chunk_frames.append(resized_image)
        #chunk_frames.append(imageio_v2.imread(buf))
        plt.close()
        buf.close()

        # If chunk is full, save it to a temporary file
        if len(chunk_frames) == chunk_size:
            temp_file = f"temp_chunk_{len(temp_files)}.mp4"
            imageio_v2.mimwrite(temp_file, chunk_frames, fps=12)
            temp_files.append(temp_file)
            chunk_frames = []  # Clear the chunk
    
    # Handle any remaining frames in the chunk
    if chunk_frames:
        temp_file = f"temp_chunk_{len(temp_files)}.mp4"
        imageio_v2.mimwrite(temp_file, chunk_frames, fps=12)
        temp_files.append(temp_file)


    # Combine all the temporary chunk files into the final video
    video_path = source_video_path[:-4] + "_diagram.mp4"
    with imageio_v2.get_writer(video_path, fps=fps) as writer:
        for temp_file in temp_files:
            with imageio_v2.get_reader(temp_file) as reader:
                for frame in reader:
                    writer.append_data(frame)
            os.remove(temp_file)  # Delete the temporary chunk file


    return video_path
    
    
