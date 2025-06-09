from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import cv2
import os
import sys
sys.path.append('../')
from utils import get_bbox_center, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
            batch_size -> we set a batch size as a precaution so that CPU/GPU doesn't get too much load
            detections -> list of predictions
        """
        batch_size = 5
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, device='cuda')
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        """
            If read_from_stub is True, then it reads from the stub_path if it exists and returns the saved tracks.
            If read_from_stub is False, then it runs detect_frames, and tracks entities in the frames and saves the tracking data into the given stub_path if mentioned
        """ 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        """
            Example of tracks dict :

            tracks = {
                "players": [
                    {0 : {"bbox": [0,0,0,0]}, 1 : {"bbox": [0,0,0,0]}}, # frame_num == 0 & contains all tracks in it
                    {0 : {"bbox": [0,0,0,0]}, 1 : {"bbox": [0,0,0,0]}}, # frame_num == 1 & contains all tracks in it
                ],
                "referees": [],
                "ball": []
            }
        """

        for frame_num, detection in enumerate(detections):
            """
                frame_num -> Each video is split into frames, 
                             and frame_num is the index of that frame
                
            """
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            # cls_names_inv is used to easily access cls_name index using the class name

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # As the model is not predicting goalkeepers accurately in every frame, we will consider goalkeepers as players too
            # Convert goalkeeper to player object
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # The ball is directly extracted from detections before tracking as there is only one ball.
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox} 
                    # Here, track_id == 1 because there exists only one ball in a game
        
        # Saving the tracking info into a binary file
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
                
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
            This function is used to draw the ellipse under each player in each frame
        """
        y_bottom = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        x_center = int(x_center)

        width = get_bbox_width(bbox) 

        cv2.ellipse(
            frame,
            center=(x_center, y_bottom),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45, # the line starts from -45deg
            endAngle=235, # the line ends at 235deg,
            # which means from 235deg to -45deg there is no line
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        reactangle_width = 40
        reactangle_height = 20

        x1_rect = x_center - reactangle_width//2
        x2_rect = x_center + reactangle_width//2

        y1_rect = (y_bottom - reactangle_height//2) + 15
        y2_rect = (y_bottom + reactangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10 # if its a 3 digit number, then text starts 10 pixels before from the left

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
            This function is used to draw the triangle over the ball in each frame
        """

        y_top = int(bbox[1]) # y1

        x_center, _ = get_bbox_center(bbox)
        x_center = int(x_center)

        width = get_bbox_width(bbox)

        p1 = [x_center, y_top]
        p2 = [x_center - 12, y_top - 20]
        p3 = [x_center + 12, y_top - 20]
        pts = np.array([p1, p2, p3], np.int32)
        pts.reshape((-1, 1, 2))

        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)

        return frame


    def draw_annotations(self, video_frames, tracks, output_video_path):
        """
            This function is used to draw circles instead of bounding boxes below the players
        """

        # fourcc -> Four Character Code
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID is a common fourcc codec for AVI files
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (video_frames[0].shape[1],video_frames[0].shape[0]))

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                """
                    track_id -> index of the player
                    player_dict -> eg., {0 : {"bbox": [0,0,0,0]}, 1 : {"bbox": [0,0,0,0]}}, # frame_num == 0 & contains all tracks in it
                    player -> value in all key-value pairs in player_dict , i.e in the above example, player -> {"bbox": [0,0,0,0]}
                    player["bbox"] -> x1,y1,x2,y2 coordinates of the players bounding box
                """

                color = player.get("team_color", (0, 0, 255)) # color of the ellipse
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
            
            # Draw Referees
            for _, referee in referee_dict.items():
                color = (0, 255, 255)
                frame = self.draw_ellipse(frame, referee["bbox"], color)

            for _, ball in ball_dict.items():
                color = (0, 255, 0)
                frame = self.draw_triangle(frame, ball["bbox"], color)

            print("Frame done", frame_num)
            out.write(frame)
        
        out.release()
        print("Video Saved: ", output_video_path)