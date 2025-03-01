import cv2

def read_video(video_path):
    """
        This function reads the input video files saved at 
        `video_path` and stores its frames in a list using cv2
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    """
        This function takes in video frames and path to save the output
        video and saves the list of frames as a video using OpenCV
    """
    # fourcc -> Four Character Code
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # XVID is a common fourcc codec for AVI files
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    # shape[1] -> width , shape[0] -> height
    for frame in output_video_frames:
        out.write(frame)
    out.release()