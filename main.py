from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read Video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    # best.pt is the best set parameters we got by training the model in google colab

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")

    # Draw output
    ## Draw object tracks & save
    output_path = 'output_videos/output_video.avi'
    tracker.draw_annotations(video_frames, tracks, output_path)

    # # Save Video
    # save_video(output_video_frames, output_path)
if __name__ == "__main__":
    main()