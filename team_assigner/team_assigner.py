import cv2
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image, reshape=True):
        # Reshape the image into a 2d array
        if reshape == True:
            image = image.reshape(-1, 3)

        # Perform KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_img = image[:len(image)//2]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_img)

        # Get the cluster labels
        labels  = kmeans.labels_

        # Reshape labels into original image shape
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])

        # Get player cluster
        corner_clusters = [clustered_img[0, 0], clustered_img[0, -1], clustered_img[-1, 0], clustered_img[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get player color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = self.get_clustering_model(player_colors, reshape=False)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        print(self.team_colors)

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # as kmeans return 0 or 1, but team_id is either 1 or 2

        self.player_team_dict[player_id] = team_id

        return team_id
    
    def crop_player_image(self, tracks, video_frames, output_image_path):
        """
            Save cropped image of a player
        """

        for track_id, player in tracks['players'][0].items():
            bbox = player["bbox"]
            frame = video_frames[0]

            # Crop bbox from frame
            cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            # Save the cropped image
            cv2.imwrite(f"{output_image_path}/player_{track_id}.jpg", cropped_image)

            print(f"Saved cropped player image at {output_image_path}/player_{track_id}.jpg")
            break