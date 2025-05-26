import cv2
import os
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics.pairwise import pairwise_distances
import detect_and_align

tf.disable_v2_behavior()

# Initialize global variables
model = '20170512-110547/20170512-110547.pb'
id_folder = '20170512-110547/ids'

# Load the TensorFlow model
def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with tf.gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")

class IdData:
    """Keeps track of known identities and calculates id matches"""
    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold):
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None
        image_paths = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return
        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths += [os.path.join(id_dir, img) for img in os.listdir(id_dir)]
        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)
        self.id_names = [os.path.basename(os.path.dirname(path)) for path in id_image_paths]

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            aligned_images += face_patches
            id_image_paths += [image_path] * len(face_patches)
        return np.stack(aligned_images), id_image_paths

    def find_matching_ids(self, embs):
        if self.id_names:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distance_treshold:
                    matching_ids.append(self.id_names[min_index])
                    matching_distances.append(distance_row[min_index])
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        return matching_ids, matching_distances

def Recognise(image_path):
    output_path = "static/out.jpg"
    with tf.Graph().as_default():
        with tf.Session() as sess:
            mtcnn = detect_and_align.create_mtcnn(sess, None)
            load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            id_data = IdData(id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, 1.0)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(image, mtcnn)
            if face_patches:
                face_patches = np.stack(face_patches)
                feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                embs = sess.run(embeddings, feed_dict=feed_dict)
                matching_ids, _ = id_data.find_matching_ids(embs)
                print(f"\n\n\n{matching_ids}\n\n\n")
                return matching_ids[0] if matching_ids[0] else "Unknown"
            else:
                return "No face detected"


def RecogniseVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            mtcnn = create_mtcnn(sess, None)
            load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            id_data = IdData(id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, 1.0)

            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.resize(frame, (640, 480))
                
                if not ret:
                    break

                result = _process_frame(frame, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, id_data)
                print("Detected:", result)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


# if __name__ == "__main__":
#     image_path = input("Enter the path to the image: ").strip()
#     if os.path.exists(image_path):
#         result = Recognise(image_path)
#         print(f"Result: {result}")
#     else:
#         print("Error: Image path does not exist.")
