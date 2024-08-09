import h5py
import numpy as np

phoneme_to_int = {
    "a": 0, "aj": 1, "aw": 2, "aː": 3, "b": 4, "bʲ": 5, "c": 6, "cʰ": 7, "cʷ": 8,
    "d": 9, "dʒ": 10, "dʲ": 11, "d̪": 12, "e": 13, "ej": 14, "eː": 15, "f": 16, "fʲ": 17, "h": 18,
    "i": 19, "iː": 20, "j": 21, "k": 22, "kp": 23, "kʰ": 24, "kʷ": 25, "l": 26, "m": 27, "mʲ": 28,
    "m̩": 29, "n": 30, "o": 31, "ow": 32, "oː": 33, "p": 34, "pʰ": 35, "pʲ": 36, "pʷ": 37, "s": 38,
    "t": 39, "tʃ": 40, "tʰ": 41, "tʲ": 42, "tʷ": 43, "t̪": 44, "u": 45, "uː": 46, "v": 47, "vʲ": 48,
    "w": 49, "z": 50, "æ": 51, "ç": 52, "ð": 53, "ŋ": 54, "ɐ": 55, "ɑ": 56, "ɑː": 57, "ɒ": 58,
    "ɒː": 59, "ɔ": 60, "ɔj": 61, "ɖ": 62, "ə": 63, "əw": 64, "ɚ": 65, "ɛ": 66, "ɛː": 67, "ɜ": 68,
    "ɜː": 69, "ɝ": 70, "ɟ": 71, "ɟʷ": 72, "ɡ": 73, "ɡʷ": 74, "ɪ": 75, "ɫ": 76, "ɲ": 77, "ɹ": 78,
    "ɾ": 79, "ʃ": 80, "ʈ": 81, "ʈʲ": 82, "ʈʷ": 83, "ʉ": 84, "ʉː": 85, "ʊ": 86, "ʋ": 87, "ʎ": 88,
    "ʒ": 89, "θ": 90
}
int_to_phoneme = {v: k for k, v in phoneme_to_int.items()}


class HDF5_Container:
    """
    A class to handle HDF5 file creation, data storage, and retrieval.
    """
    def __init__(self, path):
        """
        Initialize an HDF5_Container instance.
        
        Parameters:
        path (str): The path to the HDF5 file.
        """
        self.path = path

    def create_hdf5_file(self):
        """
         Create a new HDF5 file.
         """
        self.file = h5py.File(self.path, 'w')

    def close_hdf5_file(self):
        """
        Close the HDF5 file.
        """
        self.file.close()

    def add_video_data_batch(self, video_name, emotion, training_frames):
        """
        Add a batch of video data to the HDF5 file.
        
        Parameters:
        video_name (str): The name of the video.
        emotion (str): The emotion label for the video.
        training_frames (list): A list of Training_Frame objects containing the data to add.
        """
        if emotion not in self.file:
            emotion_group = self.file.create_group(emotion)
        else:
            emotion_group = self.file[emotion]

        for frame in training_frames:
            frame_index = frame.frame_index
            if str(frame_index) in emotion_group:
                frame_group = emotion_group[str(frame_index)]
            else:
                frame_group = emotion_group.create_group(str(frame_index))

            if isinstance(frame.landmarks[0], list):
                # Flatten the list of landmarks
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for sublist in frame.landmarks for lm in sublist], dtype=np.float64)
            else:
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in frame.landmarks], dtype=np.float64)

            frame_group.create_dataset('landmarks', data=landmarks_array, dtype='float64')
            frame_group.create_dataset('mel', data=frame.mel_segment, dtype='float64')
            translated_phoneme = phoneme_to_int.get(frame.phoneme, -1)  # Use -1 for unknown phonemes
            frame_group.create_dataset('phoneme', data=translated_phoneme, dtype='int32')
            print(f"Created datasets for {video_name}/{emotion}/{frame_index}")

    def read_video_data(self, path):
        """
        Read video data from the HDF5 file.
        
        Parameters:
        path (str): The path to the HDF5 file.
        
        Returns:
        dict: A dictionary containing the video data.
        """
        all_data = {}
        with h5py.File(path, 'r') as file:       

            emotion = list(file.keys())[0]  # Since there is only one emotion
            emotion_group = file[emotion]
            
            emotion_data = {}
            for frame_index in sorted(emotion_group, key=lambda x: int(x)):
                frame_group = emotion_group[frame_index]

                landmarks = frame_group['landmarks'][:]
                mel = frame_group['mel'][:]
                phoneme = int(frame_group['phoneme'][()])

                emotion_data[int(frame_index)] = {
                    'landmarks': landmarks,
                    'mel': mel,
                    'phoneme': phoneme
                }

            all_data = {                
                'emotion': emotion,
                'frames': emotion_data
            }

        return all_data