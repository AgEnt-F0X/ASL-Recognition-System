import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except (ImportError, AttributeError):
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path="model/keypoint_classifier/keypoint_classifier.tflite",
        num_threads=1,
    ):
        self.interpreter = Interpreter(
            model_path=model_path, num_threads=num_threads
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]["index"]

        result = self.interpreter.get_tensor(output_details_tensor_index)
        probs = np.squeeze(result)
        result_index = int(np.argmax(probs))
        self.last_confidence = float(probs[result_index])
        top_two = np.partition(probs, -2)[-2:]
        self.last_margin = float(top_two[1] - top_two[0])
        self.last_probabilities = probs.copy()

        return result_index
