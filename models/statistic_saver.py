import json
import os


class Statistics:
    """
    All the attributes are public. There is no needed for particular encapsulation and it will make the training code
    less cluttered.
    """

    def __init__(self, with_timestamp=True) -> None:
        super().__init__()
        # Loss over time
        self.loss: list = []

        # Accuracy during training. Can be by epoch or by batch
        self.training_accuracy: list = []
        self.validation_accuracy: list = []

        # Only one value because only tested in the end
        self.testing_accuracy: float = -1

        # Anything you might want to add about the model, training process, or anything useful to understand the data
        self.other_comments: str = None

        # A description of the model architecture (layers, batch_norm, drop out, input size, etc)
        self.model_structure: str = None

    @staticmethod
    def load(filename: str, folder: str):
        """
        Load statistics from a file
        :param filename: the name of the file without the extension. json files are loaded by default.
        :param folder: the folder where the file is to be saved. Can be relative or absolute path
        :return: an instance of Statistics class
        """
        with open(os.path.join(folder, filename + '.json'), 'r') as f:
            data = json.load(f)
            statistics = Statistics()
            statistics.loss = data.loss
            statistics.training_accuracy = data.training_accuracy
            statistics.validation_accuracy = data.validation_accuracy
            statistics.testing_accuracy = data.testing_accuracy
            statistics.other_comments = data.other_comments
            statistics.model_structure = data.model_structure
            return statistics

    def save(self, filename: str, folder: str):
        """
        Save statistics to a file
        :param filename: the name of the file without the extension. json files are loaded by default.
        :param folder: the folder where the file is to be saved. Can be relative or absolute path
        :return:
        """
        with open(os.path.join(folder, filename + '.json'), 'w') as f:
            json.dump(self.__dict__, f)
