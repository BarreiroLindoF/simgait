import json
import os

# Class containing a list of statistics model and the accuracy mean over each validation set
class CrossValStatistics:
    def __init__(self) -> None:
        super().__init__()

        # List of all cross validated statistic model
        self.stat_models: list = []

        # Accuracy mean over validation sets
        self.validation_mean_accuracy: float = -1

        # Accuracy mean over test sets
        self.test_mean_accuracy: float = -1

        # Variables for McNemar contingency table storing correct and wrong total of predictions over all cross validated set for a model
        self.predictionResults: list = []

    @staticmethod
    def load(filename: str, folder: str = None):
        """
        Load statistics from a file
        :param filename: the name of the file without the extension. json files are loaded by default.
        :param folder: the folder where the file is to be saved. Can be relative or absolute path.
               If no folder is specified, it will be created on the same folder as the running script.
        :return: an instance of Statistics class
        """
        if folder is None:
            folder = '.'
        with open(os.path.join(folder, filename + '.json'), 'r') as f:
            data = json.load(f)
            cross_statistics = CrossValStatistics()
            cross_statistics.stat_models = data['stat_models']
            cross_statistics.validation_mean_accuracy = data['validation_mean_accuracy']
            cross_statistics.test_mean_accuracy = data['test_mean_accuracy']
            cross_statistics.predictionResults = data['predictionResults']
            return cross_statistics

    def save(self, filename: str, folder: str = None):
        """
        Save statistics to a file
        :param filename: the name of the file without the extension. json files are loaded by default.
        :param folder: the folder where the file is to be saved. Can be relative or absolute path.
               If no folder is specified, it will be created on the same folder as the running script.
        :return:
        """
        if folder is None:
            folder = '.'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, filename + '.json'), 'w') as f:
            json.dump(self.__dict__, f, default=lambda x: x.__dict__)

# Statistic class containing training history of a model
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

        # Validation loss during the training
        self.validation_loss: list = []

        # Only one value because only tested at the end
        self.test_accuracy: float = -1

        # Anything you might want to add about the model, training process, or anything useful to understand the data
        self.other_comments: str = None

        # A description of the model architecture (layers, batch_norm, drop out, input size, etc)
        self.model_structure: str = None


if __name__ == '__main__':
    """
    DO NOT REMOVE THE IF.
    This condition is only True if this file is executed manually. 
    If the condition is removed, the tests will be executed even when importing the class.
    Also, make sure to run the script as Administrator in order to be able to remove the temporary folders.
    """
    statistics = Statistics()

    statistics.loss = [1., 2.]
    statistics.loss.append(3.)
    statistics.training_accuracy = [3., 4.]
    statistics.validation_accuracy = [5., 95.]
    statistics.testing_accuracy = .98
    statistics.other_comments = "The training was done using non-normalized data"
    statistics.model_structure = "Linear -> Relu -> Linear -> Softmax"

    statistics.save('test')

    loaded = Statistics.load('test')

    assert statistics.loss == loaded.loss
    assert statistics.training_accuracy == loaded.training_accuracy
    assert statistics.validation_accuracy == loaded.validation_accuracy
    assert statistics.testing_accuracy == loaded.testing_accuracy
    assert statistics.other_comments == loaded.other_comments
    assert statistics.model_structure == loaded.model_structure

    print("All tests passed, removing the temporary files...")

    os.remove('test.json')
