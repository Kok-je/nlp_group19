class ModelCard:
    """
    A class to represent a model card for a machine learning model.

    Attributes
    ----------
    model_name : str
        The name of the model.
    version : str
        The version of the model.
    description : str
        A brief description of the model.
    author : str
        The author of the model.
    price : double
        The API price by model provider by whom the model is released.
    size : int
        The size of the model (in Billion Parameters).

    file_path : str
        Path to the model outputs on train.csv
    Methods
    -------
    display_card():
        Displays the model card in a pretty format.
    """

    def __init__(self, model_name, version, description, author, price,size,file_path,partition = None,important = True):
        self.model_name = model_name
        self.version = version
        self.description = description
        self.author = author
        self.price = price
        self.size = size
        self.file_path = file_path
        self.report = None
        self.partition = partition
        self.important = important

    def display_card(self, verbose = False):
        """Display the model card information in a formatted way."""
        print(f"{self.author} - {self.model_name} {self.version}")
        print(f"Size: {self.size} Billion Parameters")
        print(f"Price: ${self.price}/M tokens")
        print(f"Report: {self.report.display_report()}")


        if verbose:
            print(f"Model: {self.model_name}")
            print(f"Version: {self.version}")
            print(self.description)
            print(f"Author: {self.author}")
            print(f"Price: ${self.price}/M tokens")
            print(f"Size: {self.size} Billion Parameters")
            print(f"Report: {self.report.display_report()}")
