class DVLExtractor(object):

    def __init__(self, filename):
        """
        Initialize the DVLExtractor with the given file.

        Args:
            filename (str): Path to the DVL file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.dvl_struct = dict(
            version="",
            stattion_id=0,
            tag="",
            lat=0,
            lon=0,
            date=None,
            doy=0,
            time=None,
        )
        return

    def read_file(self):
        """
        Reads the file line by line into a list.

        Returns:
            list: A list of strings, each representing a line from the file.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def extract(self):
        """
        Main method to extract data from the DVL file and populate the sao_struct dictionary.

        Returns:
            dict: The populated dvl_struct dictionary containing all extracted data.
        """
        # Read file lines
        dvl_arch = self.read_file()
        dvl_arch = list(filter(None, dvl_arch[0].split()))
        print(dvl_arch)
        return self.dvl_struct
