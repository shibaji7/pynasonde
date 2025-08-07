import datetime as dt

import cv2
import numpy as np
import pandas as pd
import pytesseract
from loguru import logger


class IonogramImageExtractor(object):
    """
    A class to extract and process data from standard Ionogram image files.

    This class provides methods to extract metadata (such as date and time) from the image filename,
    and to parse parameter tables from the image using OCR techniques.

    Attributes:
        filepath (str): The path to the ionogram image file.
        extract_time_from_name (bool): Whether to extract the timestamp from the filename.
        date (datetime.datetime): The date and time associated with the image.
        file_ext (str): The file extension of the image.
        filestr_date_format (str): The format string to parse the date from the filename.
    """

    def __init__(
        self,
        filepath: str,
        extract_time_from_name: bool = True,
        date: dt.datetime = None,
        filestr_date_format: str = "ion%y%m%d_%H%M%S.png",
    ):
        self.filepath = filepath
        self.extract_time_from_name = extract_time_from_name
        self.date = date
        self.file_ext = filepath.split(".")[-1]
        self.filestr_date_format = filestr_date_format
        if extract_time_from_name:
            self.date = dt.datetime.strptime(
                filepath.split("/")[-1], self.filestr_date_format
            )
            logger.info(f"Parsed date from file name: {self.date}")
        return

    def extract_text(
        self,
        crop_axis: np.array = np.array([[50, 650], [0, 220]]),
        cv_props: dict = dict(
            thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        ),
        OCR_custom_config: str = r"--oem 3 --psm 6",
    ):
        """
        Extracts text from a specified cropped region of an image using OCR.
        This method loads an image from the file path specified in the instance, crops a region defined by `crop_axis`, applies grayscale conversion and thresholding for improved OCR accuracy, and then extracts text using Tesseract OCR.
        Args:
            crop_axis (np.array, optional): A 2x2 numpy array specifying the crop region in the format [[y1, y2], [x1, x2]]. Defaults to np.array([[50, 650], [0, 220]]).
            cv_props (dict, optional): Dictionary of OpenCV thresholding parameters, including 'thresh', 'maxval', and 'type'. Defaults to {'thresh': 180, 'maxval': 255, 'type': cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU}.
            OCR_custom_config (str, optional): Custom configuration string for Tesseract OCR. Defaults to r"--oem 3 --psm 6".
        Returns:
            str: The text extracted from the specified region of the image.
        """
        # Load the image
        img = cv2.imread(self.filepath)
        # Crop the left table region (adjust these values as needed)
        # These coordinates are (y1:y2, x1:x2)
        cropped = img[
            crop_axis[0, 0] : crop_axis[0, 1], crop_axis[1, 0] : crop_axis[1, 1]
        ]
        # Optional: Convert to grayscale and apply threshold for better OCR
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, cv_props["thresh"], cv_props["maxval"], cv_props["type"]
        )

        # OCR extraction
        text = pytesseract.image_to_string(thresh, config=OCR_custom_config)
        return text

    def parse_artist_params_table(
        self,
        crop_axis: np.array = np.array([[50, 650], [0, 220]]),
        cv_props: dict = dict(
            thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        ),
        OCR_custom_config: str = r"--oem 3 --psm 6",
        lines_to_extracted: int = -8,
        word_filtes_for_table_values: dict = {"N/A": "nan", ":": "."},
    ):
        """
        Parses a table of artist parameters from a specified region of an image file using OCR.
        This method loads an image from the instance's `filepath`, crops a specified region containing a table,
        applies preprocessing for optimal OCR (grayscale and thresholding), and extracts parameter names and values
        from the table using Tesseract OCR. The extracted parameters are cleaned and converted to floats, then
        returned as a pandas DataFrame.
        Parameters:
            crop_axis (np.array): 2x2 array specifying the crop region as [[y1, y2], [x1, x2]]. Defaults to [[50, 650], [0, 220]].
            cv_props (dict): Dictionary of OpenCV thresholding properties. Defaults to {'thresh': 180, 'maxval': 255, 'type': cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU}.
            OCR_custom_config (str): Custom configuration string for Tesseract OCR. Defaults to "--oem 3 --psm 6".
            lines_to_extracted (int): Number of lines to extract from the OCR output. Defaults to -8 (all but last 8 lines).
            word_filtes_for_table_values (dict): Dictionary mapping substrings to replace in extracted table values. Defaults to {"N/A": "nan", ":": "."}.
        Returns:
            pd.DataFrame: DataFrame containing the parsed parameter names and their corresponding float values.
        Logs:
            Logs the parsed records at the info level.
        Note:
            Adjust the crop_axis and cv_props parameters as needed for different image layouts or OCR quality.
        """
        text = self.extract_text(crop_axis, cv_props, OCR_custom_config)
        # Extract all individual parameters
        record = dict()
        if len(text) > 0:
            lines = text.split("\n")
            # Filter all lines based on empty lines
            lines = [l for l in lines if len(l) > 0][:lines_to_extracted]
            for line in lines:
                words = list(filter(None, line.split(" ")))
                if len(words) >= 2:
                    for fw in word_filtes_for_table_values.keys():
                        words[1] = words[1].replace(
                            fw, word_filtes_for_table_values[fw]
                        )
                    record[words[0]] = float(words[1])
        record = pd.DataFrame.from_dict([record])
        logger.info(f"Parsed records: \n {record}")
        return record

    def extract_header(
        self,
        crop_axis: np.array = np.array([[0, 50], [100, 800]]),
        cv_props: dict = dict(
            thresh=180, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        ),
        OCR_custom_config: str = r"--oem 3 --psm 6",
        word_filtes_for_table_values: dict = {",": "", ":": "."},
    ):
        """
        Extracts header information from an image using OCR and returns it as a pandas DataFrame.
        This method processes a cropped region of an image, applies computer vision properties for thresholding,
        and uses OCR to extract text. It then parses the first two lines of the extracted text as header columns
        and their corresponding values, applies optional word replacements, and constructs a DataFrame from the result.
        Parameters:
            crop_axis (np.array, optional): Coordinates for cropping the image before OCR. Defaults to np.array([[0, 50], [100, 800]]).
            cv_props (dict, optional): Properties for OpenCV thresholding. Defaults to {'thresh': 180, 'maxval': 255, 'type': cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU}.
            OCR_custom_config (str, optional): Custom configuration string for the OCR engine. Defaults to r"--oem 3 --psm 6".
            word_filtes_for_table_values (dict, optional): Dictionary of string replacements to apply to header columns and values. Defaults to {",": "", ":": "."}.
        Returns:
            pd.DataFrame: A DataFrame containing the extracted header information as a single row.
        Logs:
            - Extracted text from the image at debug level.
            - Parsed records as a DataFrame at info level.
        """
        text = self.extract_text(crop_axis, cv_props, OCR_custom_config)

        # Extract all individual parameters
        record = dict()
        if len(text) > 0:
            logger.debug(f"Extracted text: \n {text}")
            lines = text.split("\n")
            # Filter all lines based on empty lines
            lines = [l for l in lines if len(l) > 0]
            if len(lines) >= 2:
                header_columns = list(filter(None, lines[0].split(" ")))
                header_values = list(filter(None, lines[1].split(" ")))

                for fw in word_filtes_for_table_values.keys():
                    header_columns, header_values = (
                        [
                            w.replace(fw, word_filtes_for_table_values[fw])
                            for w in header_columns
                        ],
                        [
                            w.replace(fw, word_filtes_for_table_values[fw])
                            for w in header_values
                        ],
                    )
                record = dict(zip(header_columns, header_values))
        record = pd.DataFrame.from_dict([record])
        logger.info(f"Parsed records: \n {record}")
        return


if __name__ == "__main__":
    iie = IonogramImageExtractor("tmp/ion250610_210301.png")
    iie.parse_artist_params_table()
    iie.extract_header()
