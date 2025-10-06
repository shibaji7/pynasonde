"""Utilities to extract tabular metadata from ionogram image files.

This module contains:class:`IonogramImageExtractor` which uses OpenCV
and Tesseract OCR to extract textual parameter tables and header
metadata from ionogram images. The functionality is intentionally
focused and small so it can be used in documentation examples.
"""

import datetime as dt

import cv2
import numpy as np
import pandas as pd
import pytesseract
from loguru import logger


class IonogramImageExtractor(object):
    """Extractor for ionogram images using OpenCV + Tesseract OCR.

    The extractor offers helpers to read date/time information from the
    filename and to OCR specific regions of the image to parse
    parameter tables and header fields into pandas DataFrames.
    """

    def __init__(
        self,
        filepath: str,
        extract_time_from_name: bool = True,
        date: dt.datetime = None,
        filestr_date_format: str = "ion%y%m%d_%H%M%S.png",
    ):
        """Create an IonogramImageExtractor.

        Parameters:
            filepath: str
                Path to the ionogram image file.
            extract_time_from_name: bool, optional
                If True (default), attempt to parse the timestamp from the
                filename using ``filestr_date_format``.
            date: datetime.datetime, optional
                Manually provided date; if None and
                ``extract_time_from_name`` is True the date will be parsed
                from the filename.
            filestr_date_format: str, optional
                Format string used when parsing the filename timestamp.
        """
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
    ) -> str:
        """Extract text from a cropped region of the image using Tesseract OCR.

        Parameters:
            crop_axis: numpy.ndarray, optional
                2x2 array specifying the crop region as [[y1, y2], [x1, x2]].
            cv_props: dict, optional
                OpenCV thresholding parameters (keys: 'thresh', 'maxval', 'type').
            OCR_custom_config: str, optional
                Tesseract configuration string (e.g. '--oem 3 --psm 6').

        Returns:
            Text extracted from the cropped region.
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
    ) -> pd.DataFrame:
        """Parse a left-side artist-parameters table from the ionogram image.

        The function OCRs a cropped region, extracts line-wise
        key/value pairs, applies simple text replacements and converts
        values to float where possible. The result is returned as a
        single-row:class:`pandas.DataFrame`.

        Parameters:
            crop_axis: numpy.ndarray, optional
                Crop region as [[y1, y2], [x1, x2]].
            cv_props: dict, optional
                OpenCV thresholding parameters.
            OCR_custom_config: str, optional
                Tesseract config string.
            lines_to_extracted: int, optional
                Number of lines to keep from OCR output (negative values
                trim from the end).
            word_filtes_for_table_values: dict, optional
                Mapping of substrings to replace in extracted values before
                conversion.

        Returns:
            Single-row DataFrame with parsed parameter names and values.
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
    ) -> pd.DataFrame:
        """Extract and parse header fields from an ionogram image.

        Parameters:
            crop_axis: numpy.ndarray, optional
                Crop region for the header area.
            cv_props: dict, optional
                OpenCV thresholding parameters.
            OCR_custom_config: str, optional
                Tesseract config string.
            word_filtes_for_table_values: dict, optional
                Mapping of substrings to replace in both header keys and
                values.

        Returns:
            Single-row DataFrame with header keys and values when found.
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
    return record


if __name__ == "__main__":
    iie = IonogramImageExtractor("tmp/ion250610_210301.png")
    iie.parse_artist_params_table()
    iie.extract_header()
