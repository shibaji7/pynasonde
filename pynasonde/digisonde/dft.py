import datetime as dt
import struct

from loguru import logger


class DftExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ):
        """
        Initialize the DftExtractor with the given file.

        Args:
            filename (str): Path to the dft file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        if extract_time_from_name:
            date = self.filename.split("_")[-1].replace(".SAO", "").replace(".sao", "")
            self.date = dt.datetime(int(date[:4]), 1, 1) + dt.timedelta(
                int(date[4:7]) - 1
            )
            self.date = self.date.replace(
                hour=int(date[7:9]), minute=int(date[9:11]), second=int(date[11:13])
            )
            logger.info(f"Date: {self.date}")
        if extract_stn_from_name:
            self.stn_code = self.filename.split("/")[-1].split("_")[0]
            logger.info(f"Station code: {self.stn_code}")
        return

    def interpret_binary_data(self, data):
        """
        Interpret binary data in different ways: hex, ASCII, and numerical.

        Args:
            data (bytes): Binary data to interpret.
        """
        print("Hexadecimal Representation:")
        print(data.hex())  # Hex representation

        print("\nASCII Representation (Printable Characters Only):")
        ascii_data = "".join(chr(b) if 32 <= b <= 126 else "." for b in data)
        print(ascii_data)  # Replace non-printable bytes with '.'

        print("\nFirst 16 Integers (if applicable):")
        integers = [int(b) for b in data[:16]]
        print(integers)

        print("\nFirst 4 Floats (if applicable):")
        if len(data) >= 16:
            floats = struct.unpack("4f", data[:16])
            print(floats)

    def fetch_data_by_block_case(self, block_size: int = 4096, case_size: int = 256):
        """
        Reads the file line by line into a list.

        Returns:
            list: A list of strings, each representing a line from the file.
        """
        parsed_data = []  # Store all cases parsed from the file
        with open(self.filename, "rb") as file:
            block_count = 0
            while True:
                # Read a 4096-byte block
                block = file.read(block_size)
                if not block:
                    break  # End of file
                block_count += 1
                print(type(block), len(block))
                print(type(block[0:8]), len(block[0:8]))
                print(self.interpret_binary_data(block))
                record_type = block[0:8]  # First byte is the Record Type
                print(record_type)
                if record_type != 0x0A:  # Verify the Record Type
                    raise ValueError(
                        f"Unexpected Record Type in block {block_count}: {record_type}"
                    )
                for i in range(16):  # Process 16 cases per block
                    start = 1 + i * case_size
                    end = start + case_size

                    if end > len(block):
                        break  # End of block

                    case = block[start:end]

                    # Extract amplitudes and phases
                    amplitudes = list(case[:128])  # First 128 bytes
                    phases = list(case[128:])  # Next 128 bytes

                    parsed_data.append(
                        {
                            "block": block_count,
                            "case": i + 1,
                            "amplitudes": amplitudes,
                            "phases": phases,
                        }
                    )

            # Check for end markers (256 bytes of 0xEE at the end of the file)
            footer = file.read()
            if footer and all(byte == 0xEE for byte in footer[:256]):
                print("End marker detected.")
        return

    def extract(self):
        """
        Main method to extract data from the dft file and populate the sao_struct dictionary.

        Returns:
            dict: The populated dft_struct dictionary containing all extracted data.
        """
        self.fetch_data_by_block_case()
        # print(dft_arch_list)
        return


if __name__ == "__main__":
    # extractor = DftExtractor("tmp/KR835_2023286235715.DFT", True, True)
    # extractor.extract()
    with open("tmp/KR835_2023286235715.DFT", "rb") as file:
        block = file.read(4096)  # Read the first 16 bytes
        import chardet

        encoding = chardet.detect(block)["encoding"]
        # print("First 16 bytes:", block)
        # print("Hexadecimal:", block.hex())
        print(encoding)
        # record_type = struct.unpack('>H', block[4:6])[0]
        # print(record_type)
