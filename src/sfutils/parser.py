
import struct


def read_chunk_header(f):
    """
    Reads a RIFF chunk header (ID and size) from a file.

    Args:
        f: The file object.

    Returns:
        A tuple containing the chunk ID and chunk size.
    """
    chunk_id = f.read(4)
    if len(chunk_id) < 4:
        raise EOFError("Unexpected end of file while reading chunk ID.")

    chunk_size_bytes = f.read(4)
    if len(chunk_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading chunk size.")

    chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
    return chunk_id, chunk_size


class SF2Parser:
    """
    A parser for SF2 files.
    """

    def __init__(self, filepath):
        """
        Initializes the SF2Parser.

        Args:
            filepath: The path to the SF2 file.
        """
        self.filepath = filepath
        self.file = None
        self.info_data = {}
        self.sample_data = b""
        self.sample_data_24 = b""
        self.pdta = {}

    def parse(self):
        """
        Parses the entire SF2 file.
        """
        with open(self.filepath, "rb") as f:
            self.file = f

            # RIFF header
            riff_id = f.read(4)
            if riff_id != b"RIFF":
                raise ValueError("Not a RIFF file")

            file_size = struct.unpack("<I", f.read(4))[0]

            form_type = f.read(4)
            if form_type != b"sfbk":
                raise ValueError("Not a SoundFont file")

            # Parse the three main chunks
            self._parse_chunks()

    def _parse_chunks(self):
        """
        Parses INFO, sdta, and pdta chunks.
        """
        while True:
            try:
                chunk_id, chunk_size = read_chunk_header(self.file)
                if chunk_id == b"LIST":
                    list_type = self.file.read(4)
                    if list_type == b"INFO":
                        self._parse_info_list(chunk_size - 4)
                    elif list_type == b"sdta":
                        self._parse_sdta_list(chunk_size - 4)
                    elif list_type == b"pdta":
                        self._parse_pdta_list(chunk_size - 4)
                    else:
                        self.file.seek(chunk_size - 4, 1)  # Skip unknown list
                else:
                    self.file.seek(chunk_size, 1)  # Skip unknown chunk

                # Align to next word
                if chunk_size % 2:
                    self.file.seek(1, 1)

            except EOFError:
                break

    def _parse_info_list(self, size):
        """
        Parses the INFO-list chunk.
        """
        chunk_end = self.file.tell() + size
        while self.file.tell() < chunk_end:
            sub_id, sub_size = read_chunk_header(self.file)
            data = self.file.read(sub_size)

            # Handle padding
            if sub_size % 2:
                self.file.read(1)

            self._store_info_data(sub_id, data)

    def _store_info_data(self, sub_id, data):
        """
        Stores INFO data.
        """
        if sub_id == b"ifil":
            # Version info
            major, minor = struct.unpack("<HH", data)
            self.info_data["version"] = f"{major}.{minor:02d}"
        else:
            key = sub_id.decode("ascii", errors="ignore").rstrip("\x00")
            value = data.decode("ascii", errors="ignore").rstrip("\x00")
            if key == "isng":
                self.info_data["sound_engine"] = value
            elif key == "INAM":
                self.info_data["bank_name"] = value
            elif key == "ICOP":
                self.info_data["copyright"] = value
            elif key == "ICMT":
                self.info_data["comment"] = value
            elif key == "ISFT":
                self.info_data["software"] = value
            elif key == "ICRD":
                self.info_data["creation_date"] = value
            elif key == "IENG":
                self.info_data["engineer"] = value
            elif key == "IPRD":
                self.info_data["product"] = value
            else:
                # Other INFO chunks
                self.info_data[key.lower()] = value

    def _parse_sdta_list(self, size):
        """
        Parses the sdta-list chunk.
        """
        chunk_end = self.file.tell() + size
        while self.file.tell() < chunk_end:
            sub_id, sub_size = read_chunk_header(self.file)
            if sub_id == b"smpl":
                # 16-bit sample data
                self.sample_data = self.file.read(sub_size)
            elif sub_id == b"sm24":
                # 24-bit LSB sample data
                self.sample_data_24 = self.file.read(sub_size)
            else:
                # Skip unknown chunks
                self.file.seek(sub_size, 1)

            # Handle padding
            if sub_size % 2:
                self.file.read(1)

    def _parse_pdta_list(self, size):
        """
        Parses the pdta-list (Hydra) chunk.
        """
        chunk_end = self.file.tell() + size
        # Read the 9 sub-chunks in order
        hydra_chunks = {"phdr", "pbag", "pmod", "pgen", "inst", "ibag", "imod", "igen", "shdr"}
        while self.file.tell() < chunk_end:
            sub_id, sub_size = read_chunk_header(self.file)
            sub_id_str = sub_id.decode("ascii", errors="ignore")
            if sub_id_str in hydra_chunks:
                # Store data
                self.pdta[sub_id_str] = self.file.read(sub_size)
            else:
                # Skip unknown chunks
                self.file.seek(sub_size, 1)

            # Handle padding
            if sub_size % 2:
                self.file.read(1)

    def _get_pdta_records(self, chunk_name, record_size, terminator):
        """
        Gets records from a pdta sub-chunk.
        """
        data = self.pdta.get(chunk_name, b"")
        records = []
        for i in range(0, len(data), record_size):
            if i + record_size > len(data):
                break
            chunk = data[i:i + record_size]
            name_bytes = chunk[0:20]
            # Skip terminator record
            if name_bytes.startswith(terminator):
                break
            records.append(chunk)
        return records

    def get_preset_headers(self):
        """
        Gets preset headers.
        """
        # sfPresetHeader = 38 bytes
        records = self._get_pdta_records("phdr", 38, b"EOP")
        headers = []
        for r in records:
            name = r[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            values = struct.unpack("<HHHIII", r[20:38])
            headers.append({
                "name": name,
                "preset": values[0],
                "bank": values[1],
                "bag_ndx": values[2],
                "library": values[3],
                "genre": values[4],
                "morphology": values[5]
            })
        return headers

    def get_instrument_headers(self):
        """
        Gets instrument headers.
        """
        # sfInst = 22 bytes
        records = self._get_pdta_records("inst", 22, b"EOI")
        headers = []
        for r in records:
            name = r[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            bag_ndx = struct.unpack("<H", r[20:22])[0]
            headers.append({"name": name, "bag_ndx": bag_ndx})
        return headers

    def get_sample_headers(self):
        """
        Gets sample headers.
        """
        # sfSample = 46 bytes
        records = self._get_pdta_records("shdr", 46, b"EOS")
        headers = []
        for r in records:
            name = r[0:20].decode("ascii", errors="ignore").rstrip("\x00")
            values = struct.unpack("<IIIIIBbHH", r[20:46])
            headers.append({
                "name": name,
                "start": values[0],
                "end": values[1],
                "start_loop": values[2],
                "end_loop": values[3],
                "sample_rate": values[4],
                "original_key": values[5],
                "correction": values[6],
                "sample_link": values[7],
                "sample_type": values[8]
            })
        return headers

    def _get_bags(self, chunk_name):
        """
        Gets bags (zones).
        """
        data = self.pdta.get(chunk_name, b"")
        bags = []
        # sfPresetBag / sfInstBag = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break
            gen_ndx, mod_ndx = struct.unpack("<HH", data[i:i + 4])
            bags.append({"gen_ndx": gen_ndx, "mod_ndx": mod_ndx})
        return bags

    def get_preset_bags(self):
        return self._get_bags("pbag")

    def get_instrument_bags(self):
        return self._get_bags("ibag")

    def _get_generators(self, chunk_name):
        """
        Gets generators.
        """
        data = self.pdta.get(chunk_name, b"")
        generators = []
        # sfGenList = 4 bytes
        for i in range(0, len(data), 4):
            if i + 4 > len(data):
                break
            oper, amount = struct.unpack("<Hh", data[i:i + 4])
            generators.append({"oper": oper, "amount": amount})
        return generators

    def get_preset_generators(self):
        return self._get_generators("pgen")

    def get_instrument_generators(self):
        return self._get_generators("igen")

    def _get_modulators(self, chunk_name):
        """
        Gets modulators.
        """
        data = self.pdta.get(chunk_name, b"")
        modulators = []
        # sfModList = 10 bytes
        for i in range(0, len(data), 10):
            if i + 10 > len(data):
                break
            values = struct.unpack("<HHhHH", data[i:i + 10])
            modulators.append({
                "src_oper": values[0],
                "dest_oper": values[1],
                "amount": values[2],
                "amt_src_oper": values[3],
                "trans_oper": values[4]
            })
        return modulators

    def get_preset_modulators(self):
        return self._get_modulators("pmod")

    def get_instrument_modulators(self):
        return self._get_modulators("imod")
