from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from app.core.models import PhotoFile
from app.xmp.writer import XMP_NS, XmpWriter, _find_standard_xmp_payload, _new_xmp_packet, _parse_xmp_payload


CFG = {
    "xmp": {
        "scheme":"bridge",
        "bridge_red":"Select",
        "bridge_yellow":"Second",
        "lightroom_red":"Red",
        "lightroom_yellow":"Yellow",
        "custom_red":"R",
        "custom_yellow":"Y",
        "jpeg_embedded": True,
        "update_existing_jpeg_sidecar": True,
    }
}


class XmpTests(unittest.TestCase):
    def test_label_write_sidecar_raw(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"x")
            p = PhotoFile(photo_path, datetime.now(), 1, ".cr3")
            xmp = XmpWriter(CFG).write(p, "red")
            root = ET.parse(xmp).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_preserves_existing_attribute(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"x")
            xmp_path = Path(td) / "A.xmp"
            xmp_path.write_text(
                '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"><rdf:RDF><rdf:Description rdf:about="" crs:Exposure2012="0.55"/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )
            p = PhotoFile(photo_path, datetime.now(), 1, ".cr3")
            XmpWriter(CFG).write(p, "red")
            root = ET.parse(xmp_path).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib["{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"], "0.55")
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_jpeg_label_is_embedded_and_image_remains_readable(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (64, 48), (120, 80, 30)).save(photo_path, "JPEG")
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            result = XmpWriter(CFG).write(p, "red")
            self.assertEqual(result, photo_path)
            self.assertFalse((Path(td) / "A.xmp").exists())

            payload = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertIsNotNone(payload)
            tree = _parse_xmp_payload(payload)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

            with Image.open(photo_path) as im:
                self.assertEqual(im.size, (64, 48))


    def test_jpeg_yellow_role_writes_yellow_label(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "Y.jpg"
            Image.new("RGB", (32, 32), (30, 40, 50)).save(photo_path, "JPEG")
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            XmpWriter(CFG).write(p, "yellow")
            payload = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertIsNotNone(payload)
            tree = _parse_xmp_payload(payload)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Second")

    def test_jpeg_existing_sidecar_is_updated_but_new_one_is_not_created(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(photo_path, "JPEG")
            sidecar = Path(td) / "A.xmp"
            sidecar.write_text(
                '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:RDF><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            XmpWriter(CFG).write(p, "red")
            root = ET.parse(sidecar).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_clear_red_label_preserves_other_sidecar_metadata(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"x")
            p = PhotoFile(photo_path, datetime.now(), 1, ".cr3")
            writer = XmpWriter(CFG)
            writer.write(p, "red")
            xmp_path = photo_path.with_suffix(".xmp")
            tree = ET.parse(xmp_path)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            desc.set("{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012", "0.25")
            tree.write(xmp_path, encoding="utf-8", xml_declaration=True)
            self.assertTrue(writer.clear_red_label(p))
            root = ET.parse(xmp_path).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertNotIn(f"{{{XMP_NS}}}Label", desc.attrib)
            self.assertEqual(desc.attrib["{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"], "0.25")

    def test_clear_red_label_from_embedded_jpeg(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (32, 32), (10, 20, 30)).save(photo_path, "JPEG")
            p = PhotoFile(photo_path, datetime.now(), 1, ".jpg")
            writer = XmpWriter(CFG)
            writer.write(p, "red")
            self.assertTrue(writer.clear_red_label(p))
            payload = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertIsNotNone(payload)
            tree = _parse_xmp_payload(payload)
            desc = next(tree.getroot().iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertNotIn(f"{{{XMP_NS}}}Label", desc.attrib)
            with Image.open(photo_path) as im:
                self.assertEqual(im.size, (32, 32))


if __name__ == "__main__":
    unittest.main()


class XmpPreservationTests(unittest.TestCase):
    def _photo(self, path: Path, extension: str) -> PhotoFile:
        return PhotoFile(path, datetime.now(), 1, extension)

    def test_existing_sidecar_is_surgically_updated(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"raw")
            sidecar = photo_path.with_suffix(".xmp")
            original = (
                b'<?xpacket begin="\xef\xbb\xbf" id="abc"?>\r\n'
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
                b'xmlns:dc="http://purl.org/dc/elements/1.1/">\r\n'
                b'<!-- keep this comment exactly -->\r\n'
                b'<rdf:RDF><rdf:Description rdf:about="" crs:Exposure2012="+0.55" crs:CropTop="0.123" '
                b'xmp:Rating="4" xmp:Label="Old"><dc:subject><rdf:Bag><rdf:li>School</rdf:li>'
                b'</rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>\r\n'
                b'<?xpacket end="w"?>'
            )
            sidecar.write_bytes(original)
            XmpWriter(CFG).write(self._photo(photo_path, ".cr3"), "red")
            updated = sidecar.read_bytes()
            self.assertEqual(updated, original.replace(b'xmp:Label="Old"', b'xmp:Label="Select"'))

    def test_clear_sidecar_removes_only_matching_label_bytes(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.NEF"
            photo_path.write_bytes(b"raw")
            sidecar = photo_path.with_suffix(".xmp")
            original = (
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
                b'<rdf:RDF><rdf:Description rdf:about="" crs:Contrast2012="17" xmp:Rating="5" xmp:Label="Select"/>'
                b'</rdf:RDF></x:xmpmeta>'
            )
            sidecar.write_bytes(original)
            writer = XmpWriter(CFG)
            self.assertTrue(writer.clear_label(self._photo(photo_path, ".nef"), "red"))
            updated = sidecar.read_bytes()
            self.assertEqual(updated, original.replace(b' xmp:Label="Select"', b""))
            self.assertIn(b'crs:Contrast2012="17"', updated)
            self.assertIn(b'xmp:Rating="5"', updated)

    def test_clear_does_not_touch_a_different_manual_label(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR3"
            photo_path.write_bytes(b"raw")
            sidecar = photo_path.with_suffix(".xmp")
            original = (
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF>'
                b'<rdf:Description rdf:about="" xmp:Label="Green" xmp:Rating="3"/>'
                b'</rdf:RDF></x:xmpmeta>'
            )
            sidecar.write_bytes(original)
            writer = XmpWriter(CFG)
            self.assertFalse(writer.clear_label(self._photo(photo_path, ".cr3"), "red"))
            self.assertEqual(sidecar.read_bytes(), original)

    def test_embedded_jpeg_preserves_non_label_xmp_verbatim(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.jpg"
            Image.new("RGB", (48, 32), (12, 34, 56)).save(photo_path, "JPEG")
            writer = XmpWriter(CFG)
            p = self._photo(photo_path, ".jpg")
            writer.write(p, "red")
            data = photo_path.read_bytes()
            payload = _find_standard_xmp_payload(data)
            self.assertIsNotNone(payload)
            # Add Camera Raw settings and a rating without using the writer.
            payload2 = payload.replace(
                b'xmp:Label="Select"',
                b'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
                b'crs:Exposure2012="0.80" xmp:Rating="4" xmp:Label="Select"',
            )
            from app.xmp.writer import _jpeg_xmp_segment, _replace_or_insert_standard_xmp
            photo_path.write_bytes(_replace_or_insert_standard_xmp(data, _jpeg_xmp_segment(payload2, photo_path)))
            before = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertTrue(writer.clear_label(p, "red"))
            after = _find_standard_xmp_payload(photo_path.read_bytes())
            self.assertEqual(after, before.replace(b' xmp:Label="Select"', b""))
            self.assertIn(b'crs:Exposure2012="0.80"', after)
            self.assertIn(b'xmp:Rating="4"', after)

    @staticmethod
    def _classic_tiff_with_xmp(packet: bytes) -> bytes:
        # Minimal little-endian TIFF container with one IFD entry: tag 700 XMP.
        ifd_offset = 8
        data_offset = 8 + 2 + 12 + 4
        entry = (
            (700).to_bytes(2, "little")
            + (1).to_bytes(2, "little")
            + len(packet).to_bytes(4, "little")
            + data_offset.to_bytes(4, "little")
        )
        return b"II*\x00" + ifd_offset.to_bytes(4, "little") + b"\x01\x00" + entry + b"\x00\x00\x00\x00" + packet

    def test_embedded_dng_xmp_clear_preserves_file_size_and_other_fields(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.dng"
            packet = (
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
                b'<rdf:RDF><rdf:Description crs:Exposure2012="1.10" xmp:Rating="5" xmp:Label="Select"/>'
                b'</rdf:RDF></x:xmpmeta>' + b" " * 128
            )
            photo_path.write_bytes(self._classic_tiff_with_xmp(packet))
            before_size = photo_path.stat().st_size
            writer = XmpWriter(CFG)
            self.assertTrue(writer.clear_label(self._photo(photo_path, ".dng"), "red"))
            self.assertEqual(photo_path.stat().st_size, before_size)
            from app.xmp.writer import _find_tiff_xmp_region
            offset, size = _find_tiff_xmp_region(photo_path)
            with photo_path.open("rb") as fh:
                fh.seek(offset)
                after = fh.read(size)
            self.assertNotIn(b'xmp:Label="Select"', after)
            self.assertIn(b'crs:Exposure2012="1.10"', after)
            self.assertIn(b'xmp:Rating="5"', after)

    def test_embedded_dng_existing_xmp_label_is_updated_in_place(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.dng"
            packet = (
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
                b'<rdf:RDF><rdf:Description crs:Texture="22" xmp:Label="Second"/></rdf:RDF></x:xmpmeta>'
                + b" " * 256
            )
            original_file = self._classic_tiff_with_xmp(packet)
            photo_path.write_bytes(original_file)
            writer = XmpWriter(CFG)
            result = writer.write(self._photo(photo_path, ".dng"), "red")
            self.assertEqual(result, photo_path)
            self.assertEqual(photo_path.stat().st_size, len(original_file))
            from app.xmp.writer import _find_tiff_xmp_region
            offset, size = _find_tiff_xmp_region(photo_path)
            with photo_path.open("rb") as fh:
                fh.seek(offset)
                after = fh.read(size)
            self.assertIn(b'xmp:Label="Select"', after)
            self.assertNotIn(b'xmp:Label="Second"', after)
            self.assertIn(b'crs:Texture="22"', after)
            self.assertFalse(photo_path.with_suffix(".xmp").exists())

    def test_embedded_dng_without_room_falls_back_to_sidecar(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.dng"
            packet = (
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                b'<rdf:RDF><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>'
            )
            original = self._classic_tiff_with_xmp(packet)
            photo_path.write_bytes(original)
            result = XmpWriter(CFG).write(self._photo(photo_path, ".dng"), "red")
            self.assertEqual(result, photo_path.with_suffix(".xmp"))
            self.assertEqual(photo_path.read_bytes(), original)
            self.assertTrue(photo_path.with_suffix(".xmp").exists())

    def test_tiff_based_proprietary_raw_is_never_modified_internally(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.CR2"
            original_raw = self._classic_tiff_with_xmp(_new_xmp_packet("Old"))
            photo_path.write_bytes(original_raw)

            result = XmpWriter(CFG).write(self._photo(photo_path, ".cr2"), "red")

            self.assertEqual(result, photo_path.with_suffix(".xmp"))
            self.assertEqual(photo_path.read_bytes(), original_raw)
            root = ET.parse(result).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertEqual(desc.attrib[f"{{{XMP_NS}}}Label"], "Select")

    def test_clear_proprietary_raw_changes_only_sidecar_label(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.NEF"
            original_raw = self._classic_tiff_with_xmp(_new_xmp_packet("Select"))
            photo_path.write_bytes(original_raw)
            sidecar = photo_path.with_suffix(".xmp")
            sidecar.write_text(
                '<?xml version="1.0"?><x:xmpmeta xmlns:x="adobe:ns:meta/" '
                'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
                'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
                '<rdf:RDF><rdf:Description rdf:about="" xmp:Label="Select" '
                'crs:Exposure2012="0.55"/></rdf:RDF></x:xmpmeta>',
                encoding="utf-8",
            )

            changed = XmpWriter(CFG).clear_label(self._photo(photo_path, ".nef"), "red")

            self.assertTrue(changed)
            self.assertEqual(photo_path.read_bytes(), original_raw)
            self.assertTrue(sidecar.exists())
            root = ET.parse(sidecar).getroot()
            desc = next(root.iter("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"))
            self.assertNotIn(f"{{{XMP_NS}}}Label", desc.attrib)
            self.assertEqual(
                desc.attrib["{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012"], "0.55"
            )

    @staticmethod
    def _minimal_psd_with_xmp(packet: bytes) -> bytes:
        header = (
            b"8BPS" + (1).to_bytes(2, "big") + b"\x00" * 6
            + (3).to_bytes(2, "big") + (1).to_bytes(4, "big")
            + (1).to_bytes(4, "big") + (8).to_bytes(2, "big")
            + (3).to_bytes(2, "big")
        )
        name = b"\x00\x00"  # empty Pascal string + even padding
        resource = (
            b"8BIM" + (1060).to_bytes(2, "big") + name
            + len(packet).to_bytes(4, "big") + packet
            + (b"\x00" if len(packet) % 2 else b"")
        )
        # Color-mode data length 0; then image-resource section; trailing bytes
        # stand in for later PSD sections and must remain untouched.
        return header + (0).to_bytes(4, "big") + len(resource).to_bytes(4, "big") + resource + b"TRAILING-PSD-DATA"

    def test_embedded_psd_xmp_clear_keeps_other_metadata_and_container_size(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.psd"
            packet = (
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">'
                b'<rdf:RDF><rdf:Description photoshop:Headline="Class 4A" xmp:Rating="2" xmp:Label="Select"/>'
                b'</rdf:RDF></x:xmpmeta>' + b" " * 96
            )
            original = self._minimal_psd_with_xmp(packet)
            photo_path.write_bytes(original)
            writer = XmpWriter(CFG)
            self.assertTrue(writer.clear_label(self._photo(photo_path, ".psd"), "red"))
            updated_file = photo_path.read_bytes()
            self.assertEqual(len(updated_file), len(original))
            self.assertTrue(updated_file.endswith(b"TRAILING-PSD-DATA"))
            from app.xmp.writer import _find_psd_xmp_region
            offset, size = _find_psd_xmp_region(photo_path)
            after = updated_file[offset:offset + size]
            self.assertNotIn(b'xmp:Label="Select"', after)
            self.assertIn(b'photoshop:Headline="Class 4A"', after)
            self.assertIn(b'xmp:Rating="2"', after)

    def test_embedded_dng_can_add_label_using_existing_padding(self):
        with TemporaryDirectory() as td:
            photo_path = Path(td) / "A.dng"
            packet = (
                b'<?xpacket begin="\xef\xbb\xbf" id="a"?>\n'
                b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                b'xmlns:xmp="http://ns.adobe.com/xap/1.0/"><rdf:RDF>'
                b'<rdf:Description rdf:about="" xmp:Rating="5"/></rdf:RDF></x:xmpmeta>'
                + b" " * 160 + b'<?xpacket end="w"?>'
            )
            original = self._classic_tiff_with_xmp(packet)
            photo_path.write_bytes(original)
            result = XmpWriter(CFG).write(self._photo(photo_path, ".dng"), "red")
            self.assertEqual(result, photo_path)
            self.assertEqual(photo_path.stat().st_size, len(original))
            from app.xmp.writer import _find_tiff_xmp_region
            offset, size = _find_tiff_xmp_region(photo_path)
            data = photo_path.read_bytes()[offset:offset + size]
            self.assertIn(b'xmp:Label="Select"', data)
            self.assertIn(b'xmp:Rating="5"', data)
            self.assertFalse(photo_path.with_suffix(".xmp").exists())
