"""Minimal TFRecord + TensorFlow tf.Example reader, without TensorFlow."""

import struct
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _build_example_class():
    proto = descriptor_pb2.FileDescriptorProto()
    proto.name = "minimal_tensorflow_example.proto"
    proto.package = "tensorflow"
    proto.syntax = "proto3"

    def add_msg(name):
        msg = proto.message_type.add()
        msg.name = name
        return msg

    def field(parent, name, number, typ, *, repeated=False, type_name=None):
        f = parent.field.add()
        f.name, f.number, f.type = name, number, typ
        f.label = 3 if repeated else 1
        if type_name:
            f.type_name = type_name
        return f

    for name, datatype in (("BytesList", 12), ("FloatList", 2), ("Int64List", 3)):
        m = add_msg(name)
        f = field(m, "value", 1, datatype, repeated=True)
        if name != "BytesList":
            f.options.packed = True
    m = add_msg("Feature")
    for number, name, cls in ((1, "bytes_list", "BytesList"),
                              (2, "float_list", "FloatList"),
                              (3, "int64_list", "Int64List")):
        field(m, name, number, 11, type_name=f".tensorflow.{cls}")
    m = add_msg("Features")
    entry = m.nested_type.add()
    entry.name, entry.options.map_entry = "FeatureEntry", True
    field(entry, "key", 1, 9)
    field(entry, "value", 2, 11, type_name=".tensorflow.Feature")
    field(m, "feature", 1, 11, repeated=True,
          type_name=".tensorflow.Features.FeatureEntry")
    m = add_msg("Example")
    field(m, "features", 1, 11, type_name=".tensorflow.Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.Example"))


Example = _build_example_class()


def _crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return (~crc) & 0xFFFFFFFF


def _masked_crc(data):
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord(path, verify_crc=True, max_record_bytes=100_000_000):
    with open(path, "rb") as stream:
        while True:
            header = stream.read(12)
            if not header:
                break
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header in {path}")
            size, head_crc = struct.unpack("<QI", header)
            if size > max_record_bytes:
                raise ValueError(f"TFRecord payload {size} exceeds {max_record_bytes}")
            if verify_crc and _masked_crc(header[:8]) != head_crc:
                raise ValueError("Invalid TFRecord length CRC")
            payload = stream.read(size)
            tail = stream.read(4)
            if len(payload) != size or len(tail) != 4:
                raise ValueError("Truncated TFRecord payload")
            if verify_crc and _masked_crc(payload) != struct.unpack("<I", tail)[0]:
                raise ValueError("Invalid TFRecord payload CRC")
            yield payload


def serialize_tfrecord(messages, path):
    with open(path, "wb") as out:
        for message in messages:
            content = message.SerializeToString() if hasattr(message, "SerializeToString") else message
            header = struct.pack("<Q", len(content))
            out.write(header + struct.pack("<I", _masked_crc(header)))
            out.write(content + struct.pack("<I", _masked_crc(content)))


def read_example(payload):
    message = Example()
    message.ParseFromString(payload)
    output = {}
    for key, feature in message.features.feature.items():
        if feature.HasField("float_list"):
            output[key] = list(feature.float_list.value)
        elif feature.HasField("int64_list"):
            output[key] = list(feature.int64_list.value)
        elif feature.HasField("bytes_list"):
            output[key] = list(feature.bytes_list.value)
    return output
