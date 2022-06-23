import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx6 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to point cloud, we also store the corresponding point cloud dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, mode="cd"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 6:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 6, got {}".format(bbox.size(-1))
            )
        if mode not in ("cd"):
            raise ValueError("mode should be 'cd'")

        self.bbox = bbox
        self.mode = mode
        self.extra_fields = {}

    """
    add extra information to Box
    """
    def add_field(self, field, field_data):
        assert len(field_data) == len(self.bbox)
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("cd"):
            raise ValueError("mode should be 'cd'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xc, yc, zc, xd, yd, zd = self._split_into_xyxy()
        if mode == "cd":
            bbox = torch.cat((xc, yc, zc, xd, yd, zd), dim=-1)
            bbox = BoxList(bbox, mode=mode)
        else:
            raise ValueError("mode should be 'cd'")
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_cd(self):
        if self.mode == "cd":
            xc, yc, zc, xd, yd, zd = self.bbox.split(1, dim=-1)
            return xc, yc, zc, xd, yd, zd

        else:
            raise RuntimeError("Should not be here")

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def area(self):
        box = self.bbox
        if self.mode == "cd":
            bomin = box[:, :3] - 1 / 2 * box[:, 3:6]
            bomax = box[:, :3] + 1 / 2 * box[:, 3:6]
            area = (bomax[:, 0] - bomin[:, 0]) * (bomax[:, 1] - bomin[:, 1]) * (bomax[:, 2] - bomin[:, 2])

        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
