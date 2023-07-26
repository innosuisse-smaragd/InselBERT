class Tag:
    def __init__(self, start: int, end: int, tag: str):
        self.start = start
        self.end = end
        self.tag = tag


class Offset_Entry:
    def __init__(
        self,
        id: str,
        fact_tags: list[dict],
        anchor_tags: list[dict],
        modifier_tags: list[dict],
        text: str,
    ):
        self.id = id
        self.fact_tags = fact_tags
        self.anchor_tags = anchor_tags
        self.modifier_tags = modifier_tags
        self.text = text

    def to_dict(self):
        return {
            "id": self.id,
            "fact_tags": self.fact_tags,
            "anchor_tags": self.anchor_tags,
            "modifier_tags": self.modifier_tags,
            "text": self.text,
        }
