class Dataset_Entry:
    def __init__(
        self,
        id: str,
        fact_tags: list[list[str]],
        anchor_tags: list[list[str]],
        modifier_tags: list[list[str]],
        tokens: list[str],
    ):
        self.id = id
        self.fact_tags = fact_tags
        self.anchor_tags = anchor_tags
        self.modifier_tags = modifier_tags
        self.tokens = tokens
