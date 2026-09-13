import json
from pathlib import Path


class ByteTokenizer:
    """UTF-8 byte tokenizer: compact, deterministic, and dependency-free."""

    SPECIAL = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<system>": 3,
        "<user>": 4,
        "<assistant>": 5,
    }
    BYTE_OFFSET = len(SPECIAL)
    vocab_size = 256 + BYTE_OFFSET

    def encode(self, text: str) -> list[int]:
        return [byte + self.BYTE_OFFSET for byte in text.encode("utf-8")]

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        special_ids = set(self.SPECIAL.values())
        values = [
            token_id - self.BYTE_OFFSET
            for token_id in token_ids
            if token_id >= self.BYTE_OFFSET and (not skip_special or token_id not in special_ids)
        ]
        return bytes(value for value in values if 0 <= value <= 255).decode("utf-8", errors="replace")

    def format_chat(self, system: str, user: str) -> list[int]:
        return (
            [self.SPECIAL["<bos>"], self.SPECIAL["<system>"]]
            + self.encode(system.strip() + "\n")
            + [self.SPECIAL["<user>"]]
            + self.encode(user.strip() + "\n")
            + [self.SPECIAL["<assistant>"]]
        )

    def training_example(self, system: str, user: str, assistant: str) -> tuple[list[int], int]:
        prompt = self.format_chat(system, user)
        return prompt + self.encode(assistant.strip()) + [self.SPECIAL["<eos>"]], len(prompt)

    def inspect(self, text: str) -> list[dict]:
        result = []
        for index, byte in enumerate(text.encode("utf-8")):
            result.append({
                "index": index,
                "token_id": byte + self.BYTE_OFFSET,
                "byte": byte,
                "piece": bytes([byte]).decode("utf-8", errors="replace"),
            })
        return result

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "utf8-byte", "special_tokens": self.SPECIAL}, indent=2))

