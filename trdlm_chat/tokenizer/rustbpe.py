import copy
import logging
import os
import pickle
from functools import lru_cache
from typing import Any, Iterator, Self, Sequence, overload

from tiktoken import Encoding, get_encoding

import rustbpe
from trdlm_chat.utils.logger import LOGGER

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RustBPETokenizer:
    """
    Basically the same as Karpathy's RustBPETokenizer. I'm going to use it here.
    """

    def __init__(
        self, encoder: Encoding, bos_token: str, logger: logging.Logger = LOGGER
    ) -> None:
        self._encoder = encoder
        self._bos_token_id = self.encode_special(bos_token)
        self._logger = logger

    @classmethod
    def train_from_iterator(cls, text_iterator: Iterator[str], vocab_size: int) -> Self:
        """
        Initialize the tokenizer from a text corpus iterator.

        Args:
            text_iterator (Iterator[str]): An iterator over the text corpus.
            vocab_size (int): The desired vocabulary size.

        Returns:
            RustBPETokenizer: A new RustBPETokenizer instance.
        """
        tokenizer = rustbpe.Tokenizer()  # type: ignore
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        if vocab_size_no_special < 256:
            raise ValueError(
                f"Vocabulary size must be at least 256 without special tokens, got {vocab_size_no_special}"
            )
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> Self:
        """
        Initialize a tokenizer from a directory.

        Args:
            tokenizer_dir (str): The directory containing the tokenizer files.

        Returns:
            RustBPETokenizer: A new RustBPETokenizer instance.
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name: str) -> Self:
        enc = get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    @property
    def vocab_size(self) -> int:
        return self._encoder.n_vocab

    @property
    def special_tokens(self) -> set[str]:
        return self._encoder.special_tokens_set

    def id_to_token(self, id):
        return self._encoder.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self._encoder.encode_single_token(text)

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @overload
    def encode(
        self,
        text: str,
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[int]: ...

    @overload
    def encode(
        self,
        text: list[str],
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[list[int]]: ...

    def encode(
        self,
        text: str | list[str],
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:

        prepend_id = 0
        append_id = 0

        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self._encoder.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self._encoder.encode_ordinary_batch(
                list(text), num_threads=num_threads
            )
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._encoder.decode(ids)

    def save(self, tokenizer_dir: str) -> None:

        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")

        with open(pickle_path, "wb") as f:
            pickle.dump(self._encoder, f)

        self._logger.info(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(
        self, conversation: dict[str, Any], max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        """
        Creates a conversation for the chatbot to be trained on.

        Args:
            conversation (dict): A dictionary containing the conversation data.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 2048.

        Returns:
            tuple[list[int], list[int]]: A tuple containing the token ids and assistant mask.
        """
        ids, mask = [], []

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            if messages[1]["role"] != "user":
                raise RuntimeError("System message must be followed by a user message")
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        if len(messages) == 0:
            raise RuntimeError("Conversation has less than one message")

        bos = self.bos_token_id
        user_start, user_end = self.encode_special(
            "<|user_start|>"
        ), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special(
            "<|assistant_start|>"
        ), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special(
            "<|python_start|>"
        ), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special(
            "<|output_start|>"
        ), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            must_be_from = "user" if i % 2 == 0 else "assistant"
            if message["role"] != must_be_from:
                raise RuntimeError(
                    f"Message {i} is from {message['role']} but should be from {must_be_from}"
                )

            content = message["content"]

            if message["role"] == "user":
                if not isinstance(content, str):
                    raise RuntimeError(
                        "User messages are simply expected to be strings"
                    )
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(
        self, ids: list[int], mask: list[int], with_token_id: bool = False
    ) -> str:
        """
        Debugging function for rendering tokenization of a conversation.

        Args:
            ids (list[int]): A list of token ids.
            mask (list[int]): A list of mask values.
            with_token_id (bool, optional): Whether to include the token id in the output. Defaults to False.

        Returns:
            str: A string representation of the tokenization.
        """
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation: dict[str, Any]) -> list[int]:
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.

        Args:
            conversation (dict[str, Any]): A conversation dictionary.

        Returns:
            list[int]: A list of token ids.
        """
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]

        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message role must be the assistant")

        messages.pop()  # remove the last message (of the Assistant) inplace
        ids, _ = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids
