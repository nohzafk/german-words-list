import os
import pathlib
import pprint
import time

import backoff
import openai
from openai import OpenAI
from tqdm import tqdm

# (input tokens cost, output tokens cost)
GPT_API_PRICES = {
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-4-1106-preview": (0.01, 0.03),
}
USD_EXCHANGE_RATE = 7.3

model = "gpt-3.5-turbo-1106"


def calculate_gpt_cost(usage):
    """
    Calculate the cost of using GPT-3.5 based on token usage and specific token prices.

    Args:
    - response (dict): The response from the GPT-3.5 API.

    Returns:
    - cost (float): The total cost for the API usage.
    """
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # Given prices per 1K tokens
    price_per_prompt_token = GPT_API_PRICES[model][0] / 1000
    price_per_completion_token = GPT_API_PRICES[model][1] / 1000

    cost_prompt = prompt_tokens * price_per_prompt_token
    cost_completion = completion_tokens * price_per_completion_token

    return (cost_prompt + cost_completion) * USD_EXCHANGE_RATE


WORD_PROMPT = """Act as an assistant for learning German vocabulary to provide a comprehensive analysis of the word user inputed.
Avoid any comments, focusing solely on delivering clear, informative responses to enhance German language learning.
Use Markdown syntax, refer to this format to provide a thorough understanding of the inquired German word, aiding in learning and retention of the word and related vocabulary:
# <word>
## Meaning and Usage
<Explanation of the word's meaning and common usage scenarios.>
## Linguistic Analysis
<Analysis of the word's structure including any prefix, root, and suffix, and its etymology if applicable.>
## Comparisons between German and English
<Highlighting any similarities or differences between the German word and its English counterpart.>
## Cultural Context
<Any pertinent cultural context related to the word, if applicable.>
## Example Sentences
<Providing sentences to illustrate the word's usage, include the english translation.>
## Memory Tips
<Suggestions for remembering the word's meaning and usage.>
## Additional Vocabulary
<Providing related words, synonyms, or antonyms to broaden vocabulary knowledge.>
## Gender and Plural (for nouns)
<Indicating the gender of the noun and its plural form, if applicable.>
## Conjugation (for verbs)
<Present tense conjugation, if applicable.>
"""

SUMMARY_MD_PATH = pathlib.Path("../src/SUMMARY.md")


def backoff_hdlr(details):
    pprint.pprint(details)


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
    on_backoff=backoff_hdlr,
)
def generate_word(word: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": WORD_PROMPT},
            {"role": "user", "content": word},
        ],
        max_tokens=500,
        timeout=30,
    )

    return completion.choices[0].message.content, calculate_gpt_cost(completion.usage)


def normalize_word(word: str):
    """GOETHE-ZERTIFIKAT WORTLISTE"""
    word = word.strip()
    if "," in word:
        word = word.split(",")[0]
    if "/" in word:
        word = word.split("/")[0]

    if word.startswith("(sich)"):
        word = word[len("(sich)") :].strip()
    if word.startswith(("der ", "das ", "die ")):
        word = word[4:].strip()
    if word.endswith("(pl.)"):
        word = word[: -len("(pl.)")]

    return word


def generate_section(
    f, section_title: str, section_file: str, words: list[str], normalize: bool = False
):
    f.write(f"\n\n- [{section_title}]({section_file})\n")

    costs = 0
    pbar = tqdm(total=len(words))
    for word in words:
        chapter_name = word
        if normalize:
            chapter_name = normalize_word(chapter_name)
        chapter_name = f"{chapter_name}.md".replace(" ", "_")

        path = pathlib.Path(f"../src/words") / chapter_name
        if path.exists():
            f.write(f"    - [{word}](words/{chapter_name})\n")
            pbar.write(f"skip {word}")
            pbar.update(1)
            continue

        content, gpt_cost = generate_word(word)
        costs += gpt_cost

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        with open(path, "w") as fpath:
            fpath.write(content)

        f.write(f"    - [{word}](words/{chapter_name})\n")
        pbar.write(f"write {word}, Total {costs=} Yuan")
        pbar.update(1)

        time.sleep(0.2)

    pbar.close()


def read_words_file(file_path: str):
    words = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            words.append(line.strip())
    return words


def generate_book():
    with open(SUMMARY_MD_PATH, "w") as f:
        f.write("# Summary\n[Introduction](README.md)\n\n")
        f.write("# Word Lists\n")

        # A1 words
        word_list_a1 = read_words_file("./A1.txt")
        generate_section(
            f, "German Vocabulary A1", "A1.md", word_list_a1, normalize=True
        )

        # Top 500 German words
        word_list_500 = read_words_file("./500-words.txt")
        generate_section(f, "Top 500 German words", "top_500.md", word_list_500)

        # 1000 most common German words
        word_list_1k = read_words_file("./1k-words.txt")
        generate_section(
            f, "1000 most commonly spoken German words", "1000_common.md", word_list_1k
        )


if __name__ == "__main__":
    generate_book()
