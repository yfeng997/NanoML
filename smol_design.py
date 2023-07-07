import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import tiktoken

DEFAULT_MODEL = "gpt-3.5-turbo"


def summarize(filepath):
    with open(filepath, "r") as f:
        code = f.read()

    # HACK: 15000 roughly gives 3600 tokens
    if report_tokens(code) > 3600:
        code = code[:15000]

    llm = ChatOpenAI(temperature=0.5, model_name=DEFAULT_MODEL)
    prompt = PromptTemplate(
        input_variables=["code"],
        template="""
Give one line summary of below with key functionality and components. Limit to 20 words.
{code}
""",
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    try:
        s = chain.run(code)
    except Exception as e:
        s = str(e)[-200:]

    return s.strip()


def traverse_summarize(root):
    summary = {}

    for path, dirs, files in os.walk(root):
        relpath = os.path.relpath(path, root)
        if relpath == ".":
            path_names = []
        else:
            path_names = relpath.split("/")

        curr = summary
        for p in path_names:
            next_curr = curr.get(p, {})
            curr[p] = next_curr
            curr = next_curr

        for f in files:
            if is_valid_file(f):
                curr[f] = summarize(os.path.join(path, f))

    return summary


def is_valid_file(filename):
    return (
        filename.endswith(".py")
        or filename.endswith(".ts")
        or filename.endswith(".js")
        or filename.endswith(".tsx")
    )


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def report_tokens(s):
    encoding = tiktoken.encoding_for_model(DEFAULT_MODEL)
    return len(encoding.encode(s))


def report_tokens_file(path):
    with open(path, "r") as f:
        s = f.read()
    return report_tokens(s)


def report_tokens_folder(path):
    total_tokens = 0
    if os.path.isdir(path):
        for p, dirs, files in os.walk(path):
            for f in files:
                if is_valid_file(f):
                    total_tokens += report_tokens_file(os.path.join(p, f))
    else:
        total_tokens += report_tokens_file(path)
    return total_tokens


if __name__ == "__main__":
    # d = traverse_summarize("/Users/yuansongfeng/Desktop/dev/civitai")
    # save_json(d, "summary.json")

    print(report_tokens_folder("/Users/yuansongfeng/Desktop/dev/SimpML/summary.json"))

    # with open(
    #     "/Users/yuansongfeng/Desktop/dev/civitai/src/server/services/tag.service.ts",
    #     "r",
    # ) as f:
    #     s = f.read()
    #     print(report_tokens(s[:15000]))
