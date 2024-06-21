from datasets import load_dataset

dataset = load_dataset("EleutherAI/truthful_qa_mc", split="validation")

print(dataset)

INPUT_TEMPLATE = """\
{question}
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}"""

def preprocess_fn(data_point):

    choices = data_point["choices"]

    data_point["input"] = INPUT_TEMPLATE.format(
        question=data_point["question"],
        choice_a=choices[0],
        choice_b=choices[1],
        choice_c=choices[2],
        choice_d=choices[3],
    )

    return data_point

dataset = dataset.map(preprocess_fn, load_from_cache_file=False)

print(dataset[0])

dataset.save_to_disk("./caches/datasets/truthful_qa_mc/")