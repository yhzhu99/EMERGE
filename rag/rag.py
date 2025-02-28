import pandas as pd
import pdb


def peek_data(input_file):
    data = pd.read_pickle(input_file)
    for item in data:
        entities = item['Entities']
        docs = item['Documents']
        triples = item['Triples']
        prompt = summary_template.replace("{entities}", entities).replace("{documents}", docs).replace("{triples}", triples)
        prompt = prompt.strip()
        pdb.set_trace()


summary_template = """
[Instruction]
As an experienced clinical professor, you have been provided with a list of diseases a patient is suffering from, alongside related documents and knowledge graph triples specific to these diseases, which should facilitate your understanding. Please endeavor to summarize the patient's health status and formulate a brief and clear summarization within 2048 tokens. Interpretation or explanation of the given terms is not required. Your summary should be helpful for healthcare prediction tasks (e.g., in-hospital mortality prediction, 30-day readmission prediction).
[Diseases]
{entities}
[Documents]
{documents}
[Related knowledge graph triples]
{triples}
"""


if __name__ == "__main__":
    peek_data('./ts_note_final.pkl')