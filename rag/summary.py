import pandas as pd
import pdb


def peek_data(input_file, output_file):
    data = pd.read_pickle(input_file)
    for item in data:
        entities = str(item['Entities'])
        item['Entities'] = entities
        nodes = item['Nodes']
        docs = item['Documents']
        triples = item['Triples']
        prompt = summary_template.replace("{entities}", entities).replace("{documents}", docs).replace("{triples}", triples)
        prompt = prompt.strip()
        # pdb.set_trace()
        del item['X']
        del item['Y']
        del item['Texts']
    pd.to_pickle(data, output_file)


summary_template = """
[Instruction]
Now you are a experienced clinical professor, given some diseases that one patient suffered from and related documents and knowledge graph triples of these diseases, which may help you understand them better. Please try your best to summary the health condition of this patient and draw a brief conclusion in 512 tokens. You donnot need to explain or give interpretation of given terms.
[Diseases]
{entities}
[Documents]
{documents}
[Related knowledge graph triples]
{triples}
"""

example_result = """
This patient's clinical presentation suggests a complex interplay of cardiopulmonary issues, skeletal abnormalities, and potential malignancies. The patient has undergone a myotomy, a surgical procedure which could possibly relate to the treatment of esophageal issues, such as those arising from a hiatus hernia, where a portion of the stomach pushes through the diaphragm into the chest cavity.

The cardiopulmonary complications are significant, with an ST-elevated myocardial infarction indicating a severe heart attack, as well as pneumonia and pulmonary edema suggesting acute respiratory distress. Dyspnea, or difficulty breathing, is a symptom consistent with these diagnoses, and collectively, these conditions pose a risk for respiratory failure.

The patient's skeletal concerns include a right proximal humerus issue, possibly a fracture, and a right humeral diaphysis enchondroma—a benign bone tumor made of cartilage. Lytic lesions indicate bone destruction, commonly associated with malignancy or other systemic pathology.

There are indications of a lung mass and pleural effusions, which are significant collections of fluid in the pleural space signifying either infection, inflammation, or malignancy. While the specific nature of the lung mass isn't detailed, in the context of lytic skeletal lesions, a differential could include primary lung cancer with metastases to bone.

The lordotic position noted may relate to attempts to relieve dyspnea or could be a postural response to musculoskeletal pain or discomfort. Minimal lung volumes and upper zone redistribution might imply restrictive lung disease or compensatory response to pulmonary pathology.

"Chondroid calcifications" suggest areas of calcification within cartilage, which could be incidental or relate to systemic disease. The mention of retrocardiac density on imaging could be due to the aforementioned lung or cardiac pathology.

In conclusion, the patient requires careful multidisciplinary management, focused on respiratory and cardiac support due to acute events, further diagnostic workup for the lung mass and suspicious bone lesions, and treatment for the plethora of ongoing issues, ranging from infectious to potentially neoplastic processes.
"""

if __name__ == "__main__":
    print(int(len(example_result.split(' ')) * 4/3))  # 508 ~ 512
    # peek_data('./ts_note_kg.pkl', './ts_note_final.pkl')