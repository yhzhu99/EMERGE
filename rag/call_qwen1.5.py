import json
from tqdm import tqdm
from collections import Counter
import random
random.seed(42)
import requests
import pandas as pd
import pdb

port = 8029
ner_prompt_tmpl = """
[Instruction]
You are tasked with performing Named Entity Recognition (NER) specifically for diseases in a given medical case description to help with healthcare tasks (eg. readmission, motality, length of stay, drug prediction).  Follow the instructions below:
1. Input: You will receive a medical case description in the [Input].
2. NER Task: Focus on extracting the names of diseases as the target entity.
3. Output: Provide the extracted disease names in JSON format.

Ensure that the JSON output only includes the names of diseases mentioned in the provided [Input], excluding any additional content. The goal is to perform NER exclusively on disease names within the given text.

Example:
[Input]
 1:19 pm abdomen ( supine and erect ) clip  reason : sbo medical condition : 63 year old woman with reason for this examination : sbo final report indication : 63-year-old woman with small bowel obstruction . findings : supine and upright abdominal radiographs are markedly limited due to the patient 's body habitus . on the supine radiograph , there is a large distended loop of bowel in the right mid-abdomen , which extends towards the left . this measures 8 cm . it is difficult to accurately ascertain whether this is small or large bowel . if this is large bowel , a volvulus should be considered . if this is small bowel , it is markedly dilated . other prominent loops of small bowel are seen . additionally on the supine radiograph centered lower , there is a small amount of gas centered over the left femoral head . this could represent an incarcerated left inguinal hernia . impression : limited radiographs due to patient 's body habitus . this is consistent with an obstruction which may or may not involve a large bowel volvulus . these findings were telephoned immediately to the emergency room physician , . , caring for the patient .\",\"12:56 am chest ( portable ap ) clip reason : placement of cvl- r/o ptx , check position admitting diagnosis : small bowel obstruction medical condition : 63 year old woman with reason placement of cvl- r/o ptx , check position admitting diagnosis : small bowel obstruction medical condition : 63 year old woman with reason for this examination : placement of cvl- r/o ptx , check position final report indication : central venous line placement . views : single supine ap view , no prior studies . findings : the endotracheal tube is in satisfactory position approximately 4 cm from the carina . the right internal jugular central venous line is in satisfactory position with tip at the proximal superior vena cava . the study is limited by a lordotic position . low lung volumes are present bilaterally . the heart size appears enlarged . the pulmonary vascularity is difficult to assess . no pneumothorax is identified . no definite pulmonary infiltrates are present . the right costophrenic angle is sharp . the left costophrenic angle is excluded from the study . a nasogastric tube is seen which is looped within the fundus of the stomach with the tip pointing caudad within the distal stomach.
 
[Answer]
```json
{
"entities": ["small bowel obstruction",
            "large bowel volvulus",
            "incarcerated left inguinal hernia",
            "lordotic position",
            "low lung volumes",
            "enlarged heart",
            "pneumothorax"]
}
```

[Input]
{input}

[Answer]
"""


def parse_json(s: str) -> dict:
    lines = s.split('\n')
    p_start, p_end = -1, -1
    for idx, line in enumerate(lines):
        if '```' in line:
            if p_start == -1: p_start = idx
            else: p_end = idx
    try:
        if p_start == -1:
            p_start = 0
            p_end = len(lines)-1
        obj = json.loads('\n'.join(lines[p_start+1: p_end]))
    except:
        obj = {}
    return obj


def LLM_single(prompt: str):
    url = f'http://localhost:{port}/ner'
    method = 'POST'
    headers = {'Content-Type': 'application/json'}
    data = {
        "inputs": prompt
    }
    json_data = json.dumps(data)
    for try_time in range(3):
        response = requests.request(method, url, headers=headers, data=json_data)
        if response.status_code == 200:
            response_data = response.json()
            item = response_data["outputs"]
            obj = parse_json(item)
            try:
                entities = obj.get("entities", [])
            except:
                entities = []
            # pdb.set_trace()
            if len(entities) > 0:
                return entities
    print(f"try {try_time} times, error occur")
    return []


def LLM_single_test():
    query = """
sex m service neurology allergies morphine attending . chief complaint seizures, presurgical evaluation major surgical or invasive procedure none. history of present illness hpi the patient is a year old right handed man with a history of refractory temporal lobe epilepsy, depression, and prior spinal fusion who presents with simple partial, complex partial, and gtc seizures who presents with increased seizure frequency for presurgical evaluation. the patient reports having seizures since age when he had a febrile seizure gtc in the setting of high fever. between the ages of he had gtc seizures, the longest of which was 1.5 hours. at age , he no longer had gtc seizures and developed simple partial and complex partial seizures. the patient currently reports the following seizure types simple partial the feeling like he is in shock like the feeling when jumping into cold water with piloerection of his bilateral arms and testicular contraction. he occasionally smacks his lips and gets a metallic taste in his mouth. he still understands language during these events. they last seconds. he is currently having these 9 timesday, and these have become more frequent over the past few years. his last sps was a few days ago. his simple partial seizures can progress into complex partial seizures approximately onceweek. complex partial episodes in which he cannot get his words out and has difficulty understanding language with decreased awareness. these last up to 3 minutes, and occur once each week. it can take him 30 minutes to recover his language completely after a cps. his last cps was a few days ago. gtc he initially had these between ages , but started having them again this past year. his last gtc was 2 months ago, and they can last minutes. they are associated with tongue biting, but no loss of bowelbladder fucntion. they can be followed by a headache. he denies a history of a postictal paralysis. he has never been intubated for a seizure. he denies any preceding foul smells. he denies a history of head trauma. he denies any problems with his mothers pregnancy or delivery, and was in special classes for reading. he reports significant fatigue which limits his work. he denies any recent fevers. he denies any recent missed doses of medications. his seizure triggers include stress and lack of sleep. prior medications include dilantin, topamax, phenobarbital, and neurontin. he is currently on tegretol, keppra, and lamictal. per their report, prior mri brain in showed leftright hippocampal atrophy, and pet showed leftright medial temporal lobe and hippocampal hypometabolism. per report, prior ltm showed left frontal temporal interictal epileptiform discharge. the patient was seen by dr. dr. on . he reported that he has recently been struggling with fatigue and phonemic paraphasias, which they thought was due to both his seizures and medication side effects. they started a cross taper of his keppra to lamictal on with the plan to eventually taper off keppra. on he had a simple partial seizure which secondarily generalized lasting 10 minutes. he followed up with dr. dr. on at which time he reported increased frequency of his simple partial seizures. past medical history refractory temporal lobe epilepsy depression asthma kidney stones sp t11t12 and l5s1 spinal fusion social history family history there is no family history of epilepsy or febrile seizures. his paternal uncle has syndrome, his maternal grandfather had an mi at ages and , his mother has breast cancer. physical exam vs temp 96.5, bp 13872, hr 77, rr 16, sao2 97 on ra genl awake, alert, nad heent sclerae anicteric, no conjunctival injection, oropharynx clear with enlarged rightleft tonsils. during the exam the patient had a simple partial seizure with staring, brief lip smacking, and preserved language lasting 1 minute. neurologic examination mental status awake and alert, cooperative with exam, normal affect. speech is fluent with normal repetition; naming intact. no dysarthria. reading intact. cranial nerves pupils equally round and reactive to light, 5 to 3 mm bilaterally. extraocular movements intact bilaterally without nystagmus. sensation intact v1v3. facial movement symmetric. hearing intact to finger rub bilaterally. palate elevation symmetric. sternocleidomastoid and trapezius full strength bilaterally. tongue midline, movements intact. motor no observed myoclonus, asterixis, or tremor. slight left pronator drift. del tri bi we fe ff ip h q df r 5 l 5 reflexes 1 and symmetric in biceps, brachioradialis, triceps, knees, ankles. coordination finger to nose normal. pertinent results 0515pm wbc5.3 rbc4.78 hgb15.0 hct43.9 mcv92 mch31.3 mchc34.1 rdw12.8 0515pm plt count129 0515pm glucose82 urea n13 creat0.9 sodium137 potassium4.0 chloride100 total co231 anion gap10 0515pm calcium9.2 phosphate3.2 magnesium2.2 0515pm asaneg ethanolneg carbamzpn7.2 acetmnphnneg bnzodzpnneg barbitrtneg tricyclicneg 0559pm urine bloodneg nitriteneg proteinneg glucoseneg ketoneneg bilirubinneg urobilngnneg ph6.5 leuksm 0559pm urine bnzodzpnneg barbitrtneg opiatesneg cocaineneg amphetmnneg mthdoneneg levels keppra 5.8, tegretol 7.2, lamictal 3.1 brief hospital course the patient is a year old right handed man with a history of refractory temporal lobe epilepsy, depression, and prior spinal fusion who presents with simple partial, complex partial, and gtc seizures who presents with increased seizure frequency for presurgical evaluation. per report, prior mri brain in showed leftright hippocampal atrophy, and pet showed leftright medial temporal lobe and hippocampal hypometabolism. the exam is significant for slight left pronator drift and during the exam he had a simple partial seizure with staring, brief lip smacking, and preserved language lasting 1 minute. he will be admitted for eeg for further characterization and localization of his events. brief hospital course the patient was admitted to the epilepsy service . he had videoeeg throughout the hospitalization and his aeds were weaned off, except lamictal 200mg bid.
"""
    test_prompt = ner_prompt_tmpl.replace('{input}', query)
    for _ in tqdm([_ for _ in range(50)]):
        res = LLM_single(test_prompt)
        print(res)
    return res


def LLM_batch(prompts: list[str]):
    url = f'http://localhost:{port}/ner_batch'
    method = 'POST'
    headers = {'Content-Type': 'application/json'}
    data = {
        "inputs": prompts
    }
    json_data = json.dumps(data)
    for try_time in range(3):
        response = requests.request(method, url, headers=headers, data=json_data)
        total_entities = []
        if response.status_code == 200:
            response_data = response.json()
            for item in response_data["outputs"]:
                obj = parse_json(item)
                try:
                    entities = obj.get("entities", [])
                except:
                    entities = []
                total_entities += entities
            # pdb.set_trace()
            if len(total_entities) > 0:
                return total_entities
    print(f"try {try_time} times, error occur")
    return []


def LLM_batch_test():
    query = """
sex m service neurology allergies morphine attending . chief complaint seizures, presurgical evaluation major surgical or invasive procedure none. history of present illness hpi the patient is a year old right handed man with a history of refractory temporal lobe epilepsy, depression, and prior spinal fusion who presents with simple partial, complex partial, and gtc seizures who presents with increased seizure frequency for presurgical evaluation. the patient reports having seizures since age when he had a febrile seizure gtc in the setting of high fever. between the ages of he had gtc seizures, the longest of which was 1.5 hours. at age , he no longer had gtc seizures and developed simple partial and complex partial seizures. the patient currently reports the following seizure types simple partial the feeling like he is in shock like the feeling when jumping into cold water with piloerection of his bilateral arms and testicular contraction. he occasionally smacks his lips and gets a metallic taste in his mouth. he still understands language during these events. they last seconds. he is currently having these 9 timesday, and these have become more frequent over the past few years. his last sps was a few days ago. his simple partial seizures can progress into complex partial seizures approximately onceweek. complex partial episodes in which he cannot get his words out and has difficulty understanding language with decreased awareness. these last up to 3 minutes, and occur once each week. it can take him 30 minutes to recover his language completely after a cps. his last cps was a few days ago. gtc he initially had these between ages , but started having them again this past year. his last gtc was 2 months ago, and they can last minutes. they are associated with tongue biting, but no loss of bowelbladder fucntion. they can be followed by a headache. he denies a history of a postictal paralysis. he has never been intubated for a seizure. he denies any preceding foul smells. he denies a history of head trauma. he denies any problems with his mothers pregnancy or delivery, and was in special classes for reading. he reports significant fatigue which limits his work. he denies any recent fevers. he denies any recent missed doses of medications. his seizure triggers include stress and lack of sleep. prior medications include dilantin, topamax, phenobarbital, and neurontin. he is currently on tegretol, keppra, and lamictal. per their report, prior mri brain in showed leftright hippocampal atrophy, and pet showed leftright medial temporal lobe and hippocampal hypometabolism. per report, prior ltm showed left frontal temporal interictal epileptiform discharge. the patient was seen by dr. dr. on . he reported that he has recently been struggling with fatigue and phonemic paraphasias, which they thought was due to both his seizures and medication side effects. they started a cross taper of his keppra to lamictal on with the plan to eventually taper off keppra. on he had a simple partial seizure which secondarily generalized lasting 10 minutes. he followed up with dr. dr. on at which time he reported increased frequency of his simple partial seizures. past medical history refractory temporal lobe epilepsy depression asthma kidney stones sp t11t12 and l5s1 spinal fusion social history family history there is no family history of epilepsy or febrile seizures. his paternal uncle has syndrome, his maternal grandfather had an mi at ages and , his mother has breast cancer. physical exam vs temp 96.5, bp 13872, hr 77, rr 16, sao2 97 on ra genl awake, alert, nad heent sclerae anicteric, no conjunctival injection, oropharynx clear with enlarged rightleft tonsils. during the exam the patient had a simple partial seizure with staring, brief lip smacking, and preserved language lasting 1 minute. neurologic examination mental status awake and alert, cooperative with exam, normal affect. speech is fluent with normal repetition; naming intact. no dysarthria. reading intact. cranial nerves pupils equally round and reactive to light, 5 to 3 mm bilaterally. extraocular movements intact bilaterally without nystagmus. sensation intact v1v3. facial movement symmetric. hearing intact to finger rub bilaterally. palate elevation symmetric. sternocleidomastoid and trapezius full strength bilaterally. tongue midline, movements intact. motor no observed myoclonus, asterixis, or tremor. slight left pronator drift. del tri bi we fe ff ip h q df r 5 l 5 reflexes 1 and symmetric in biceps, brachioradialis, triceps, knees, ankles. coordination finger to nose normal. pertinent results 0515pm wbc5.3 rbc4.78 hgb15.0 hct43.9 mcv92 mch31.3 mchc34.1 rdw12.8 0515pm plt count129 0515pm glucose82 urea n13 creat0.9 sodium137 potassium4.0 chloride100 total co231 anion gap10 0515pm calcium9.2 phosphate3.2 magnesium2.2 0515pm asaneg ethanolneg carbamzpn7.2 acetmnphnneg bnzodzpnneg barbitrtneg tricyclicneg 0559pm urine bloodneg nitriteneg proteinneg glucoseneg ketoneneg bilirubinneg urobilngnneg ph6.5 leuksm 0559pm urine bnzodzpnneg barbitrtneg opiatesneg cocaineneg amphetmnneg mthdoneneg levels keppra 5.8, tegretol 7.2, lamictal 3.1 brief hospital course the patient is a year old right handed man with a history of refractory temporal lobe epilepsy, depression, and prior spinal fusion who presents with simple partial, complex partial, and gtc seizures who presents with increased seizure frequency for presurgical evaluation. per report, prior mri brain in showed leftright hippocampal atrophy, and pet showed leftright medial temporal lobe and hippocampal hypometabolism. the exam is significant for slight left pronator drift and during the exam he had a simple partial seizure with staring, brief lip smacking, and preserved language lasting 1 minute. he will be admitted for eeg for further characterization and localization of his events. brief hospital course the patient was admitted to the epilepsy service . he had videoeeg throughout the hospitalization and his aeds were weaned off, except lamictal 200mg bid.
"""
    test_prompt = ner_prompt_tmpl.replace('{input}', query)
    batch_size = 1
    test_prompts = [test_prompt for _ in range(batch_size)]
    for _ in tqdm([_ for _ in range(50)]):
        res = LLM_batch(test_prompts)
        print(res)
    return res


def extract_dataset(input_pkl, output_json, start_idx=0, end_idx=None):
    window_thresh = 2000  # qwen 8192,留一些生成文本的空间
    lines = pd.read_pickle(input_pkl)
    lines_part = lines[start_idx:end_idx]
    with open(output_json, 'a+', encoding='utf8') as fout:
        for idx, obj in enumerate(tqdm(lines_part)):
            words = obj['Texts'].split()
            records = [' '.join(words[i:i+window_thresh]) for i in range(0, len(words), window_thresh)]
            print(f"split into {len(records)} chunks")
            sample_entities = []
            if len(records) > 4:
                records = random.sample(records, 4)  # 太多了，推理巨慢
            for r in records:  # for each record
                ner_prompt = ner_prompt_tmpl.replace('{input}', r)
                record_entities = LLM_single(ner_prompt)
                sample_entities += record_entities  # 核心代码
            new_obj = {
                'PatientID': obj['PatientID'],
                'Entities': sample_entities,
                'Texts': obj['Texts']
            }
            l = json.dumps(new_obj, ensure_ascii=False)
            print(l, file=fout, flush=True)



if __name__ == "__main__":
    extract_dataset('./mimic4_all/ts_note_all.pkl', './mimic4_all/output6000.json', start_idx=6000+2895, end_idx=9000)
    # LLM_batch_test()
    # LLM_single_test()