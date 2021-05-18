import xml.etree.cElementTree as ET
import os
import json

dic={}
dir='D:/Dataset/en'
cnt=0
for root_, dirs, files in os.walk(dir):
    for file in files:
        file_dir=os.path.join(root_,file)
        # print(file_dir)
        tree = ET.ElementTree(file=file_dir)
        root=tree.getroot()
        root=root[0]
        for entry in root:
            entities = []
            relations = []
            relationMentions = []
            for child in entry:
                if child.tag=='modifiedtripleset':
                    for triple in child:
                        tmp_1=triple.text.split('|')[0].strip().replace('_', ' ').strip('\"')
                        relation=triple.text.split('|')[1].strip().replace('_', ' ').strip('\"')
                        tmp_2=triple.text.split('|')[2].strip().replace('_', ' ').strip('\"')
                        relationMentions.append({"em1Text":tmp_1,
                                                 "em2Text":tmp_2,
                                                 "label":relation})
                        if tmp_1 not in entities:
                            entities.append(tmp_1)
                        if tmp_2 not in entities:
                            entities.append(tmp_2)
                        if relation not in relations:
                            relations.append(relation)
                flag=False
                if child.tag=='lex':
                    text = child.text
                    for entity in entities:
                        if text.find(entity)==-1:
                            flag=True
                            break
                    if not flag:
                        entityMentions=[]
                        for entity in entities:
                            entityMentions.append({"text":entity})
                        for relation in relations:
                            if relation not in dic:
                                dic[relation]=[]
                            dic[relation].append({"sentText":text,
                                                  "relationMentions":relationMentions,
                                                  "entityMentions":entityMentions})
relations=[]
for key in dic:
    if len(dic[key])<30:
        relations.append(key)
for relation in relations:
    dic.pop(relation)
json_obj=json.dumps(dic, indent=4)
with open('data/webnlg.json', 'w') as json_file:
    json.dump(dic, json_file)
print(json_obj)

