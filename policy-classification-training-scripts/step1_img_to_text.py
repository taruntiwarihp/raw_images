import boto3
import json
import pandas as pd
from glob import glob
from tqdm import tqdm 
# Get the credentials values from the response
access_key = "AKIAV625SOF2HBPAH2VJ"
secret_key = "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"
textract_client = boto3.client('textract', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name='us-east-2')
input_type = 'jpg'
img_pth = sorted(glob("data/*/*.jpg"))
print(len(img_pth))

data = {}
data['Labels'] = []
data['Img_pth'] = []
data['Texts'] = []
data['Sentence'] = []
data['Multi_column'] = []

c = 0
for img in tqdm(img_pth):
    # print(img)
    data['Labels'].append(img.split("/")[1])
    data['Img_pth'].append(img)
    
    with open(img, 'rb') as file:
        file_bytes = file.read()
        
    response = textract_client.detect_document_text(Document={'Bytes': file_bytes})
    save_file = open("json_data/{}_{}_data.json".format(img.split("/")[1], img.split("/")[2].split(".")[0]), "w")  
    # Detect columns and print lines
    columns = []
    lines = []
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            column_found=False
            for index, column in enumerate(columns):
                bbox_left = item["Geometry"]["BoundingBox"]["Left"]
                bbox_right = item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"]["Width"]
                bbox_centre = item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"]["Width"]/2
                column_centre = column['left'] + column['right']/2

                if (bbox_centre > column['left'] and bbox_centre < column['right']) or (column_centre > bbox_left and column_centre < bbox_right):
                    #Bbox appears inside the column
                    lines.append([index, item["Text"]])
                    column_found=True
                    break
            if not column_found:
                columns.append({'left':item["Geometry"]["BoundingBox"]["Left"], 'right':item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"]["Width"]})
                lines.append([len(columns)-1, item["Text"]])

    lines.sort(key=lambda x: x[0])
    val = []
    for line in lines:
        val.append(line[1])
        
    data['Multi_column'].append(" ".join(val))

    json.dump(response['Blocks'], save_file, indent = 6)  
    save_file.close() 
    line_lst = []
    for pair in response['Blocks']:
        if pair['BlockType'] == "LINE":
            line_lst.append(pair['Text'])

    data['Texts'].append(line_lst)
    data['Sentence'].append(" ".join(line_lst))


df = pd.DataFrame(data)
df.to_csv("final_data.csv", index=False)
print(df.head())

