"""
use chatGPT or GPT4 to complete the chat
"""
import os
import wandb
import openai
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
#os.environ["OPENAI_API_BASE"] = 'https://one-api.bltcy.top'
openai_key = 'sk-4rsvPuSYzRTLcRVCFa985b112fDd4f7283B48e8eCa080aBf'
def request(
        model,
        messages,
        max_tokens=512,
        temperature=0.85,
        top_p=0.7,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        api_key=None
):
    #print(model,messages,api_key)
    openai.api_base = 'https://one-api.bltcy.top/v1'
    openai.api_key = api_key
    if type(messages) is str:
        messages = [
            {
                "role": "user",
                "content": messages
            }
        ]
    if model == "gpt-35-turbo":
        model = "gpt-3.5-turbo"
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                timeout=2
            )
            break
        except Exception as e:
            print(str(e))
            print("Retrying...")
            if "Limit: 3 / min" in str(e):
                time.sleep(10)
            time.sleep(2)

    generations = [gen['message']['content'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if len(_) != 0]
    return generations



'''
I have some texts along with their corresponding scores. The texts are arranged in ascending order
based on their scores, where higher scores indicate better quality.
text:
Let’s figure it out!
score:
61
text:
Let’s solve the problem.
score:
63
(. . . more instructions and scores . . . )
The following exemplars show how to apply your text: you replace <INS> in each input with your
text, then read the input and give an output. We say your output is wrong if your output is different
from the given output, and we say your output is correct if they are the same.
input:
Q: Alannah, Beatrix, and Queen are preparing for the new school year and have been given books
by their parents. Alannah has 20 more books than Beatrix. Queen has 1/5 times more books than
Alannah. If Beatrix has 30 books, how many books do the three have together?
A: <INS>
output:
140
(. . . more exemplars . . . )
Write your new text that is different from the old ones and has a score as high as possible. Write the
text in square brackets
'''
# parser = argparse.ArgumentParser()
# parser.add_argument("--tokenizer_path", type=str, default="gpt2", help="Path, url or short name of the tokenizer")
# args = parser.parse_args()
def Aviaiable_test():
    messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": "hello"
            }
        ]
    re = request(model="gpt-4-0613", messages=messages, api_key=openai_key)
    print(re)
Aviaiable_test()
wandb.init(project='length_vis')
for number in [150,200,250,300]:
    re = []
    for idx in tqdm(range(100)):
        content =  '''I will give you a piece of text and its corresponding summary, ###Text: {}\n ###Summary: {}.\n Please rewrite the summary part so that it only contains {} characters.'''.format(data['test']['article'][idx],data['test']['highlights'][idx],number)
        print(content)
        messages = [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
        re.append(request(model="gpt-4-1106-preview", messages=messages, api_key=openai_key)[0])
        print(re[-1])
    for x in re:
        wandb.log({f'{number}':len(x)})
        x=x.replace(' ','')
        wandb.log({f'{number} without block':len(x)})
wandb.finish()
# '''
# I want you to predict the congestion status on the freeways in the San Francisco Bay area. 
# I will provide you with some data from different sensors, and I would like you to output the measured road occupancy rates (ranging from 0 to 1).
# I want you to only reply the number of the measured road occupancy rates, and nothing else. Do not write explanations.

# Q:The data from sensors is "0.0048  0.0146  0.0289  0.0142  0.0064  0.0232  0.0162  0.0242  0.0341  0.0375"
# A:0.0121
# Q:The data from sensors is "0.0285  0.0253  0.0542  0.0485  0.0131  0.0331  0.0372  0.0569  0.0669  0.0409"
# A:
# '''

# '''
# I want you to predict the temperature of the oil. You will play the role of a time series data predictor, and I will provide you with the data for oil temperature and the extreme load on the power transformer for the previous time points. 
# For each row of data, each datum is separated by a comma. The first datum represents the timestamp of that row's data, followed by High Useful Load, High Useless Load, Middle Useful Load, Middle Useless Load, Low Useful Load, Low Useless Load, and the last element is the oil temperature, which is the value we need to predict. I have provided you with all the data, including timestamps and oil temperatures for the previous time points. Your task is to identify relationships within this data to predict the oil temperature for the last timestamp (i.e., the last row).
# Based on the data relationships from earlier times, your goal is to predict the last element of the last row, which is the value at the question mark.
# I want you to only reply the number of the predicted data(Output with 15 decimal places preserved), and nothing else. Do not write explanations.

# The recorded date,High UseFul Load,High UseLess Load,Middle UseFul Load,Middle UseLess Load,Low UseFul Load,Low UseLess Load,Oil Temperature(target)
# 2016-07-01 00:00:00,5.827000141143799,2.009000062942505,1.5989999771118164,0.4620000123977661,4.203000068664552,1.3400000333786009,30.5310001373291
# 2016-07-01 01:00:00,5.692999839782715,2.075999975204468,1.4919999837875366,0.4259999990463257,4.142000198364259,1.371000051498413,27.78700065612793
# 2016-07-01 02:00:00,5.1570000648498535,1.741000056266785,1.2790000438690186,0.35499998927116394,3.776999950408936,1.218000054359436,27.78700065612793
# 2016-07-01 03:00:00,5.0900001525878915,1.9420000314712524,1.2790000438690186,0.3910000026226044,3.806999921798706,1.2790000438690186,25.04400062561035
# 2016-07-01 04:00:00,5.357999801635742,1.9420000314712524,1.4919999837875366,0.4620000123977661,3.868000030517578,1.2790000438690186,21.947999954223643
# 2016-07-01 05:00:00,5.625999927520752,2.1429998874664307,1.5279999971389768,0.5329999923706055,4.051000118255615,1.371000051498413,21.173999786376953
# 2016-07-01 06:00:00,7.166999816894531,2.9470000267028813,2.131999969482422,0.7820000052452087,5.026000022888184,1.8580000400543213,?
# '''
# def cal_correlation(predict,test):
#     predict = np.array(predict)
#     Ytest = np.array(test)
#     sigma_p = (predict).std(axis = 0)
#     sigma_g = (Ytest).std(axis = 0)
#     mean_p = predict.mean(axis = 0)
#     mean_g = Ytest.mean(axis = 0)
#     index = (sigma_g!=0)
#     correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g)
#     correlation = (correlation[index]).mean()
#     return correlation

# df = pd.read_csv('./prediction/ETT-small/ETTh1.csv')
# data_x = df.iloc[:,1:].values
# data_y = df.iloc[:,-1].values.tolist()
# response = {'pred':[],'mean':[],'ref':[]}
# data_len = 48
# for idx in tqdm(range(data_len,data_len+500,10)):
#     persona =  '''I want you to predict the temperature of the oil. You will play the role of a time series data predictor, and I will provide you with the data for oil temperature and the extreme load on the power transformer for past time points. 
#     For each row of data, each datum is separated by a comma. The first datum represents the timestamp of that row's data, followed by High Useful Load, High Useless Load, Middle Useful Load, Middle Useless Load, Low Useful Load, Low Useless Load, and the last element is the oil temperature, which is the value we need to predict.  
#     Your task is to predict the oil temperature data for the next 24 time points
#     I want you to only reply 24 numbers (Output with 15 decimal places preserved), representing the oil temperature for the next 24 time points and nothing else. Do not write explanations.
#     '''
#     content = []
#     for i in range(idx-data_len,idx+1):
#         content.append(','.join(['{:.10f}'.format(value) for value in data_x[i]]))
#         content[-1] = df.iloc[i,0] + ',' + content[-1]
#     content = "\n".join(content)
#     content = f'''
#     For each row of data, each datum is separated by a comma. The first datum represents the timestamp of that row's data, followed by High Useful Load, High Useless Load, Middle Useful Load, Middle Useless Load, Low Useful Load, Low Useless Load, and the last element is the oil temperature, which is the value we need to predict. 
#     The data is:
#     <
#     The recorded date,High UseFul Load,High UseLess Load,Middle UseFul Load,Middle UseLess Load,Low UseFul Load,Low UseLess Load,Oil Temperature(target)
#     {content}
#     >
#     Perform the following actions:
#     1.Identify patterns in each line of data.
#     2.Based on the recorded data, identify patterns among the various data points between the previous moment and the next moment in time.
#     3.Use the identified pattern to predict the oil temperature values for the next 24 time points.
#     the format of output is following:
#         <oil temperature value1>
#         <oil temperature value2>
#         ....
#         <oil temperature value24>
#     '''
#     messages = [
#         {
#             "role": "system",
#             "content": persona
#         },
#         {
#             "role": "user",
#             "content": content
#         },
#     ]
#     try:
#         re = request(model="gpt-35-turbo", messages=messages, api_key=openai_key)[0]
#         re = [float(x) for x in re.split('\n')]
#         y = data_y[idx+1:idx+1+24]
#         assert len(re) == 24
#         # pred_list = []
#         # pred_times = 5
#         # for x in range(pred_times):
#         #     re = request(model="gpt-35-turbo", messages=messages, api_key=openai_key)[0]
#         #     try:
#         #         re = float(re)
#         #         pred_list.append(re)
#         #     except ValueError:
#         #         continue
#         # print(pred_list)
#         # print(f"{y},medium:{statistics.median(pred_list)},mean:{statistics.mean(pred_list)}")
#         response['pred'] = response['pred']+re
#         response['ref'] = response['ref'] + y
#     except:
#         continue