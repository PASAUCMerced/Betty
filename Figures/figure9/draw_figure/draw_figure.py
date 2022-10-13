import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import json
import ast
 


def draw(full_dict,  micro_dict_list):
    full_dict = full_dict[0]
    print(full_dict.keys())
    lists = dict(sorted(full_dict.items()))
    print(lists)
    
    micro_dict_0 = micro_dict_list[0]
    micro_dict_1 = micro_dict_list[1]
    dict_micro_0 = dict(sorted(micro_dict_0.items()))
    dict_micro_1 = dict(sorted(micro_dict_1.items()))
    print(dict_micro_1)
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4),)
    ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    width = .27
    x1, x2 = list(lists.keys()), list(lists.keys())
    x1 = np.array(x1)
    bar_full = ax1.bar(x1, lists.values(), width,  fill=False, hatch='//', edgecolor='g', label='Full batch',)
    ax1.set(xlabel='# of in-degree', ylabel='# of nodes')
    t = ax1.text(1, 24000,"Full batch", bbox=dict(facecolor='none', edgecolor='grey', alpha=0.5))
    bb = t.get_bbox_patch()
    bb.set_boxstyle("round", pad=0.6)
    
    x2 = np.array(x2)
   
   
    bar_micro_0=ax2.bar(x2-0.25, dict_micro_0.values(),width, label='Micro batch 0',fill=False, hatch='///', edgecolor='g')
    bar_micro_1=ax2.bar(x2+0.25, dict_micro_1.values(),width,  label='Micro batch 1', fill=False, hatch='..', edgecolor='#9a0eea')
    
    ax2.set(xlabel='# of in-degree', ylabel='# of nodes')
    
    plt.legend()
   
    plt.savefig('Figure_9.png')
    # plt.show()


def read_in_degree(filename):
    dic_list = []
    with open(filename) as f:
        for line in f:
            if ('Counter'in line.strip() ):
                # print(line.strip())
                tmp = line.strip()
                str_dict = tmp.split("(")[-1]
                # print(str_dict)
                str_dict = str_dict[:-1]
                # print(str_dict)
                # 
              
                str_dict = json.dumps(ast.literal_eval(str_dict))
                dict = json.loads(str(str_dict))
                print(dict)
                
                dict = {int(k):int(v) for k,v in dict.items()}
                dic_list.append(dict)
    
    return dic_list

if __name__=='__main__':
    full_batch_file = "./full_batch_train.log"
    micro_batch_file = "./2_micro_batch_train.log"
    full_in_degree = read_in_degree(full_batch_file)
    micro_in_degree_list = read_in_degree(micro_batch_file)
    draw(full_in_degree,  micro_in_degree_list)
                                
    