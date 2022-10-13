import numpy as np
import matplotlib.pyplot as plt
import os



# def read_test_acc(filename):
#     array=[]
#     max_run=0
#     with open(filename) as f:
#         for line in f:
#             if ('Run'in line.strip() )and ( 'Test' in line.strip()):
#                 # print(type(acc))
#                 acc=line.split()[-1]
#                 run=line.split()[1]
#                 if ',' in run:
#                     run=run.strip(',')
#                 run=int(run)
#                 max_run = run if run > max_run else max_run
#                 # print(type(acc))
#                 if '%' in acc:
#                     acc=acc[:-1] 
#                     acc=float(acc)
#                     acc=float("{0:.4f}".format(acc/100))
#                 else:
#                     acc=float(acc)
#                 array.append(acc)
#     print(array[:10])
#     print(len(array))
#     return array, max_run+1


# def draw(DATASET,  model, my_full, pseudo_mini_batch, path, n_run, fan_out=None):
    
#     fig,ax=plt.subplots(figsize=(24,6))
#     # x=range(len(bench_full))
#     length_full=len(my_full)
#     len_pseudo=len(pseudo_mini_batch)
#     if n_run>1:
#         len_pseudo=int(len_pseudo/n_run)
#     if len_pseudo<=100:
#         fig,ax=plt.subplots(figsize=(6,6))
#     # if len_pseudo<=100:
#     #     fig,ax=plt.subplots(figsize=(12,6))
#     len_cut = len_pseudo if len_pseudo < length_full else length_full
#     my_full=my_full[:len_cut]
#     pseudo_mini_batch=pseudo_mini_batch[:len_cut]
#     x1=range(len(my_full))
#     x2=range(len(pseudo_mini_batch))
#     # ax.plot(x, bench_full, label='benchmark '+DATASET )
    
#     ax.plot(x1, my_full, label='my script full graph '+DATASET)
#     ax.plot(x2, pseudo_mini_batch, label='pseudo_mini_batch_full_batch '+DATASET + '_fan-out_'+str(fan_out))
#     ax.set_title(model+' '+DATASET)
#     plt.ylim([0,1])
#     plt.xlabel('epoch')
    
#     # fig,ax=plt.subplots()
#     # ax.autoscale(enable=True,axis='y',tight=False)
#     # y_pos= np.arange(0,1000,step=100)
#     # labels=np.arange(0,1,step=0.1)
#     # print(labels)
#     # plt.yticks(y_pos,labels=labels)
#     plt.ylabel('Test Accuracy')
    
#     plt.legend()
#     # plt.savefig('reddit.pdf')
#     plt.savefig(path+DATASET+'.png')
#     # plt.show()

# def get_fan_out(filename):
#     fan_out=filename.split('_')[6]
#     print(fan_out)
#     return fan_out


# def full_graph_and_pseudo_mini(files, my_path, pseudo_mini_batch_path, model_p):
#     my_full=[]
#     pseudo_mini_batch=[]

#     my_path = my_path+model_p+'1_runs/'
#     # pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'10_runs/'
#     pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'1_runs/'
#     for file_in in files:
#         n_run=0
#         for filename in os.listdir(my_path):
#             if filename.endswith(".log"):
#                 f = os.path.join(my_path, filename)
#                 if file_in in f:
#                     my_full, n_run_full = read_test_acc(f)
#         f_i=0
#         for filename in os.listdir(pseudo_mini_batch_path):
#             if filename.endswith(".log"):
#                 f = os.path.join(pseudo_mini_batch_path, filename)
#                 if file_in in f:
#                     print(f)
#                     f_i+=1
#                     pseudo_mini_batch, n_run = read_test_acc(f)
#                     fan_out = get_fan_out(filename)
#                     draw(file_in,  model, my_full, pseudo_mini_batch, pseudo_mini_batch_path+'convergence_curve/'+str(f_i)+'_', n_run,fan_out)
#                     pseudo_mini_batch=[]
#         my_full=[]
        
#         print()

# def full_batch_and_pseudo_mini(files, my_path, pseudo_mini_batch_path, model_p):
#     my_full=[]
#     pseudo_mini_batch=[]

#     my_path = my_path+model_p+'1_runs/'
#     # pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'10_runs/'
#     pseudo_mini_batch_path = pseudo_mini_batch_path+model_p+'1_runs/'
#     for file_in in files:
#         n_run=0
#         for filename in os.listdir(my_path):
#             if filename.endswith(".log"):
#                 f = os.path.join(my_path, filename)
#                 if file_in in f:
#                     my_full, n_run_full = read_test_acc(f)
#         f_i=0
#         for filename in os.listdir(pseudo_mini_batch_path):
#             if filename.endswith(".log"):
#                 f = os.path.join(pseudo_mini_batch_path, filename)
#                 if file_in in f:
#                     print(f)
#                     f_i+=1
#                     pseudo_mini_batch, n_run = read_test_acc(f)
#                     fan_out = get_fan_out(filename)
#                     draw(file_in,  model, my_full, pseudo_mini_batch, pseudo_mini_batch_path+'convergence_curve/'+str(f_i)+'_', n_run,fan_out)
#                     pseudo_mini_batch=[]
#         my_full=[]
        
#         print()

def data_formalize(full, mini):
    length_full=len(full)
    len_mini=len(mini)
    len_cut = len_mini if len_mini < length_full else length_full
    full = full[:len_cut]
    mini = mini[:len_cut]
    x1=range(len(full))
    x2=range(len(mini))
    return x1, x2

def draw(acc_full,  acc_mini, loss_full, loss_mini):
    # plt.figure(figsize=(16, 4))
    # fig,ax=plt.subplots(figsize=(24,6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),)
    
    x1, x2 = data_formalize(acc_full, acc_mini)
    ax2.plot(x1, acc_full, label='full batch train')
    ax2.plot(x2, acc_mini, label='small batch train (batch size = 16)')
    ax2.text(0.5, 0.6, str("3-layer GraphSAGE using ogbn-products"),
           fontsize=12, ha='center')
    # ax2.set_title('GraphSAGE using OGBN-products')
    # plt.ylim([0,1])
    ax1.plot(x1, loss_full, label='full batch train')
    ax1.plot(x2, loss_mini, label='small batch train (batch size = 16)')
    
    ax2.set(xlabel='Epoch', ylabel='Test Accuracy')
    ax1.set(xlabel='Epoch', ylabel='Loss')
    # plt.xlabel('epoch')
    plt.legend()
    plt.savefig('full_v.s._mini.png')
    # plt.show()


def read_acc_loss(filename):
    acc_array = []
    loss_array = []
    with open(filename) as f:
        for line in f:
            if ('Run'in line.strip() )and ( 'Test' in line.strip()):
                # print(type(acc))
                acc=line.split()[-1]
                loss=line.split()[7]
                acc = float(acc)
                loss = float(loss)
                acc_array.append(acc)
                loss_array.append(loss)
                
    print(acc_array[:10])
    print(len(acc_array))
    return acc_array, loss_array

if __name__=='__main__':
    full_batch_file = "./full_batch_train_hidden_32.log"
    mini_batch_file = "./mini_batch_train_hidden_32.log"
    test_acc_full, loss_full = read_acc_loss(full_batch_file)
    test_acc_mini, loss_mini = read_acc_loss(mini_batch_file)
    draw(test_acc_full,  test_acc_mini, loss_full, loss_mini)
                                
    