import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.LViT import LViT
from utils import *
import cv2

import time
import psutil
import numpy as np
import torch

issave = 0

def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred)
    target = np.array(target)
    
    if pred.ndim == 3:  
        batch_size = pred.shape[0]
        acc_list = []
        recall_list = []
        tp_list = []
        tn_list = []
        fp_list = []
        fn_list = []
        
        for i in range(batch_size):
            acc, recall, tp, tn, fp, fn = calculate_metrics(pred[i], target[i], threshold)
            acc_list.append(acc)
            recall_list.append(recall)
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
        
        return np.mean(acc_list), np.mean(recall_list), np.mean(tp_list), np.mean(tn_list), np.mean(fp_list), np.mean(fn_list)

    pred_binary = (pred >= threshold).astype(np.int32)
    target_binary = (target > 0).astype(np.int32)
    
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    
    return accuracy, recall, tp, tn, fp, fn

def show_image_with_dice(predict_save,predict_cu_save, labs, save_path,save_cu_path=None):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    acc, recall, tp, tn, fp, fn = calculate_metrics(predict_save, labs)
    if issave==1:
        predict_save = cv2.pyrUp(predict_save, (224, 224))
        predict_save = cv2.resize(predict_save, (224, 224))
        cv2.imwrite(save_path, predict_save * 255)

        predict_cu_save = cv2.pyrUp(predict_cu_save, (224, 224))
        predict_cu_save = cv2.resize(predict_cu_save, (224, 224))
        cv2.imwrite(save_cu_path, predict_cu_save * 255)
    return dice_pred, iou_pred, acc, recall, tp, tn, fp, fn

def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens,name):
    model.eval()

    infer_time = 0.0
    gpu_max_memory_allocated = 0.0
    cpu_memory_used = 0.0
    
    with torch.no_grad(): 
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()  
            torch.cuda.empty_cache() 

        torch.cuda.synchronize()
        start_time = time.time()  
        
        output,preds_cu = model(input_img.cuda(), text.cuda(),name = name)
        
        torch.cuda.synchronize()
        end_time = time.time()  
        
        infer_time = end_time - start_time
    
    gpu_memory_allocated = 0.0
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  
        gpu_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  
    
    
    process = psutil.Process(os.getpid())
    cpu_memory_used = process.memory_info().rss / (1024 ** 3) 
 
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    pred_cu_class = torch.where(preds_cu > 0.5, torch.ones_like(preds_cu), torch.zeros_like(preds_cu))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    predict_cu_save = pred_cu_class[0].cpu().data.numpy()
    predict_cu_save = np.reshape(predict_cu_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp, acc, recall, tp, tn, fp, fn = show_image_with_dice(
        predict_save,predict_cu_save, labs, save_path=vis_save_path+'.png',save_cu_path=vis_save_path+'_cu.png'
    )
    return dice_pred_tmp, iou_tmp, acc, recall, tp, tn, fp, fn, infer_time, gpu_memory_allocated, gpu_max_memory_allocated, cpu_memory_used

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    if config.task_name == "MoNuSeg":
        
        lunmodel = ""
        test_num = 360
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/"+ lunmodel +"/best_model-" + model_type + ".pth.tar"
    
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = ""
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    acc_t = 0.0
    recall_t = 0.0


    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # print(names)
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255


            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t, acc, recall, tp, tn, fp, fn, infer_time, gpu_mem, gpu_max_mem, cpu_mem = vis_and_save_heatmap(
                model, input_img, test_text, None, lab, vis_path + str(names), dice_pred, dice_ens, name=str(names)
            )

            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            acc_t += acc
            recall_t +=recall
            torch.cuda.empty_cache()
            pbar.update()
    print("dice", dice_pred / test_num)
    print("iou", iou_pred / test_num)
    print("acc", acc_t / test_num)
    print("recall", recall_t / test_num)