import gc
import math
from pathlib import Path
from lib.models.bat import build_batrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import vot
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import torch.nn as nn
import numpy as np

class BATTrack(BaseTracker):
    def __init__(self, params):
        super(BATTrack, self).__init__(params)
        network = build_batrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)  
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()


        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = True #params.debug   
        #self._init_visdom(None, 1)
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

        self.last_template = None
        self.dynamic_template = None
        self.templates_list = []
        self.search_list = []
        self.update_templates_list = []

        # ========= template save & log =========
        # self.template_save_dir = None
        # self.template_log_file = None
        # self.sequence_name = None
        # self.template_update_count = 0



    def initialize(self, image, info: dict):
        # forward the template once
        H, W, _ = image.shape
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        #uncertainty note
        self.u_min = None
        self.ui_min = None
        self.new_template = None
        self.rgb_update = False
        self.tir_update = False
        self.count = 0 
        self.score_list = []
        self.ulist = []
        self.score_mean = None
        self.u_mean = None
        self.score_var  = None
        self.u_var = None
        self.interval = 10

        # 清理缓存
        torch.cuda.empty_cache()
        self.new_template_list = []
        self.search_lists = []
        self.max_score_list = []
        # self.params.save_dir = 'templates_log'
        

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
       
        self.search_init = self.preprocessor.process(x_patch_arr)

        template = self.preprocessor.process(z_patch_arr)
        # print('template',template.size())
        self.initial_template = template
        self.last_template = template


        
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)
        H,W,_ = z_patch_arr.shape
        self.template_mask = self.get_bboxes_masks(template_bbox,1,H,W)
        self.template_bboxes_masks = self.template_mask.view(1, -1)
        self.template_mask = torch.where(self.template_bboxes_masks.to('cuda:0'), torch.tensor(1).to('cuda:0'), torch.tensor(0).to('cuda:0')).to('cuda:0') 

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


        # ========= sequence info =========
        # self.sequence_name = info.get('sequence_name', name)

        # self.template_save_dir = Path(self.params.save_dir) / self.sequence_name / 'templates'
        # self.template_save_dir.mkdir(parents=True, exist_ok=True)

        # self.template_log_file = Path(self.params.save_dir) / self.sequence_name / 'template_quality.txt'

        # 写文件头（只在初始化写一次）
        # with open(self.template_log_file, 'w') as f:
        #     f.write('frame_id init_template_score new_template_score\n')



    def _save_template_image(self, template_tensor, frame_id):
        """
        保存模板 tensor 为 png
        template_tensor: [1, C, H, W]
        """
        T_r = template_tensor[ :, :,:3]
        T_t = template_tensor[ :, :,3:]        


        save_path_r = self.template_save_dir / f'template_rgbf{frame_id:04d}.jpg'
        save_path_t = self.template_save_dir / f'template_tirf{frame_id:04d}.jpg'
        cv2.imwrite(str(save_path_r), cv2.cvtColor(T_r, cv2.COLOR_BGR2RGB))
        cv2.imwrite(str(save_path_t), cv2.cvtColor(T_t, cv2.COLOR_BGR2RGB))



    def track(self, image, info: dict = None,vis_feat = None):
        
        H, W, _ = image.shape
        self.frame_id += 1

        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        new_template = self.preprocessor.process(z_patch_arr)

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
       
        search = self.preprocessor.process(x_patch_arr)
        search_init = search
        # print('search',search.size())
        if self.frame_id == 1:
            self.templates_list = [self.initial_template,self.initial_template]
            self.search_list = [search_init,search_init]
            x_tensor = torch.cat(self.search_list,dim=0)
            self.z_tensor = torch.cat(self.templates_list,dim=0)

        elif self.frame_id % self.interval == 1 and self.frame_id !=1 and self.new_template is not None:
            self.search_list = [search,search,self.new_search,self.new_search]
            x_tensor = torch.cat(self.search_list,dim=0)
            self.templates_list = [self.initial_template,self.last_template,self.initial_template,self.new_template]
            self.z_tensor = torch.cat(self.templates_list,dim=0)
        else:
            self.search_list = [search,search]
            x_tensor = torch.cat(self.search_list,dim=0)
            self.templates_list = [self.initial_template,self.last_template]
            self.z_tensor = torch.cat(self.templates_list,dim=0)        

        if self.frame_id == 1:
            c = 1

        with torch.no_grad():
            # x_tensor = search
            x_tensor = x_tensor
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z,Test=True,frame_id =self.frame_id,template_masks=self.template_mask)
            # if self.frame_id % 4 == 0:
            #     self.search_list.pop()
            

        # add hann windows
        #torch.Size([1, 1, 16, 16])
        [u_m,u,ui] = out_dict['u_m']
        self.u_m = u_m[0]        



        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes[0].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        self.max_score_list.append(max_score)

        #test for max template update or random template update
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state , self.params.template_factor,
                                                    output_sz=self.params.template_size)
        new_template = self.preprocessor.process(z_patch_arr)
        # self.update_templates_list.append(new_template)


        # if self.frame_id % self.interval == 0:
        #     # 根据最大分数选择模板
        #     if self.max_score_list:  # 确保列表不为空
        #         max_score_index = self.max_score_list.index(max(self.max_score_list))
        #         selected_template = self.update_templates_list[max_score_index]
        #         self.last_template = selected_template
                
        #         # 清空列表以便下一轮使用
        #         self.max_score_list.clear()
        #         self.update_templates_list.clear()            


        # import random

        # if self.frame_id % self.interval == 0:
        #     # 随机选择模板
        #     if self.update_templates_list:  # 确保列表不为空
        #         # 随机选择一个索引
        #         random_index = random.randint(0, len(self.update_templates_list) - 1)
        #         selected_template = self.update_templates_list[random_index]
        #         self.last_template = selected_template
                
        #         # 使用选中的模板进行更新
        #         # 这里需要根据您的具体需求添加模板更新的逻辑
        #         # 例如：self.update_template(selected_template)
                
        #         # 清空列表以便下一轮使用
        #         self.max_score_list.clear()
        #         self.update_templates_list.clear()


        threshold = np.mean(self.max_score_list)
        template_scores = out_dict["relative_score"]
        scores = out_dict["score"]

        if self.frame_id % self.interval == 1 and self.frame_id != 1 and self.new_template is not None:
            init_t = template_scores[1, 0].item()
            new_t  = template_scores[1, 1].item()

            # ===== write quality log =====
            # with open(self.template_log_file, 'a') as f:
            #     f.write(f'{self.frame_id} {init_t:.6f} {new_t:.6f} {best_score[2][0].item():.6f} {best_score[3][0]:.6f}\n')

            # ===== template update =====
            if init_t < new_t or best_score[3][0].item() >= best_score[2][0].item():
                self.last_template = self.new_template
                # self.template_update_count += 1

                # ===== save template image =====
            # self._save_template_image(z_patch_arr, self.frame_id)



        if max_score > threshold:
            z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state , self.params.template_factor,
                                                        output_sz=self.params.template_size)
            self.new_template = self.preprocessor.process(z_patch_arr)
            # self.new_template_list.append(new_template)
        if self.frame_id % (self.interval/2) == 0:
            self.search_lists.append(search)
            if self.frame_id % self.interval == 0:
                num = int(len(self.search_lists)/2)
                self.new_search = self.search_lists[num]
                z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, self.state , self.params.template_factor,
                                                            output_sz=self.params.template_size)
                # 替换为 None  删减内存/显存占用
                for i in range(num):
                    obj = self.search_lists[i]
                    self.search_lists[i] = None
                    del obj   # 删除原对象的引用
                # 强制垃圾回收（CPU）
                gc.collect()
                # 如果是 PyTorch Tensor，释放显存缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            


        if vis_feat:
            x_rs = search[0, :3, :, :]
            x_is = search[0, 3:, :, :]
            T_r = new_template[:, :3, :, :]
            T_i = new_template[:, 3:, :, :]
            x_dict = [x_rs,x_is,T_r,T_i]
            u_m = out_dict['u_m']         
            # return x_dict,out_dict,u_m,max_score,self.score_mean,self.u_mean,self.score_var,self.u_var 
            return x_dict,out_dict,u_m,max_score  
        
        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_BGR, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_BGR)
            cv2.waitKey(1)

        #get new template for next predict#
        # z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
        #                                             output_sz=self.params.template_size)
        # self.z_patch_arr = z_patch_arr
        ###-------###

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    def find_median(self,lst):
        # 先对列表进行排序
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        
        # 找到中位数的位置
        if n % 2 == 1:
            # 如果列表长度为奇数，中位数是中间的数
            median_index = (n + 1) // 2 - 1
            return sorted_lst[median_index]
        else:
            # 如果列表长度为偶数，中位数是中间两个数的平均值
            median_index1 = median_index2 = n // 2 - 1
            return (sorted_lst[median_index1] + sorted_lst[median_index1 + 1]) / 2
  
    def get_bboxes_masks(self, bboxes, B, H, W, patch_size=16):
        # 计算grid大小
        grid_size = H // patch_size

        # 初始化mask，大小为[B, grid_size, grid_size]
        bboxes_masks = torch.zeros(B, grid_size, grid_size, dtype=torch.bool, device=bboxes.device)

        # 获取所有bbox的归一化坐标
        x1, y1, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        # 将归一化坐标转换为像素坐标
        x1_pixel = x1 * W
        y1_pixel = y1 * H
        w_pixel = w * W
        h_pixel = h * H
        x2_pixel = x1_pixel + w_pixel
        y2_pixel = y1_pixel + h_pixel

        # 计算patch的索引
        patch_x1 = (x1_pixel // patch_size).long()
        patch_y1 = (y1_pixel // patch_size).long()
        patch_x2 = (x2_pixel // patch_size).long()
        patch_y2 = (y2_pixel // patch_size).long()

        # 手动限制索引在有效范围内 (0 - grid_size - 1)
        patch_x1 = torch.clamp(patch_x1, 0, grid_size - 1)
        patch_y1 = torch.clamp(patch_y1, 0, grid_size - 1)
        patch_x2 = torch.clamp(patch_x2, 0, grid_size - 1)
        patch_y2 = torch.clamp(patch_y2, 0, grid_size - 1)

        # 使用广播将bbox位置标记在mask中
        for b in range(B):
            bboxes_masks[b, patch_y1[b]:patch_y2[b] + 1, patch_x1[b]:patch_x2[b] + 1] = True

        return bboxes_masks 

    def calculate_p_mean(self,classification_confidences):
        """
        计算改进的动态阈值 p_mean。
        
        参数:
        classification_confidences (list): 每一帧的分类置信度列表。
        
        返回:
        float: 计算得到的改进动态阈值 p_mean。
        """
        n = len(classification_confidences)
        p_mean = 0.0
        
        for m in range(1, n + 1):
            # 计算从第1帧到第m帧的置信度之和
            sum_confidences = sum(classification_confidences[:m])
            # 计算加权和
            p_mean += sum_confidences / m
        mean = sum(classification_confidences)/n
        # 计算平均值
        p_mean /= n
        return p_mean,mean 


def get_tracker_class():
    return BATTrack
