import pdb
import torch.nn as nn
from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from lib.train.admin import multigpu

from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np

class BATActor(BaseActor):
    """ Actor for training BAT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        # 定义损失函数
        self.criterion = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss 会自动处理 Sigmoid 激活

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # print('data',data)
        # forward pass
        out_dict = self.forward_pass(data)
        # print("out_dict",out_dict)
        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status
    
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



    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['template_images']) == 1
        # assert len(data['search_images']) == 1

        # seq_name = data['seq_name'][0]
        # print('seq_name',seq_name)
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 6, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 6, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        self.epoch = data['epoch']
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
            # ce_keep_rate = 0.7

        if len(template_list) == 1:
            template_list = template_list[0]
        # print('ce_keep_rate',ce_keep_rate) 

        B,C,H_S,W_S = search_img.size()
        search_bboxes = torch.tensor(data['search_anno'][-1], dtype=torch.float32, device='cuda')
        self.search_bboxes_masks = self.get_bboxes_masks(search_bboxes, B, H_S, W_S)
        #torch.Size([32, 256])
        self.search_bboxes_masks = self.search_bboxes_masks.view(B, -1)
        self.search_masks = torch.where(self.search_bboxes_masks, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda')).to('cuda')
        
        
        B,C,H_T,W_T = template_img_i.size()
        template_bboxes = torch.tensor(data['template_anno'][-1], dtype=torch.float32, device='cuda')  # 假设模板目标的bbox
        template_bboxes_masks = self.get_bboxes_masks(template_bboxes, B, H_T, W_T)
        self.template_bboxes_masks = template_bboxes_masks.view(B, -1)     
        self.template_masks = torch.where(self.template_bboxes_masks, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda')).to('cuda')  

        if len(template_list) == 2:
            template_bboxes = torch.tensor(data['template_anno'][0], dtype=torch.float32, device='cuda')  # 假设模板目标的bbox
            template_bboxes_masks = self.get_bboxes_masks(template_bboxes, B, H_T, W_T)
            self.template_bboxes_masks = template_bboxes_masks.view(B, -1)     
            self.template_masks2 = torch.where(self.template_bboxes_masks, torch.tensor(1).to('cuda'), torch.tensor(0).to('cuda')).to('cuda') 
            search_mask_list = []
            for i in range(B):
                search_mask_list.append(torch.cat([self.search_masks[i].unsqueeze(0),self.search_masks[i].unsqueeze(0)],dim=0))
            self.search_masks = torch.cat(search_mask_list,dim=0)


        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            # seq_name =seq_name,
                            template_masks = self.template_masks,
                            )

        return out_dict

    def compute_losses(self, pred_dict, gt_dict,return_status=True):

         

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        B,_ = gt_bbox.shape
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)

        gt_bbox_list,gt_gaussian_maps_list  = [],[]
        for i in range(B):
            gt_bbox_list.append(torch.cat([gt_bbox[i].unsqueeze(0),gt_bbox[i].unsqueeze(0)],dim=0))
            gt_gaussian_maps_list.append(torch.cat([gt_gaussian_maps[i].unsqueeze(0),gt_gaussian_maps[i].unsqueeze(0)],dim=0))
        gt_bbox = torch.cat(gt_bbox_list,dim=0)
        gt_gaussian_maps = torch.cat(gt_gaussian_maps_list,dim=0)

        futrue_loss = pred_dict['future_loss'].mean()*0.1
        parameters = pred_dict['weights']

        total_loss = 0
        sigma = 0.1*(self.epoch/30)
        y = self.search_masks
        y_n = 1 - y
        num = len(parameters)
        for i in range(num):
            weight = (parameters[i].float().squeeze(1)-1)*y + 1
            weight2 = (parameters[i].float().squeeze(1)-1)*y_n + 1
            # loss1 = torch.sum(torch.exp(-(weight-1)), dim=1, keepdim=True)/256
            # loss2 = torch.sum((weight2-1)**2, dim=1, keepdim=True)/256 
            # mseloss = - torch.log(loss2/(loss1+loss2))
            loss1 = torch.sum(torch.exp(-(weight-1)), dim=1, keepdim=True)/256
            loss2 = torch.sum(torch.exp(-(weight2-1)), dim=1, keepdim=True)/256 
            mseloss = - torch.log(loss2/(loss1+loss2))            
            # weight = weights[i].float().squeeze(1)
            # mseloss = torch.sum(torch.exp(-(weight-1))**2, dim=1, keepdim=True)/256 + torch.sum((weight2-1)**2, dim=1, keepdim=True)/256 
            
            # KL_divergence = F.kl_div(weight.log(),Q,reduction='batchmean')
            # KL_divergence = self.kl_divergence(weight,256,device)
            # loss = self.criterion(weight,y.float())
            total_loss = total_loss + mseloss
        uncertainty_loss = total_loss / num
        uncertainty_loss = sigma*uncertainty_loss.mean()


       

    

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec= box_cxcywh_to_xyxy(pred_boxes)
        pred_boxes_vec = pred_boxes_vec.view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)
        gt_boxes_vec = gt_boxes_vec[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        
        # emd_loss = self.emd_loss(gt_boxes_vec,pred_boxes_vec)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        #计算一对模板iou差值 获得伪标签
        label_list = []
        for i in range(0,B*2,2):
            t1,t2 = iou[i],iou[i+1]
            if t1>t2:
                label = torch.tensor([1,0]).unsqueeze(0)
            elif t1 == t2:
                label = torch.tensor([1,1]).unsqueeze(0)
            else:
                label = torch.tensor([0,1]).unsqueeze(0)
            label_list.append(label)
        labels = torch.cat(label_list,dim=0)
        scores = pred_dict["relative_score"]
        templaterouter = 0.01*self.criterion(scores.to('cuda'),labels.float().to('cuda'))

        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+ uncertainty_loss + total_kl_loss + stage_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+ uncertainty_loss + templaterouter
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+ uncertainty_loss + templaterouter+ futrue_loss
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss+0.2*pred_dict['ce_loss']
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item(),
                      "u_m":uncertainty_loss.item(),
                      "templaterouter":templaterouter.item(),
                      "furuteLoss":futrue_loss.item(),
                      }
            return loss, status
        else:
            return loss


    
    # 定义EMD损失函数
    def emd_loss(self,gt_track,pred_track):
        device = torch.device('cuda:0')
        gt_centroids = (gt_track[:, :2] + gt_track[:, 2:]) / 2
        pred_centeroids = (pred_track[:, :2] + pred_track[:, 2:]) / 2
        sequence = torch.arange(1, 33, 1).to(device) #构建时间序列
        # 合并 x, y 坐标和时序，构建三维坐标
        true_3d_coords = torch.cat((gt_centroids, sequence.unsqueeze(1)), dim=1)
        predicted_3d_coords = torch.cat((pred_centeroids, sequence.unsqueeze(1)), dim=1)

        # 确保输入是NumPy数组
        predicted_points_np = predicted_3d_coords.cpu().detach().numpy()
        true_points_np = true_3d_coords.cpu().detach().numpy()

        x,y,z = predicted_points_np[:, 0],predicted_points_np[:, 1],predicted_points_np[:, 2]
        x2,y2,z2 = true_points_np[:, 0],true_points_np[:, 1],true_points_np[:, 2]

        distance_matrix = np.sqrt(np.power(x-x2,2)+np.power(y-y2,2)+np.power(z-z2,2))

        emd = torch.from_numpy(distance_matrix).to(device).mean() * 0.1
        # print('emd_torch',emd)
        return emd

    def Uncertainy_Location(self,sigma_w,sigma_h,gw,gh,w,h):
        w_loss = self.gaussian_dist_pdf(gw,w,sigma_w)+1e-9
        h_loss = self.gaussian_dist_pdf(gh,h,sigma_h)+1e-9

        return 0.05*(w_loss + h_loss)


    def gaussian_dist_pdf(self,val, mean, var):
        return torch.exp((-var))*(val - mean) ** 2.0 + var / 2


    def kl_divergence(self,alpha, num_classes, device):
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl


        
