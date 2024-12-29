from typing import Tuple, Union, Optional, List

from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from torchvision import transforms

transform = transforms.Lambda(lambda t: (t + 1) / 2)


class LGDLoss():
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = int = 256
        self.patch_size: List[int] = [1, 2]

    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        # bs, c,res,res = ref_noise.shape
        # #res = int(np.sqrt(res2))

        # ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2) 
        # trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)
        ref_noise_reshape = ref_noise
        trg_noise_reshape = trg_noise
        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2) 
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) 
            
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean() 
        return loss*0.06

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=2, nce_T = 0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()
        
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) 

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim) 
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))

        return loss
    




    def to_patches(sefl, data, kernel_size):

        patches = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)(torch.mean(data, dim=1, keepdim=True))
        patches = patches.transpose(2,1)

        return patches


    def calcu_kmeans(self, data, num_clusters):

        [b, h, w] = data.shape
        cluster_ids_all = np.empty([b, h])
        cluster_ids_all = torch.from_numpy(cluster_ids_all)
        for i in range(b):
            # cluster_ids, cluster_centers = kmeans(
            #     X=data[i,:,:], num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
            # )

            # DBSCAN
            # model = DBSCAN(eps=5)
            # cluster_ids = model.fit_predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # MeanShift
            # model = MeanShift()
            # cluster_ids = model.fit_predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # Spectral Clustering
            # model = SpectralClustering(n_clusters=num_clusters)
            # cluster_ids = model.fit_predict(transform(data[i,:,:].cpu()))
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # Hierarchical Clustering
            model = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_ids = model.fit_predict(transform(data[i,:,:].cpu()))
            cluster_ids = torch.from_numpy(cluster_ids).cuda()

            # # gmm
            # model = GaussianMixture(n_components=num_clusters)
            # model.fit(data[i,:,:].cpu())
            # cluster_ids = model.predict(data[i,:,:].cpu())
            # cluster_ids = torch.from_numpy(cluster_ids).cuda()
            # print(cluster_ids)

            # # kmeans
            # km = kmeans_core(k=num_clusters,data_array=data[i,:,:].cpu().numpy(),batch_size=400,epochs=1000)
            # km.run()
            # cluster_ids = km.idx

            # print(cluster_ids)
            cluster_ids_all[i, :] = cluster_ids
        
        return cluster_ids_all

    def calcu_svd(self, data):

        u, sv, v = torch.svd(data)
        #sv_F2 = torch.norm(sv, dim=1)
        #sv_F2 = sv_F2.unsqueeze(1)
        #normalized_sv = sv / sv_F2

        return sv

    def calcu_svd_distance(self, data1, data2, cluster_ids, num_clusters):

        [b, h, w] = data1.shape 
        sv_ab_dis = np.empty([b, num_clusters])
        sv_ab_dis = torch.from_numpy(sv_ab_dis)
        for i in range(num_clusters):

            indices = (cluster_ids[0] ==i).nonzero(as_tuple=True)[0]
            
            if len(indices)==0:
                sv_ab_dis[:, i] = 1e-5
            else:
                data1_select = torch.index_select(data1, 1, indices.cuda())
                data2_select = torch.index_select(data2, 1, indices.cuda())
                sv1 = self.calcu_svd(data1_select.cpu())
                sv2 = self.calcu_svd(data2_select.cpu())
   
                sv_ab_dis_i = torch.abs(sv1 - sv2)
                sv_ab_dis[:, i] = torch.sum(sv_ab_dis_i, dim=1)
        return sv_ab_dis
