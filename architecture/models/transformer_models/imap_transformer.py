import numpy as np

# TODO:
# how to get rgb, depth, and agent pose data from input_sensors?
# filter CHORES dataset to only include OBJNAV tasks: see get_dataloader in train_pl.py
# create preproc (called by build_model) to create observation embedding (self.obs_pos_layer)
    ## use CLIP Resnet50 to encode rgb data
    ## use Resnet50 pretrained in PointNav to encode depth data
    ## see GPSCompassEmbedding for how to encode agent pose
if self.model_config.STATE_ENCODER.add_pos_attn:
    obs_pos_embeds = self.obs_pos_layer(batch['gps'], batch['compass'])   

    

class ImapEmbedding(nn.Module):
    def __init__(self, model_config) -> None: # model_config contains imap_size, hidden_size
        super().__init__()
        self.imap_size = model_config.imap_size
        # imap consists of imap_size x imap_size embeddings of size hidden_size
        
        # lookup table
        self.imap_token_embedding = nn.Embedding(self.imap_size**2, model_config.hidden_size)
        
        # tensor of 2D coordinates for imap_size x imap_size square centered around (0,0) (flattened into a single vector of coordinates so we can pass
        # it into the fully connected network)
        self.imap_pos_fts = self._create_imap_pos_features(self.imap_size) 

        # FFN from equation 1
        self.imap_pos_layer = nn.Sequential(
            nn.Linear(2, model_config.hidden_size), # (location: x, y)
            nn.LayerNorm(model_config.hidden_size)
        )
            
        self.ft_fusion_layer = nn.Sequential(
            nn.LayerNorm(model_config.hidden_size),
            nn.Dropout(model_config.dropout_rate),
        )

    def _create_imap_pos_features(self, imap_size):
        x, y = torch.meshgrid(torch.arange(imap_size), torch.arange(imap_size))
        xy = torch.stack([x, y], dim=2)
        xy = (xy + 0.5 - imap_size / 2).float() # relative distance to the center
        xy = xy.view(-1, 2)
        return xy

    def forward(self, batch_size):
        '''Get the initialized imap embedding'''
        device = self.imap_token_embedding.weight.device

        # initialize embedding for each element in the imap using the lookup table
        token_types = torch.arange(self.imap_size**2, dtype=torch.long, device=device)
        embeds = self.imap_token_embedding(token_types)

        # create embedding to capture spatial relationship between position within the imap using equation 1
        pos_embeds = self.imap_pos_layer(self.imap_pos_fts.to(device))

        embeds = embeds + pos_embeds
        embeds = self.ft_fusion_layer(embeds)

        # (imap_size**2, hidden_size) -> (batch_size, imap_size**2, hidden_size)
        embeds = einops.repeat(embeds, 'n d -> b n d', b=batch_size) 
        pos_embeds = einops.repeat(pos_embeds, 'n d -> b n d', b=batch_size)

        return embeds, pos_embeds 

class GpsCompassEmbedding(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.gps_layer = nn.Sequential(
            nn.Linear(2, hidden_size), # (location: x, y)
            nn.LayerNorm(hidden_size)
        )
        self.compass_layer = nn.Sequential(
            nn.Linear(2, hidden_size), # (heading: sin, cos)
            nn.LayerNorm(hidden_size)
        )

    def forward(self, gps, compass):
        gps_embeds = self.gps_layer(gps)
        compass_fts = torch.cat(
            [torch.cos(compass), torch.sin(compass)], -1,
        )
        compass_embeds = self.compass_layer(compass_fts)
        embeds = gps_embeds + compass_embeds
        return embeds


class IMapTransformerConfig():


class IMapTransformer(nn.Module):
    def __init__(
        self,
        cfg: IMapTransformerConfig,
    ):
        super().__init__()
        self.cfg = cfg

    def build_model(
        cls,
        model_version,
        input_sensors,
        loss,
        ckpt_pth=None,
    ):
        model = IMapTransformer(self.cfg) # main transformer
        self.obs_pos_layer = GpsCompassEmbedding(model_config.hidden_size) # for finding positional embeddings
        
        if ckpt_pth is not None:
            load_pl_ckpt(model, ckpt_pth)

        # implement preprocessor for processing input_sensors
        # fields that it needs:
        # 'rgb_features', 'depth_features', 'sem_features', 'compass', 'gps', 'infer_gps', 'infer_compass', 'infer_visual_features', 'step_ids' (what action is taken at each time step)
        infer_time_ids = np.array([
            np.random.randint(start_step, min(num_steps, t+max_future_step_size)) \
                for t in range(start_step, end_step)
        ])
        outs['infer_gps'] = outs['gps'][infer_time_ids]
        outs['infer_compass'] = outs['compass'][infer_time_ids]
        outs['infer_visual_features'] = outs['rgb_features'][infer_time_ids]

        return model, preproc

    # go through this
    def encode_step_obs_concat(self, batch, step_embeddings=None):
        batch_size, num_steps, _ = batch['gps'].size()

        x = []
        depth_embedding = self.depth_encoder(batch)
        if self.model_config.num_ft_views > 1: # (batch, nsteps, nviews, dim)
            depth_embedding = depth_embedding.view(batch_size, num_steps, -1)
        x.append(depth_embedding)

        if self.rgb_encoder is not None:
            rgb_embedding = self.rgb_encoder(batch)
            if len(rgb_embedding.size()) == 2:  # (batch x nsteps, dim)
                rgb_embedding = torch.split(rgb_embedding, batch['num_steps'], dim=0)
                rgb_embedding = pad_tensors_wgrad(rgb_embedding)
            if self.model_config.num_ft_views > 1: # (batch, nsteps, nviews, dim)
                rgb_embedding = rgb_embedding.view(batch_size, num_steps, -1)
            x.append(rgb_embedding)

        if self.sem_ft_layer is not None:
            sem_embedding = self.sem_ft_layer(batch['sem_features'])
            x.append(sem_embedding)

        if self.model_config.USE_SEMANTICS:
            # batch["semantic"]: (num_steps x num_envs, h, w, 1)
            if self.embed_sge:
                sge_embedding = self._extract_sge(batch)
                if len(sge_embedding.size()) == 2:
                    sge_embedding = torch.split(sge_embedding, batch['num_steps'], dim=0)
                    sge_embedding = pad_tensors_wgrad(sge_embedding)
                x.append(sge_embedding)

            batch['semantic'] = batch['semantic'].squeeze(dim=3)
            sem_seg_embedding = self.sem_seg_encoder(batch)
            if len(sem_seg_embedding.size()) == 2:  # (batch x nsteps, dim)
                sem_seg_embedding = torch.split(sem_seg_embedding, batch['num_steps'], dim=0)
                sem_seg_embedding = pad_tensors_wgrad(sem_seg_embedding)
            x.append(sem_seg_embedding)

        if self.model_config.USE_GPS:
            x.append(self.gps_embedding(batch['gps']))
        
        if self.model_config.USE_COMPASS:
            compass_observations = torch.concat(
                [torch.cos(batch['compass']), torch.sin(batch['compass'])],
                -1,
            )
            compass_embedding = self.compass_embedding(compass_observations)
            x.append(compass_embedding)

        if self.obj_category_to_embeds is None:
            obj_embedding = self.obj_categories_embedding(batch['objectgoal'])
        else:
            obj_embedding = torch.stack(
                [self.obj_category_to_embeds[k] for k in batch['object_category']], 0
            )
            obj_embedding = self.obj_categories_embedding(obj_embedding)

        x.append(
            einops.repeat(
                obj_embedding, 'b d -> b t d', t=num_steps
            )
        )

        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions = torch.zeros(batch_size, num_steps, dtype=torch.long).to(self.device)
            prev_actions[:, 0] = -1
            prev_actions[:, 1:] = batch['demonstration'][:, :-1]
            prev_actions[prev_actions == -100] = -1
            prev_actions_embedding = self.prev_action_embedding(
                prev_actions + 1
            )
            x.append(prev_actions_embedding)
        
        x = torch.cat(x, dim=2)
        if self.ft_fusion_layer is not None:
            x = self.ft_fusion_layer(x)

        if step_embeddings is not None:
            x = x + step_embeddings
        
        return x


    def forward(self, batch, compute_loss=False, return_imap_embeds=False):

        # initialize imap embeddings
        imap_embeds, imap_pos_embeds = self.imap_embedding() # edit param

        # create observation embeddings
        stepid_embeds = self.step_embedding(batch['step_ids'])
        inputs = self.encode_step_obs(batch, step_embeddings=stepid_embeds)
        input_embeds = inputs['fused_embeds']

        # positional embedding for o
        obs_pos_embeds = self.obs_pos_layer(batch['gps'], batch['compass'])  

        # setup query pose for visual feature prediction
        # format: 00...0000 (in place of rgb and depth data) | gps estimate | compass estimate 
        masked_input_embeds = torch.zeros_like(o_embeds)
        dim_vis = # depth width + rgb width
        masked_input_embeds[:, :, dim_vis: dim_vis+32] = self.gps_embedding(batch['infer_gps']) # replace this
        compass_fts = torch.concat(
            [torch.cos(batch['infer_compass']), torch.sin(batch['infer_compass'])],
            -1,
        )
        masked_input_embeds[:, :, dim_vis+32: dim_vis+64] = self.compass_embedding(compass_fts) # replace this

        # positional embedding for q
        masked_obs_pos_embeds = self.obs_pos_layer(batch['infer_gps'], batch['infer_compass'])

        # ground truth values for calculating loss
        target_visual_features = batch['infer_visual_features'] # target_visual_features[:, t] is the correct rgb values corresponding to q
        seq_visual_features = batch['rgb_features'] 
        if self.model_config.infer_depth_feature:
            seq_visual_features = torch.cat([seq_visual_features, batch['depth_features']], dim=-1)
        seq_visual_features = F.normalize(seq_visual_features, p=2, dim=-1) # rgb and depth values for every time step

        # attention masking trick for visual feature prediction
        ntokens = imap_embeds.size(1) + 2
        attn_masks = torch.zeros(ntokens, ntokens).bool().to(imap_embeds.device)
        attn_masks[:-1, -1] = True  # imap and obs tokens should not see the masked token
        attn_masks[-1, -2] = True   # the masked token should not see the current observation

        logits = []
        vis_pred_losses, map_pred_losses, sem_pred_losses = [], [], []

        
        for t in range(num_steps):
            # concatenate M_t, o_t, q_t for transformer input
            t_input_embeds = torch.cat(
                [imap_embeds, input_embeds[:, t:t+1], masked_input_embeds[:, t:t+1]], dim=1
            )

            # positional embedding for transformer input
            t_pos_embeds = torch.cat(
                [imap_pos_embeds, obs_pos_embeds[:, t:t+1], masked_obs_pos_embeds[:, t:t+1]], dim=1
            )

            # pass through transformer
            hiddens = self.state_encoder(
                t_input_embeds,
                pos=t_pos_embeds, mask=attn_masks,
            )

            # retrieve values from transformer output
            obs_hiddens = hiddens[:, -2] # o_hat
            masked_obs_hiddens = hiddens[:, -1] # q_hat
            imap_embeds = hiddens[:, :-2] # updated imap embeddings
            
            # pass q_hat through FFN to get final visual feature prediction
            masked_visual_preds = self.vis_pred_layer(masked_obs_hiddens)
            
            # NCE loss for visual feature prediction
            masked_visual_preds = F.normalize(masked_visual_preds, p=2, dim=-1)
            pos_sim_scores = torch.einsum(
                'bd,bd->b', F.normalize(target_visual_features[:, t], p=2, dim=-1), 
                masked_visual_preds
            )
            neg_sim_scores = torch.einsum(
                'btd,bd->bt', seq_visual_features, masked_visual_preds
            )
            neg_sim_scores.masked_fill_(batch['demonstration'] == -100, -float('inf'))
            sim_scores = torch.cat([pos_sim_scores.unsqueeze(1), neg_sim_scores], 1)
            sim_scores = sim_scores / 0.1
            vis_pred_losses.append(F.cross_entropy(
                sim_scores, 
                torch.zeros(sim_scores.size(0), dtype=torch.long, device=sim_scores.device),
                reduction='none'
            ))
                
            # explicit map construction loss
            pred_maps = self.map_pred_layer(imap_embeds).view(
                batch_size, self.model_config.pred_map_nchannels, -1
            )
            if self.model_config.infer_local_map_loss_type == 'mse':
                t_map_pred_loss = F.mse_loss(
                    pred_maps, batch['infer_local_maps'][:, t], reduction='none'
                ).view(batch_size, -1).mean(1)
            elif self.model_config.infer_local_map_loss_type == 'clf':
                t_map_pred_loss = F.binary_cross_entropy(
                    pred_maps, batch['infer_local_maps'][:, t], reduction='none'
                ).view(batch_size, -1).mean(1)
            map_pred_losses.append(t_map_pred_loss)

            # semantic prediction loss
            sem_input_hiddens = inputs[:, t, :self.model_config.DEPTH_ENCODER.output_size+self.model_config.RGB_ENCODER.output_size]
            pred_semantics = torch.sigmoid(self.sem_pred_layer(sem_input_hiddens))
            sem_labels = batch['sem_features'][:, t, :-1]
            sem_clf_loss = F.binary_cross_entropy(
                pred_semantics[:, :21], (sem_labels > 0).float(), reduction='none'
            ).mean(1)
            sem_mse_loss = F.mse_loss(
                pred_semantics[:, 21:], batch['sem_features'][:, t], reduction='none'
            ).mean(1)
            sem_pred_losses.append(sem_clf_loss + sem_mse_loss)

            #???
            add_objgoal = self.model_config.get('encoder_add_objgoal', True)
            if not add_objgoal:
                obs_hiddens = obs_hiddens + inputs['objectgoal']
            step_logits = self.action_distribution(obs_hiddens)
            logits.append(step_logits)

        logits = torch.stack(logits, 1)
        
        # primary task loss
        act_loss = self.compute_loss(logits, batch)
        loss_dict = {'overall': act_loss, 'action_loss': act_loss}
        
        # visual feature prediction loss
        vis_pred_losses = torch.stack(vis_pred_losses, 1) # (batch, nsteps)
        vis_pred_masks = (batch['inflection_weight'] > 0).float()
        vis_pred_loss = torch.mean(
            torch.sum(vis_pred_losses * vis_pred_masks, 1) / \
                torch.sum(vis_pred_masks, 1)
        )
        loss_dict['overall'] = loss_dict['overall'] + vis_pred_loss * self.model_config.infer_visual_feature_loss
        loss_dict['vis_pred_loss'] = vis_pred_loss

        # explicit map construction loss
        map_pred_masks = (batch['inflection_weight'] > 0).float()
        map_pred_losses = torch.stack(map_pred_losses, 1) # (batch, nsteps)
        map_pred_loss = torch.mean(
            torch.sum(map_pred_losses * map_pred_masks, 1) / \
                torch.sum(map_pred_masks, 1)
        )
        loss_dict['overall'] = loss_dict['overall'] + map_pred_loss * self.model_config.infer_local_map_loss
        loss_dict['map_pred_loss'] = map_pred_loss

        # semantic prediction loss
        sem_pred_losses = torch.stack(sem_pred_losses, 1)
        sem_pred_masks = (batch['inflection_weight'] > 0).float()
        sem_pred_loss = torch.mean(
            torch.sum(sem_pred_losses * sem_pred_masks, 1) / torch.sum(sem_pred_masks, 1)
        )
        loss_dict['overall'] = loss_dict['overall'] + sem_pred_loss * self.model_config.infer_sem_label_loss
        loss_dict['sem_pred_loss'] = sem_pred_loss

        return loss_dict, logits



class IMapTransformerAgent(AbstractAgent):

