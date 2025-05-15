import numpy as np

class IMapTransformerConfig:
    

class ImapEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # imap consists of imap_size x imap_size embeddings of size hidden_size
        
        # lookup table
        self.imap_token_embedding = nn.Embedding(map_cfg.imap_size**2, model_config.hidden_size)
        
        # tensor of 2D coordinates for imap_size x imap_size square centered around (0,0) (flattened into a single vector of coordinates so we can pass
        # it into the fully connected network)
        self.imap_pos_fts = self._create_imap_pos_features(map_cfg.imap_size) 

        # FFN from equation 1
        self.imap_pos_layer = nn.Sequential(
            nn.Linear(2, model_config.hidden_size), # (location: x, y)
            nn.LayerNorm(model_config.hidden_size)
        )
            
        self.ft_fusion_layer = nn.Sequential(
            nn.LayerNorm(model_config.hidden_size),
            nn.Dropout(model_config.dropout_rate),
        )

        self.imap_token_type = map_cfg.token_embed_type
        self.imap_size = map_cfg.imap_size
        self.encode_position = map_cfg.encode_position

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



class IMapTransformer(nn.Module):
    def __init__(
        self,
        cfg: IMapTransformerConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # filter CHORES dataset to only include OBJNAV tasks

        # combine data to create observation feature (self.obs_pos_layer)
            ## use CLIP Resnet50 to encode rgb data
            ## use Resnet50 pretrained in PointNav to encode depth data


    def forward(self, batch):
        # initialize imap embeddings
        imap_embeds, imap_pos_embeds = self.imap_embedding(batch_size)

        # set up q and attention mask for visual feature prediction


        for t in range(num_steps):
            # pass through transformer
            hiddens = self.state_encoder(
                torch.cat([t_input_embeds, masked_input_embeds[:, t:t+1]], dim=1),
                pos=t_pos_embeds, mask=attn_masks,
            )

            # retrieve values from transformer output
            obs_hiddens = hiddens[:, -2] # o_hat
            masked_obs_hiddens = hiddens[:, -1] # q_hat
            imap_embeds = hiddens[:, :-2] # updated imap embeddings

            # (batch_size, dim)
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
            # print(t, sim_scores)
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
