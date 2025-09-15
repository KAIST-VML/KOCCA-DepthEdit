from json import decoder
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import inference.optimizer as optimizer

def get_predictor(net, device,
                  prob_thresh=0.49,
                  with_flip=True,
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):
    
    lbfgs_params_ = {
        'm': 20,
        'factr': 0,
        'pgtol': 1e-8,
        'maxfun': 20,
    }

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if lbfgs_params is not None:
        lbfgs_params_.update(lbfgs_params)
    lbfgs_params_['maxiter'] = 2 * lbfgs_params_['maxfun']

    if brs_opt_func_params is None:
        brs_opt_func_params = dict()


    predictor_params_.update({
        'net_clicks_limit': 8,
    })
    if predictor_params is not None:
        predictor_params_.update(predictor_params)

    insertion_mode = 'after_deeplab'

    opt_functor = optimizer.ScaleBiasOptimizer(prob_thresh=prob_thresh,
                                        with_flip=with_flip,
                                        optimizer_params=lbfgs_params_,
                                        **brs_opt_func_params)

    predictor = FeatureBRSPredictor(net, device,
                                    opt_functor=opt_functor,
                                    with_flip=with_flip,
                                    insertion_mode=insertion_mode,
                                    **predictor_params_)

    return predictor

class BasePredictor(object):
    def __init__(self, net, device, opt_functor,
                 net_clicks_limit=None,
                 with_flip=False,
                 max_size=None,
                 **kwargs):
        self.net = net
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.opt_data = None
        self.input_data = None
        self.opt_functor = opt_functor

    def set_input_image(self, image_nd):
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)

        self.rgb = self.original_image[:,:3]
        self.seg = self.original_image[:,3:]

    def get_prediction(self, guide):
        prediction = self._get_prediction(self.rgb, guide)

        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1)]

            neg_clicks = [click.coords for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {'transform_states': self._get_transform_states()}

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])

class FeatureBRSPredictor(BasePredictor):
    def __init__(self, model, device, opt_functor, insertion_mode='after_deeplab', **kwargs):
        super().__init__(model, device, opt_functor=opt_functor, **kwargs)
        self.insertion_mode = insertion_mode
       

    def _get_prediction(self, image_nd, guide):
        bs = image_nd.shape[0]

        '''actual feature from model'''
        if self.net.opt.input_ch == 3:
            with torch.no_grad():
                x = self.net.conv(torch.cat((image_nd, guide), dim = 1))
        else:
            x = torch.cat((image_nd, guide), dim = 1)

        self.input_data= self._get_head_input(x)
        self.num_channels = self.input_data[-1].shape[1]
        '''initial feature to optimize (set to zero)'''
        if self.opt_data is None or self.opt_data.shape[0] // (2 * self.num_channels) != bs:
            self.opt_data = np.zeros((bs * 2 * self.num_channels), dtype=np.float32)


        def get_prediction_logits(scale, bias):
            scale = scale.view(bs, -1, 1, 1)
            bias = bias.view(bs, -1, 1, 1)

            scaled_backbone_features = self.input_data[-1] * scale
            scaled_backbone_features = scaled_backbone_features + bias
            
            self.input_data[-1] = scaled_backbone_features
            if self.location == "after_encoder":
                pred = self.net.decoder(self.input_data, None, self.seg)
            elif self.location == 'before_up3':
                x = self.input_data[-1]
                x = self.net.decoder.up3(x) # C:24
                '''spade normalization'''
                x = F.leaky_relu(self.net.decoder.spade_norm3(x, self.seg), 2e-1) # C: 24*3
                x = self.net.decoder.up4(x) # C:32
                x = self.net.decoder.up5(x)

                '''spade normalization'''
                x = F.leaky_relu(self.net.decoder.spade_norm5(x, self.seg), 2e-1) # C: 24*3
                '''-------------------'''
                pred = self.net.decoder.last(self.net.decoder.act(self.net.decoder.last_(x)))
            pred = torch.nn.functional.threshold(input=pred, threshold=0.0, value=0.0)

            return pred

        self.opt_functor.init_click(get_prediction_logits, guide, self.device) 
      
        opt_result = fmin_l_bfgs_b(func=self.opt_functor, x0=self.opt_data,
                                    **self.opt_functor.optimizer_params)
        self.opt_data = opt_result[0]

        with torch.no_grad():
            if self.opt_functor.best_prediction is not None:
                opt_pred = self.opt_functor.best_prediction
            else:
                opt_data_nd = torch.from_numpy(self.opt_data).to(self.device)
                opt_vars, _ = self.opt_functor.unpack_opt_params(opt_data_nd)
                opt_pred = get_prediction_logits(*opt_vars)

        return opt_pred

    def _get_head_input(self, input):
        if self.location == "after_encoder":
            with torch.no_grad():
                features = self.net.encoder(input)
        elif self.location == "before_up3":
            with torch.no_grad():
                features = self.net.encoder(input)
                x = self.net.decoder.up1(features[8]) # C:64
            
                x = F.leaky_relu(self.net.decoder.spade_norm1(x, self.seg), 2e-1) # C: 24*3
                
                x = torch.cat((x, features[4]), dim=1) # C:64*2
                # x = self.net.decoder.att1(x)
                
                x = self.net.decoder.up2(x) # C:32
                x = torch.cat((x, features[3]), dim=1) # C:32*2
                # x = self.net.decoder.att2(x)
                features[-1]= x
        return features
