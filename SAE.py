from NNBP import NNBP
from AE import AE



class SAE:
    def __init__(self, nnbp_params, ae_params ):
        x, N_x, y, N_y, n_hl, g, o, rng, initParamsFun = nnbp_params
        theano_rng, ae_g, ae_o = ae_params

        self.nnbp = NNBP(x, N_x, y, N_y, n_hl, g, o, rng, initParamsFun)

        self.aes = []
        for ilayer in xrange(self.nnbp.L):
            ae__x       = self.nnbp.hiddenLayers[ilayer].h_p
            ae__N_x     = self.nnbp.hiddenLayers[ilayer].N_h_p
            ae__N_hl    = self.nnbp.hiddenLayers[ilayer].N_h
            ae__g       = ae_g
            ae__o       = ae_o
            ae__rng     = rng
            ae__theano_rng  = theano_rng
            ae__initParamsFun = initParamsFun
            ae__W_hl    = self.nnbp.hiddenLayers[ilayer].W
            ae__b_hl    = self.nnbp.hiddenLayers[ilayer].b
            ae__b_ol    = None
            self.aes.append(
                AE(ae__x, ae__N_x, ae__N_hl, ae__g, ae__o, ae__rng,
                   ae__theano_rng, ae__initParamsFun, ae__W_hl,
                   ae__b_hl, ae__b_ol))

    def compile__pt(self, x_image_global, examples, ib, B, K, corrupt):
        preTrainFuns = []
        for ilayer in xrange(self.nnbp.L):
            preTrainFuns.append(
                self.aes[ilayer].compileFunctions(
                    x_image_global, examples, ib, B,
                    K, corrupt[ilayer])  )
        return preTrainFuns

    def compile__ft(self, shr, ib, B, K):
        dataset = {"train": shr["ft_tr"], "valid": shr["ft_v"], "test": shr["te"]}
        return self.nnbp.compileFunctions(dataset, ib, B, K)

