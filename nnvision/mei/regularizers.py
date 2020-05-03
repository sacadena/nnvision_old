from featurevis.ops import ChangeStd, GaussianBlur
from .utility import cumstom_initial_guess
from functools import partial
from featurevis import ops, utils


# Contrast
postup_contrast_01 = ChangeStd(0.1)
postup_contrast_1 = ChangeStd(1)
postup_contrast_125 = ChangeStd(12.5)
postup_contrast_100 = ChangeStd(10)
postup_contrast_5 = ChangeStd(5)

# Blurring
Blur_sigma_1 = GaussianBlur(1)

# Initial Guess
rgb_initial_guess = partial(cumstom_initial_guess, mean=111, std=60)

# DiCarlo (Bashivan):

dc_transform = ops.Jitter(max_jitter=(2, 4))
dc_regularization = ops.TotalVariation(weight=0.001)
dc_gradient = utils.Compose([ops.ChangeNorm(1), ops.ClipRange(-1, 1)])


# Walker:

walker_gradient = utils.Compose([ops.FourierSmoothing(0.04), # not exactly the same as fft_smooth(precond=0.1) but close
                                 ops.DivideByMeanOfAbsolute(),
                                 ops.MultiplyBy(1/850, decay_factor=(1/850 - 1/20400) /(1-1000))])  # decays from 1/850 to 1/20400 in 1000 iterations
bias, scale = 111.28329467773438, 60.922306060791016
walker_postup = utils.Compose([ops.ClipRange(-bias / scale, (255 - bias) / scale), 
                               ops.GaussianBlur(1.5, decay_factor=(1.5 - 0.01) /(1-1000))]) # decays from 1.5 to 0.01 in 1000 iterations

