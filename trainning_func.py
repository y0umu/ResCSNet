import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def get_evaluation(model, loader_val, loader_subtrain, device, fn_mse, fn_cwssim, mse_weight=0.3, cwssim_weight=0.7, 
                   print_every=10):
    '''
    Evaluate the subtrain set and validation set, returns
    subtrain_psnr, val_psnr, subtrain_mix_psnr, val_mix_psnr
    '''
    model.eval()  # ensure the model is in evaluation mode
    subtrain_avg_psnr = 0.0
    subtrain_avg_mse = 0.0
    val_avg_psnr = 0.0
    val_avg_mse = 0.0
    
    fn_mse = fn_mse.to(device=device)
    fn_cwssim = fn_cwssim.to(device=device)
    subtrain_avg_cwssim_loss = 0.0
    val_avg_cwssim_loss = 0.0
    
    with torch.no_grad():
        for t, val_im in enumerate(loader_val):
            if t % int(print_every) == 0:
                logging.info(f"checked {t}/{len(loader_val)} in loader_val")
            val_original = val_im.to(device)
            val_recovered = model(val_original)
            val_mse = fn_mse(val_recovered, val_original)
            val_avg_mse += val_mse
            # PSNR
            val_psnr = -10 * torch.log10(val_mse)
            val_avg_psnr += val_psnr
            # CW-SSIM
            val_original = val_im.to(device)
            val_recovered = model(val_original)
            val_cwssim_loss = 1 - fn_cwssim(val_recovered, val_original)
            val_avg_cwssim_loss += val_cwssim_loss
            
        val_avg_mse /= len(loader_val)
        val_avg_psnr /= len(loader_val)
        val_avg_cwssim_loss /= len(loader_val)

        for t, subtrain_im in enumerate(loader_subtrain):
            if t % int(print_every) == 0:
                logging.info(f"checked {t}/{len(loader_subtrain)} in loader_subtrain")
            subtrain_original = subtrain_im.to(device)
            subtrain_recovered = model(subtrain_original)
            subtrain_mse = fn_mse(subtrain_recovered, subtrain_original)
            subtrain_avg_mse += subtrain_mse
            # PSNR
            subtrain_psnr = -10 * torch.log10(subtrain_mse)
            subtrain_avg_psnr += subtrain_psnr
            # CW-SSIM
            subtrain_original = subtrain_im.to(device)
            subtrain_recovered = model(subtrain_original)
            subtrain_cwssim_loss = 1 - fn_cwssim(subtrain_recovered, subtrain_original)
            subtrain_avg_cwssim_loss += subtrain_cwssim_loss

        subtrain_avg_mse /= len(loader_subtrain)
        subtrain_avg_psnr /= len(loader_subtrain)
        subtrain_avg_cwssim_loss /= len(loader_subtrain)
    
    subtrain_psnr = subtrain_avg_psnr
    val_psnr = val_avg_psnr
    subtrain_mix_psnr = -10 * torch.log10(mse_weight*subtrain_avg_mse + cwssim_weight*subtrain_avg_cwssim_loss)
    val_mix_psnr = -10 * torch.log10(mse_weight*val_avg_mse + cwssim_weight*val_avg_cwssim_loss)
    return subtrain_psnr, val_psnr, subtrain_mix_psnr, val_mix_psnr